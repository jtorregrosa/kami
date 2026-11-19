"""What a profile needs before a pod is billed: GPU memory, the weights' download size,
and access to every Hugging Face repo the pipelines load (several are gated).

Everything comes from the Hugging Face API, so no checks need a download. A check
that cannot be done (offline) never stops `kami up`: only a definite answer does.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from . import USER_AGENT
from . import catalog as gpu_table   # `catalog` is the provider's list below

# Files the pipelines load: diffusers and transformers keep safetensors (some older
# repos .bin / .pth), TRELLIS.2 its checkpoints as safetensors plus json configs.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt")


@dataclass
class Fit:
    gpu: str
    vram_gb: float | None   # None: the GPU is in neither the provider's catalog nor kami's table
    needed_gb: int          # the profile's min_vram_gb (0 = no requirement)

    @property
    def fits(self) -> bool:
        return not self.needed_gb or self.vram_gb is None or self.vram_gb >= self.needed_gb


def fits(profile, catalog: list[dict]) -> list[Fit]:
    """One Fit per GPU of the profile, in order: does its memory (pooled over gpu_count)
    reach the profile's min_vram_gb? GPUs are matched whatever their spelling."""
    vram = {gpu_table.key(g["id"]): g.get("memory") for g in catalog}
    count = max(1, getattr(profile, "gpu_count", 1))
    result = []
    for gpu in profile.gpus:
        gb = vram.get(gpu_table.key(gpu)) or gpu_table.vram_gb(gpu)
        result.append(Fit(gpu, gb * count if gb else None, profile.min_vram_gb))
    return result


# --- download size (what `kami prices` bills as transfer) ---

HF_FILES = "https://huggingface.co/api/models/{}?blobs=true"
HF_RESOLVE = "https://huggingface.co/{}/resolve/main/{}"


class HubError(LookupError):
    """Hugging Face answered with an HTTP error (`status`)."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return headers


@lru_cache(maxsize=None)
def repo_files(repo: str, timeout: float = 10) -> tuple[tuple[str, int], ...]:
    """(path, bytes) of every file in a Hugging Face repo. Raises LookupError (HubError
    when Hugging Face answers with an error)."""
    url = HF_FILES.format(urllib.parse.quote(repo, safe="/"))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_hf_headers()),
                                    timeout=timeout) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise HubError(f"Hugging Face answered {e.code} for {repo}", e.code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
        raise LookupError(f"could not reach Hugging Face ({e})") from None
    return tuple((s["rfilename"], s.get("size") or 0) for s in info.get("siblings", []))


def weight_files(files) -> list[tuple[str, int]]:
    """The weight files out of a repo's `files`. A repo with safetensors is loaded from
    them alone, so .bin / .pth copies of the same weights are not counted."""
    safetensors = [f for f in files if f[0].endswith(".safetensors")]
    return safetensors or [f for f in files if f[0].endswith(WEIGHT_SUFFIXES)]


def split_repo(entry: str) -> tuple[str, str]:
    """("org/name", "folder") of a profile repo entry: "org/name/folder" stands for the
    part of a repo under that folder (Hunyuan3D-2 keeps several models in one repo), or
    for one file of it ("org/name/lora.safetensors": a LoRA)."""
    parts = entry.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


def entry_files(entry: str, timeout: float | None = None) -> list[tuple[str, int]]:
    """The weight files of a profile repo entry (only under its folder, if it has one)."""
    repo, folder = split_repo(entry)
    files = repo_files(repo) if timeout is None else repo_files(repo, timeout)
    if folder:
        files = tuple(f for f in files if f[0] == folder or f[0].startswith(folder + "/"))
    return weight_files(files)


def download_gb(profile) -> float:
    """GB (10^9 bytes, how hosts bill transfer) the profile downloads on each boot.
    Raises LookupError when it cannot be known."""
    total = 0
    for entry in profile.repos:
        files = entry_files(entry)
        if not files:
            raise LookupError(f"no weight files found in {entry}")
        total += sum(size for _, size in files)
    return total / 1e9


# --- access to the weights (checked before a pod is created) ---

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None             # the redirect to the CDN is the answer: never fetch the file


def repo_problem(entry: str, timeout: float = 10) -> str | None:
    """Why the pod could not download `entry` (a repo, or "org/name/folder": private,
    missing, gated without access), or None when it can -- or when it cannot be told
    (offline)."""
    repo, _ = split_repo(entry)
    page = f"https://huggingface.co/{repo}"
    token = bool(os.environ.get("HF_TOKEN"))
    try:
        files = entry_files(entry, timeout)
    except HubError as e:
        if e.status in (401, 403, 404):
            return (f"{repo} not found on Hugging Face"
                    + (" (or HF_TOKEN cannot see it)" if token else
                       " (a private repo needs HF_TOKEN in .env)"))
        return None
    except LookupError:
        return None
    if not files:
        return None
    path = files[0][0]
    req = urllib.request.Request(HF_RESOLVE.format(repo, urllib.parse.quote(path)),
                                 method="HEAD", headers=_hf_headers())
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout).close()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return (f"{repo} is gated: " + (f"accept its terms at {page} with the account of HF_TOKEN"
                                            if token else f"accept its terms at {page} and put HF_TOKEN in .env"))
        if e.code == 404:
            return f"{path} is missing from {repo}"
        return None             # 3xx (the CDN) is fine; anything else cannot be told
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    return None


def access_problems(profile, timeout: float = 10) -> list[str]:
    """Every repo of the profile the pod could not download (see repo_problem)."""
    return [p for repo in profile.repos if (p := repo_problem(repo, timeout))]
