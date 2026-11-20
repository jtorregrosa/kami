"""Client side of kami-server's jobs (see server.py): submit, wait, download.

A job answers at once with an id; kami then polls it, so no request stays open longer
than the RunPod proxy allows (~100 s) however long the GPU works.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import USER_AGENT, ui
from .config import ConfigError

POLL_EVERY = 2                  # seconds between job status checks
JOB_TIMEOUT = 30 * 60           # give up waiting (the job keeps running on the pod)
REQUEST_TIMEOUT = 60
RETRIES = 3                     # transient network errors per request

# Image formats TRELLIS.2 (PIL) reads; anything else is refused before a job is sent.
# what kami-server reads (server.open_image)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif", ".tif", ".tiff",
                  ".bmp", ".gif")
MAX_IMAGE_BYTES = 48 * 1024 * 1024


class JobError(ConfigError):
    pass


def _request(url: str, api_key: str, body: dict | None = None, timeout: float = REQUEST_TIMEOUT):
    """(status, bytes) of a request to the server; retries transient network errors."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    last = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(1 + attempt * 2)
    raise JobError(f"Could not reach the pod: {last}\n`kami status` to check it.")


def _json(status: int, raw: bytes, what: str) -> dict:
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        body = {"error": raw[:200].decode(errors="replace")}
    if status >= 400:
        detail = body.get("error") if isinstance(body, dict) else None
        if status == 401:
            detail = "the pod refused kami's API key (a pod from another session?)"
        raise JobError(f"{what} failed: HTTP {status} {detail or ''}".rstrip())
    return body


def info(url: str, api_key: str) -> dict:
    """What the pod serves: {'mesh_model', 'image_model', 'mesh_defaults', ...}."""
    return _json(*_request(url + "/v1/info", api_key), "Asking the pod what it serves")


VIEW_NAMES = ("front", "left", "back", "right")     # Hunyuan3D-2mv's; TRELLIS.2 needs none
# the views `kami views` can make (the camera LoRAs' azimuths: 0 front, 90 right...)
VIEW_ANGLES = ("front", "front-right", "right", "back-right", "back", "back-left", "left",
               "front-left")


def view_arg(text: str) -> tuple[str | None, str]:
    """("front", "a.png") out of "front=a.png"; (None, text) for a plain path."""
    name, sep, path = text.partition("=")
    if sep and name.strip().lower() in VIEW_ANGLES and path:
        return name.strip().lower(), path
    return None, text


def read_image(path: Path) -> str:
    """The image as base64, after checking it is one kami can send."""
    if not path.is_file():
        raise JobError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise JobError(f"{path.name} is not an image kami can send "
                       f"({', '.join(IMAGE_SUFFIXES)})")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise JobError(f"{path.name} is {len(data) / 2**20:.0f} MiB: "
                       f"over {MAX_IMAGE_BYTES // 2**20} MiB, resize it first")
    return base64.b64encode(data).decode()


def submit(url: str, api_key: str, body: dict) -> dict:
    return _json(*_request(url + "/v1/jobs", api_key, body), "Sending the job")


def check(url: str, api_key: str, job_id: str) -> dict:
    """The job's current state on the pod (status, position, files, error)."""
    return _json(*_request(f"{url}/v1/jobs/{job_id}", api_key), "Checking the job")


def listing(url: str, api_key: str) -> dict | None:
    """Every job on the pod ({'jobs': [...], 'running', 'queued'}), or None when its
    server predates GET /v1/jobs."""
    status, raw = _request(url + "/v1/jobs", api_key)
    if status in (404, 405):
        return None
    return _json(status, raw, "Listing the pod's jobs")


STATUS_TEXT = {"queued": "Queued on the pod", "running": "Generating on the GPU"}


def wait(url: str, api_key: str, job: dict, label: str, sleep=time.sleep) -> dict:
    """Poll the job until it is done; the progress line shows its state and time."""
    progress = ui.Progress(f"{label}: queued")
    started = time.monotonic()
    try:
        while True:
            status = job.get("status")
            if status == "done":
                return job
            if status == "error":
                raise JobError(f"{label} failed on the pod: {job.get('error') or 'unknown error'}\n"
                               "`kami logs` shows the traceback.")
            text = STATUS_TEXT.get(status, status or "?")
            if status == "queued" and job.get("position"):
                text += f" (#{job['position']})"
            if status == "running" and job.get("stage_label"):
                text = job["stage_label"]           # servers with progress (see server.py)
            progress.phase(f"{label}: {text}")
            if status == "running" and isinstance(job.get("progress"), (int, float)):
                detail = f"step {job['step']}/{job['steps']}" if job.get("steps") else ""
                progress.bar(job["progress"], detail)
            if time.monotonic() - started > JOB_TIMEOUT:
                raise JobError(f"{label} is taking over {ui.duration(JOB_TIMEOUT)}\n"
                               f"It keeps running on the pod: `kami logs` to follow it.")
            for _ in range(POLL_EVERY * 4):
                progress.tick()
                sleep(0.25)
            job = _json(*_request(f"{url}/v1/jobs/{job['id']}", api_key), "Checking the job")
    finally:
        progress.stop()


def download(url: str, api_key: str, job: dict, out_dir: Path) -> list[Path]:
    """Save the job's files into out_dir (never overwriting: name-2.glb, name-3.glb...)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in job.get("files") or []:
        status, data = _request(f"{url}/v1/jobs/{job['id']}/files/{name}", api_key,
                                timeout=5 * REQUEST_TIMEOUT)
        if status >= 400:
            _json(status, data, f"Downloading {name}")
        path = free_path(out_dir / name)
        path.write_bytes(data)
        saved.append(path)
    return saved


# What is added to a prompt so the picture is of something a printer can make, and
# `kami print` can colour: it stands, it is one thick piece, and its colours are few and
# flat, with no light painted into them (a highlight would become a filament).
# Said in the positive: the fast image models run without a negative prompt.
# {colors} is the number of filaments. `print_hint` in kami.toml replaces it; "" turns it off.
PRINT_HINT = ("designed as a collectible vinyl toy figure for 3D printing, one solid piece "
              "standing upright on a wide flat base, chunky thick rounded shapes, every part "
              "attached to the body, {colors} flat solid colours in large clearly separated "
              "areas with crisp edges, matte uniform surfaces, flat even lighting")
PRINT_COLORS = 4                 # one AMS; `kami print --colors` has the same default
MAX_PRINT_COLORS = 16


def printable(prompt: str, hint: str, colors: int | None = None) -> str:
    """The prompt with the printing hint after it (none if `hint` is empty)."""
    hint = (hint or "").strip()
    if not hint:
        return prompt
    return f"{prompt}, {hint.replace('{colors}', str(colors or PRINT_COLORS))}"


def free_path(path: Path) -> Path:
    """`path`, or the first `stem-N.suffix` next to it that does not exist yet."""
    if not path.exists():
        return path
    n = 2
    while (candidate := path.with_name(f"{path.stem}-{n}{path.suffix}")).exists():
        n += 1
    return candidate


def views_name(front_stem: str) -> str:
    """The name of a views job started from a front picture: "owl-front" -> "owl", so its
    views are owl-left.png, not owl-front-left.png (which reads as a front-left view)."""
    name = safe_name(front_stem)
    for angle in sorted(VIEW_ANGLES, key=len, reverse=True):
        if name.endswith(f"-{angle}"):
            return name[:-len(angle) - 1] or name
    return name


def safe_name(text: str, fallback: str = "kami") -> str:
    """A file name out of a prompt or a file stem: 'A red robot!' -> 'a-red-robot'."""
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    name = "-".join(words)[:48].strip("-")
    return name or fallback
