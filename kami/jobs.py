"""Client side of kami-server's jobs (see server.py): submit, wait, download.

A job answers at once with an id; kami then polls it, so no request stays open longer
than the RunPod proxy allows (~100 s) however long the GPU works.
"""

from __future__ import annotations

import re
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
# How long a job already on the pod is waited for while the pod cannot be reached. The
# job goes on there whatever this PC's network does: a mesh that was finished and being
# saved was lost to a few seconds without DNS, when one request gave up after RETRIES.
UNREACHABLE_PATIENCE = 10 * 60

# Image formats TRELLIS.2 (PIL) reads; anything else is refused before a job is sent.
# what kami-server reads (server.open_image)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif", ".tif", ".tiff",
                  ".bmp", ".gif")
MAX_IMAGE_BYTES = 48 * 1024 * 1024


class JobError(ConfigError):
    pass


class Unreachable(JobError):
    """The pod did not answer at all (no network, no DNS, a timeout): not a refusal."""


def patiently(call, patience: float = UNREACHABLE_PATIENCE, sleep=time.sleep, waiting=None):
    """`call()`, tried again while the pod cannot be reached, for up to `patience` seconds
    in a row. `waiting(seconds)` is told how long it has been. For what happens after a
    job is on the pod (asking how it goes, fetching its files): the work is done or under
    way there, and giving up would throw it away."""
    lost = None
    while True:
        try:
            return call()
        except Unreachable as e:
            lost = time.monotonic() if lost is None else lost
            gone = time.monotonic() - lost
            if gone >= patience:
                raise Unreachable(f"{str(e).splitlines()[0]}\nNo answer for {ui.duration(gone)}. The job may "
                                  "still be on the pod: `kami status`, `kami jobs`.") from None
            if waiting:
                waiting(gone)
            sleep(POLL_EVERY)


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
    raise Unreachable(f"Could not reach the pod: {last}\n`kami status` to check it.")


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
            job = patiently(lambda: _json(*_request(f"{url}/v1/jobs/{job['id']}", api_key), "Checking the job"),
                            sleep=sleep, waiting=lambda gone: progress.phase(
                                f"{label}: cannot reach the pod, trying again ({ui.duration(gone)})"))
    finally:
        progress.stop()


def download(url: str, api_key: str, job: dict, out_dir: Path, sleep=time.sleep) -> list[Path]:
    """Save the job's files into out_dir (never overwriting: name-2.glb, name-3.glb...)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in job.get("files") or []:
        status, data = patiently(lambda: _request(f"{url}/v1/jobs/{job['id']}/files/{name}", api_key,
                                                  timeout=5 * REQUEST_TIMEOUT), sleep=sleep)
        if status >= 400:
            _json(status, data, f"Downloading {name}")
        path = free_path(out_dir / name)
        path.write_bytes(data)
        saved.append(path)
    return saved


# What is added to a prompt so the picture is of something a printer can make, and
# `kami print` can colour. Each preset is a kind of print:
# (label, what it is for, the text, how its front picture is shown or None for the usual).
# Said in the positive: the fast image models run without a negative prompt, and naming
# what is not wanted ("no base", "no joints") tends to draw it.
# All of them end the same way: flat matte paint, evenly lit, because a highlight painted
# into the texture would become a filament. The colours are the subject's own and are not
# counted here: `kami print` reduces them to the filaments afterwards. The full Z-Image
# takes every word at its value (84 pictures, kami-out/prompt-tests/sheets, 2026-11-20):
# "N flat solid colours in large clearly separated areas" made toys out of building
# blocks, "in 3 colours" a gecko half yellow and half blue, "matte uniform surfaces" a
# figure of unpainted clay, "painted in flat matte colours" a rainbow. A custom
# print_hint may still say {colors}: it becomes the number of filaments.
_COLOURS = "its natural colours in flat matte paint, flat even lighting"
PRINT_PRESETS = {
    # The style of every preset: a figurine that simplifies its subject without ceasing to
    # be it. "collectible vinyl toy figure ... chunky thick rounded shapes" came first, and
    # on a model that obeys it was drawn over the subject (a snake as a sausage). On
    # FLUX.2, 18 fronts in three styles kept every animal itself; this one, between the
    # toy and the realistic, read best for printing (kami-out/prompt-tests/sheets/grid-run8).
    "figure": ("Figure", "A stylised figure standing on its own wide base",
               "stylised figurine with simplified natural proportions, smooth sculpted forms, one "
               "solid piece standing upright on a wide flat base, " + _COLOURS, None),
    # For models that get flexible joints cut into them afterwards (jointwright): the body
    # must come whole and smooth, with everything that will bend lying on the plate, and
    # the parts apart from each other so there is room to cut between them.
    # What ran on a views pod on 2026-11-20, and what it taught:
    # 1. "one-piece print-in-place toy, the whole creature lying flat on its belly ...
    #    head, body, legs and tail all stretched out ... even thickness with a flat
    #    underside": a good gecko (whole, smooth, on its belly, limbs apart, coherent
    #    views), but a caterpillar in glossy separate segments, and a snake as a flat
    #    cut-out with arms. It listed parts ("legs") and said "flat" of the body.
    # 2. "a single smooth mass of matte modelling clay, a plump fully three-dimensional
    #    creature ... as tall as it is wide ... anything that sticks out ...": far worse.
    #    "cute snake" came out a shapeless lump in four arbitrary colours. Eighty words
    #    about shape against two about the subject: the model drew the hint.
    # Meanwhile `figure` drew that same snake well: short, a style to hang on to, and
    # nothing about anatomy. That is the rule this wording still follows; the exact words
    # are row T of the FLUX.2 run, the only ones here tried as written.
    # Then 50 front pictures, the same subjects (snake, caterpillar, gecko) and seeds
    # through ten wordings (kami-out/prompt-tests/sheets/grid-run1.png, grid-run2.png):
    # - "N flat solid colours in large clearly separated areas" makes the model invent
    #   them on a lying body (a red head, a blue segment); "in N colours" keeps the
    #   subject's own (a yellow gecko with a red tail), and none at all gives one colour;
    # - "head resting on the ground" works every time, and "body in a straight line" for
    #   the gecko and the caterpillar; a snake coils whatever is said;
    # - a caterpillar comes in segments, with feet and antennae, whatever is said;
    # - none is a true front view; "seen from directly above" gave the gecko a perfect
    #   plan view, and the caterpillar ignored it.
    "flexi": ("Flexi (joints added later)",
              "A creature lying on its belly, whole, with nothing under it: joints are cut afterwards",
              "stylised animal figurine with simplified natural anatomy, smooth sculpted forms, "
              "one solid piece lying on its belly directly on the ground, body in a straight "
              "line, head resting on the ground, " + _COLOURS, None),
    "lowpoly": ("Low poly", "A faceted sculpture: flat triangles, one colour each",
                "designed as a low-poly sculpture for 3D printing, one solid piece standing on a "
                "wide flat base, large flat triangular facets with sharp edges, thick sturdy "
                "shapes, every part attached to the body, each facet a single colour, " + _COLOURS,
                None),
    "bust": ("Bust", "Head and shoulders, cut flat across the chest",
             "designed as a sculpted bust for 3D printing, head and shoulders only, cut straight "
             "across the chest into a wide flat base, one solid piece, thick sturdy shapes, "
             "simplified bold features, " + _COLOURS, None),
    "relief": ("Medallion", "The subject in raised relief on a thick flat-backed disc",
               "designed as a thick round medallion for 3D printing, the subject in bold raised "
               "relief on a flat-backed disc, simple chunky shapes with deep clear steps between "
               "levels, one colour per level, " + _COLOURS, None),
    "natural": ("Natural", "The object as it is, only without light painted into it",
                "the whole object in one solid piece, complete and true to life, matte finish, "
                "flat even lighting", None),
}
# What each preset keeps out, for models that read a negative prompt (Z-Image, with
# guidance above 0; Z-Image-Turbo ignores it). Here, and only here, the unwanted is named.
_UNWANTED = ("glossy, shiny, reflections, specular highlights, harsh shadows, gradient shading, "
             "text, watermark, extra objects, cropped")
PRINT_NEGATIVES = {
    "figure": _UNWANTED,
    "flexi": _UNWANTED + ", base, pedestal, stand, platform, joints, hinges, seams, separate "
                         "segments, standing, raised head, coiled",
    "lowpoly": _UNWANTED + ", smooth curved surfaces",
    "bust": _UNWANTED + ", full body, arms, legs",
    "relief": _UNWANTED + ", free-standing figure",
    "natural": "glossy highlights, harsh shadows, text, watermark, extra objects, cropped",
}
PRINT_PRESET = "figure"          # the one used when nothing says otherwise
PRINT_HINT = PRINT_PRESETS[PRINT_PRESET][2]
PRINT_COLORS = 4                 # one AMS; `kami print --colors` has the same default
MAX_PRINT_COLORS = 16

# Words that take something away. The image models here run without a negative prompt, so
# "a caterpillar without antennae" is, to them, a caterpillar and antennae.
NEGATION = re.compile(r"\b(no|not|without|never|none|sin|ningun[ao]?)\b", re.IGNORECASE)


def negations(prompt: str) -> list[str]:
    """The words of a prompt that ask for something to be left out, as written."""
    return list(dict.fromkeys(m.group(0) for m in NEGATION.finditer(prompt or "")))


NEGATION_ADVICE = ("a prompt draws what it names, also after \"no\" or \"without\": say what you want "
                   "in its place (\"a smooth round head\"), or name it in the negative prompt (--negative)")


def print_hint(text: str | None = None, preset: str | None = None) -> str:
    """The hint in force: `text` when given (kami.toml's print_hint, "" for none), else
    the preset's, else the usual one."""
    if text is not None:
        return text
    return PRINT_PRESETS[preset or PRINT_PRESET][2]


def preset_negative(preset: str | None) -> str | None:
    """What a preset keeps out of the picture (a negative prompt), or None."""
    return PRINT_NEGATIVES.get(preset)


def preset_front(preset: str | None) -> str | None:
    """How a preset wants its front picture shown, or None for the pod's usual words."""
    return PRINT_PRESETS[preset][3] if preset in PRINT_PRESETS else None


def printable(prompt: str, hint: str, colors: int | None = None) -> str:
    """The prompt with the printing hint after it (none if `hint` is empty)."""
    hint = (hint or "").strip()
    if not hint:
        return prompt
    return f"{prompt}, {hint.replace('{colors}', str(colors or PRINT_COLORS))}"


# Asking Gemini (gemini.google.com, by hand, on one's own account) for the pictures: what
# `kami ui` copies to paste there. Its fronts kept each subject itself where the pod's
# models did not (2026-11-20); asked in the same conversation, its sides stay the same
# object. Ground level and sharp focus, since a mesh model reads a blurred tail as missing.
GEMINI_FRONT = ("Everything in sharp focus from end to end, no depth-of-field blur. "
                "Square image, 1:1.")
GEMINI_SAME = ("Same object, same colours, same size in frame, same white background and "
               "lighting. Camera at ground level, at the height of the object. Everything in "
               "sharp focus from end to end, no depth-of-field blur. Square image, 1:1.")
GEMINI_VIEWS = {
    "front": "Show the exact same figurine from the front again, facing the camera.",
    "left": "Show the exact same figurine from its left side, in profile.",
    "back": "Show the exact same figurine from behind.",
    "right": "Show the exact same figurine from its right side, in profile.",
    "front-left": "Show the exact same figurine three-quarters from the front, between its front and its left side.",
    "front-right": "Show the exact same figurine three-quarters from the front, between its front and its right side.",
    "back-left": "Show the exact same figurine three-quarters from behind, between its back and its left side.",
    "back-right": "Show the exact same figurine three-quarters from behind, between its back and its right side.",
}


def gemini_view(view: str) -> str:
    """What to ask Gemini, in the conversation that made the front, for one more side."""
    return f"{GEMINI_VIEWS[view]} {GEMINI_SAME}"


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
