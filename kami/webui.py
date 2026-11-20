"""`kami ui`: a local web page to go from a concept to a mesh to a print.

The page has four stages, Concept (text -> pictures of an object's front), Views (a
picture of the front -> its sides), Mesh (views -> GLB) and Print (GLB -> 3MF, on this
PC), and opens on the first one the active pod serves. A small HTTP server on 127.0.0.1 serves the page, the files of the output folder
(the library, grouped by object), and a JSON API the page uses: the pod's state, the jobs
sent from the page, and the actions (send jobs, prepare a print, delete the pod). kami relays jobs to the pod itself, so the pod's API key
never reaches the browser. It never brings a pod up: that stays a deliberate `kami up`.

Only this PC can reach it (127.0.0.1, and the Host header is checked against DNS
rebinding). Requests that act (POST) also need the page's own token and a same-origin
Origin header, so another web page cannot send jobs or delete the pod.

The 3D viewer is Google's <model-viewer>, loaded from a CDN: without internet the images
still show, the meshes do not.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import html
import itertools
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from . import jobs
from .config import ConfigError

MESH_SUFFIXES = (".glb",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PRINT_SUFFIXES = (".3mf",)
RESULT_SUFFIXES = (*MESH_SUFFIXES, *IMAGE_SUFFIXES, *PRINT_SUFFIXES)
TYPES = {".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg",
         ".jpeg": "image/jpeg", ".webp": "image/webp", ".3mf": "model/3mf"}
MODEL_VIEWER = "https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"

MAX_BODY = 200 * 1024 * 1024     # bytes of a POST (several images, base64)
MAX_IMAGES = 20                  # images in one mesh request
MAX_COUNT = 8                    # images of one prompt
MAX_ACTIVE = 16                  # jobs in flight from the page at once
KEPT_JOBS = 40                   # finished jobs listed on the page
MAX_THUMBS = 200                 # mesh thumbnails kept in memory
STATUS_TTL = 10                  # seconds the pod's state is cached (provider API calls)
JOBS_TTL = 2                     # seconds the pod's job listing is cached
PROGRESS_FIELDS = ("stage", "stage_label", "step", "steps", "progress")

# what the page may set per kind, and the type each value must have
MESH_PARAMS = {"seed": int, "texture_size": int, "decimation_target": int, "remesh": bool,
               "multiview_mode": str, "texture": bool}
MULTIVIEW_MODES = ("stochastic", "multidiffusion")
MAX_VIEWS = 8                    # views of one object in one job (the pod may take fewer)
IMAGE_PARAMS = {"seed": int, "steps": int, "guidance": float, "width": int, "height": int,
                "negative": str}
VIEWS_PARAMS = {"seed": int, "steps": int, "guidance": float, "lora_scale": float,
                "elevation": str, "distance": str, "front_hint": str}
MAX_HINT = 2000                  # characters of a hint written on the page


# --- results (the library) ---

def results(folder: Path) -> list[Path]:
    """The meshes, images and 3MFs in `folder` and in the folders directly inside it
    (where one tidies them by hand), newest first."""
    if not folder.is_dir():
        return []
    files = []
    for entry in folder.iterdir():
        inside = [entry] if entry.is_file() else (
            list(entry.iterdir()) if entry.is_dir() and not entry.name.startswith(".") else [])
        files += [p for p in inside if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def signature(folder: Path) -> str:
    """Changes when a result is added, replaced or removed (the library refreshes then).
    A hash, so no file name ends up in the page's script."""
    listing = ";".join(f"{p.relative_to(folder).as_posix()}:{p.stat().st_mtime_ns}:{p.stat().st_size}"
                       for p in results(folder))
    return hashlib.sha256(listing.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def resolve(folder: Path, name) -> Path | None:
    """The result `name` ("owl.glb", "pixal3d/owl.glb") as a file, or None: a plain name
    or one folder deep, of a kind the page shows, and nothing else on the disk."""
    if not isinstance(name, str) or "\\" in name:
        return None
    parts = name.split("/")
    if not 1 <= len(parts) <= 2 or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    target = folder.joinpath(*parts)
    return target if target.suffix.lower() in TYPES and target.is_file() else None


# "owl-left.png", "owl-front-2.png": a view of "owl" (the 2 is a later take of it)
VIEW_OF = re.compile(r"^(.+?)-(%s)(?:-\d+)?$" % "|".join(sorted(jobs.VIEW_ANGLES, key=len, reverse=True)))


def objects(folder: Path) -> list[dict]:
    """The library as the things being made, newest first: an object is the pictures of
    its sides, its mesh and its 3MF, found by name (owl-front.png, owl-left.png,
    owl-front.glb and owl.3mf are all "owl"). The newest take of each view counts."""
    found: dict[str, dict] = {}
    for p in results(folder):
        rel = p.relative_to(folder).as_posix()
        sub = rel.rpartition("/")[0]
        suffix = p.suffix.lower()
        match = VIEW_OF.match(p.stem)
        base = match.group(1) if match else p.stem
        key = f"{sub}/{base}" if sub else base
        stat = p.stat()
        obj = found.setdefault(key, {"key": key, "name": base, "folder": sub, "views": {},
                                     "images": [], "meshes": [], "prints": [], "mtime": 0})
        obj["mtime"] = max(obj["mtime"], stat.st_mtime)
        if suffix in IMAGE_SUFFIXES:
            if match:
                obj["views"].setdefault(match.group(2), rel)      # newest first: the first stays
            else:
                obj["images"].append(rel)
        else:
            obj["meshes" if suffix in MESH_SUFFIXES else "prints"].append(
                {"file": rel, "size": stat.st_size})
    return sorted(found.values(), key=lambda o: (o["mtime"], o["key"]), reverse=True)


# --- printing (on this PC) ---

class Printer:
    """`kami print` for the page: the mesh of an object made printable, shown painted
    before the 3MF is saved. The slow half (the solid on the plate) is kept between
    requests, so changing colours answers at once."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.key = None
        self.prepared = None
        self.preview = (None, b"")               # (id, the painted GLB)
        self.thumbs: dict[tuple, bytes | None] = {}

    def thumbnail(self, name: str) -> bytes | None:
        """A small PNG of the mesh `name`, made once per file; None when it cannot be
        (not a mesh of the library, a GLB that does not load, no numpy on this PC)."""
        source = resolve(self.folder, name)
        if source is None or source.suffix.lower() not in MESH_SUFFIXES:
            return None
        key = (name, source.stat().st_mtime_ns)
        if key not in self.thumbs:
            try:
                from . import printing
                picture = printing.thumbnail(source)
            except Exception:                     # a picture is a nicety: never an error page
                picture = None
            with self.lock:
                if len(self.thumbs) >= MAX_THUMBS:
                    self.thumbs.clear()
                self.thumbs[key] = picture
        return self.thumbs[key]

    def run(self, request: dict) -> dict:
        """Raises ValueError (a bad request) or ConfigError (what `kami print` would say)."""
        from . import printing                    # numpy and friends: only when printing
        if not isinstance(request, dict):
            raise ValueError("The request must be a JSON object")
        source = resolve(self.folder, request.get("file"))
        if source is None or source.suffix.lower() not in MESH_SUFFIXES:
            raise ValueError("Pick a mesh (.glb) from the library")
        geometry = _params({k: request.get(k) for k in ("height", "base", "detail", "rebuild", "keep_cavities")},
                           {"height": float, "base": float, "detail": float, "rebuild": str,
                            "keep_cavities": bool})
        finish = _params({k: request.get(k) for k in ("smoothing", "min_patch")},
                         {"smoothing": int, "min_patch": float})
        colors = request.get("colors")
        if colors is not None and (isinstance(colors, bool) or not isinstance(colors, int)):
            raise ValueError("colors must be a whole number")
        palette = request.get("palette")
        if palette is not None:
            if not isinstance(palette, list) or not all(isinstance(c, str) for c in palette):
                raise ValueError("palette must be a list of colours like #1a4fa0")
            palette = printing.parse_palette(",".join(palette))
        key = (str(source), source.stat().st_mtime_ns, tuple(sorted(geometry.items())))
        with self.lock:
            if key != self.key:
                self.prepared = printing.prepare(
                    source, height=geometry.get("height", 80.0), base=geometry.get("base", 2.0),
                    detail=geometry.get("detail", printing.DETAIL),
                    rebuilding=geometry.get("rebuild", "auto"),
                    keep_cavities=geometry.get("keep_cavities", False))
                self.key = key
            mesh, report = self.prepared
            labels, swatches, share = printing.paint(mesh, colors=colors or jobs.PRINT_COLORS,
                                                     palette=palette, **finish)
            saved = None
            if request.get("save") is True:
                target = jobs.free_path(source.with_suffix(".3mf"))
                printing.write_3mf(target, mesh, labels, swatches, source.stem)
                saved = target.relative_to(self.folder).as_posix()
            done = printing.describe(None, mesh, report, swatches, share)
            self.preview = (secrets.token_urlsafe(8), printing.preview_glb(mesh, labels, swatches))
        return {"preview": f"preview/{self.preview[0]}.glb", "saved": saved,
                "size": [round(x, 1) for x in done.size], "triangles": done.triangles,
                "base_area": round(done.base_area), "closed": report.closed,
                "rebuilt": report.rebuilt, "detail": round(report.detail, 2),
                "cavities": report.cavities, "fragments": report.fragments,
                "colors": [{"hex": c, "share": round(s, 4)} for c, s in done.colors],
                "warnings": printing.warnings(done)}


# --- jobs sent from the page ---

class Work:
    """The jobs the page sent: each runs in its own thread, which submits it to the pod,
    follows it there and downloads its files into the folder. The pod runs them one at a
    time, in order; this only relays them."""

    def __init__(self, backend, folder: Path, sleep=time.sleep):
        self.backend = backend
        self.folder = folder
        self.sleep = sleep
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.ids = itertools.count(1)

    def listing(self, pod: dict | None = None) -> list[dict]:
        """The page's jobs, newest first; with the pod's listing (GET /v1/jobs), every job
        on the pod instead, in its order, with stage and progress: the page's own ones
        keep their label and downloaded files, the rest (sent from a terminal or another
        kami ui) come marked `external`."""
        with self.lock:
            mine = [dict(j) for j in reversed(self.jobs.values())]
        if not isinstance(pod, dict) or not isinstance(pod.get("jobs"), list):
            return mine
        by_remote = {j["remote"]: j for j in mine if j.get("remote")}
        merged, seen = [], set()
        for r in pod.get("jobs") or []:
            progress = {k: r.get(k) for k in PROGRESS_FIELDS}
            own = by_remote.get(r.get("id"))
            if own:
                seen.add(own["id"])
                # the pod knows better while it works; past that this side downloads
                if own["status"] in ("queued", "running") and r.get("status") in ("queued", "running"):
                    own.update(status=r.get("status") or own["status"], position=r.get("position"))
                merged.append({**own, **progress})
            else:
                merged.append({"id": "pod-" + str(r.get("id")), "label": r.get("name") or r.get("id"),
                               "kind": r.get("kind"), "status": r.get("status"),
                               "position": r.get("position"), "error": r.get("error"),
                               "files": r.get("files") or [], "created": r.get("created"),
                               "finished": r.get("finished"), "external": True, **progress})
        rest = [j for j in mine if j["id"] not in seen]
        # still being sent on top; finished ones the pod no longer keeps at the end
        return ([j for j in rest if j["status"] not in ("done", "error")] + merged +
                [j for j in rest if j["status"] in ("done", "error")])

    def active(self) -> int:
        with self.lock:
            return sum(j["status"] not in ("done", "error") for j in self.jobs.values())

    def create(self, request: dict) -> list[dict]:
        """Check a request from the page and start its jobs. Raises ValueError."""
        bodies = build_bodies(request, getattr(self.backend, "print_hint", ""), self.folder)
        if self.active() + len(bodies) > MAX_ACTIVE:
            raise ValueError(f"Too many jobs in flight (at most {MAX_ACTIVE}): wait for some to finish")
        created = []
        for label, body in bodies:
            job = {"id": f"w{next(self.ids)}", "label": label, "kind": body["kind"],
                   "status": "sending", "position": None, "error": None, "files": [], "remote": None,
                   "created": time.time(), "finished": None}
            with self.lock:
                self.jobs[job["id"]] = job
                self._prune()
            threading.Thread(target=self._run, args=(job["id"], body), daemon=True).start()
            created.append(dict(job))
        return created

    def _prune(self):
        done = [k for k, j in self.jobs.items() if j["status"] in ("done", "error")]
        for key in done[:max(0, len(done) - KEPT_JOBS)]:
            del self.jobs[key]

    def _set(self, job_id, **fields):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(fields)

    def _run(self, job_id: str, body: dict):
        try:
            url, key = self.backend.target()
            remote = jobs.submit(url, key, body)
            self._set(job_id, remote=remote.get("id"))
            started = time.monotonic()
            while remote.get("status") not in ("done", "error"):
                self._set(job_id, status=remote.get("status") or "queued",
                          position=remote.get("position"))
                if time.monotonic() - started > jobs.JOB_TIMEOUT:
                    raise jobs.JobError("Taking too long; it keeps running on the pod (`kami logs`)")
                self.sleep(jobs.POLL_EVERY)
                remote = jobs.check(url, key, remote["id"])
            if remote["status"] == "error":
                raise jobs.JobError(remote.get("error") or "failed on the pod (`kami logs`)")
            self._set(job_id, status="downloading", position=None)
            saved = jobs.download(url, key, remote, self.folder)
            self._set(job_id, status="done", files=[p.name for p in saved], finished=time.time())
        except (ConfigError, OSError, KeyError, ValueError) as e:
            self._set(job_id, status="error", error=str(e).splitlines()[0] if str(e) else type(e).__name__,
                      position=None, finished=time.time())


def _params(raw, allowed: dict) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    out = {}
    for key, value in raw.items():
        if key not in allowed:
            raise ValueError(f"unknown parameter: {key}")
        if value is None or value == "":
            continue
        kind = allowed[key]
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
                raise ValueError(f"{key} must be a whole number")
            value = int(value)
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        out[key] = value
    return out


def _prompt(request: dict) -> str:
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Write a prompt")
    return prompt.strip()


def _hint(request: dict, key: str) -> str | None:
    """A hint the page wrote itself, or None when it sent none."""
    text = request.get(key)
    if text is None:
        return None
    if not isinstance(text, str) or len(text) > MAX_HINT:
        raise ValueError(f"{key} must be text of at most {MAX_HINT} characters")
    return text.strip()


def _sent(request: dict, prompt: str, hint: str) -> str:
    """The prompt as the pod gets it: with the printing hint, unless the request says
    `"print_hint": false`. `"hint"` is the page's own wording of it (else kami.toml's),
    `"colors"` how many filaments it asks for, and `"front_hint"` (image jobs: a front to
    choose) what the pod itself adds to the front of a views job."""
    if request.get("print_hint") is not False:
        colors = request.get("colors")
        if colors is not None and (isinstance(colors, bool) or not isinstance(colors, int)
                                   or not 1 <= colors <= jobs.MAX_PRINT_COLORS):
            raise ValueError(f"colors must be 1 to {jobs.MAX_PRINT_COLORS}")
        own = _hint(request, "hint")
        prompt = jobs.printable(prompt, hint if own is None else own, colors)
    front = _hint(request, "front_hint") if request.get("kind") == "image" else None
    return f"{prompt}, {front}" if front else prompt


def _picture(item, folder: Path | None) -> tuple[str, bytes]:
    """(name, bytes) of a picture the page sends: {name, data} uploaded as base64, or
    {file} naming one already in the library. Raises ValueError."""
    if not isinstance(item, dict):
        raise ValueError("Each image must be {name, data} or {file}")
    if item.get("file") is not None:
        source = resolve(folder, item["file"]) if folder else None
        if source is None or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{item['file']}: not a picture in the library")
        name, data = source.name, source.read_bytes()
    else:
        name = str(item.get("name") or "image")
        if Path(name).suffix.lower() not in jobs.IMAGE_SUFFIXES:
            raise ValueError(f"{name}: not an image kami can send ({', '.join(jobs.IMAGE_SUFFIXES)})")
        try:
            data = base64.b64decode(item.get("data") or "", validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise ValueError(f"{name}: the image data is not valid base64") from None
    if not data:
        raise ValueError(f"{name}: the image is empty")
    if len(data) > jobs.MAX_IMAGE_BYTES:
        raise ValueError(f"{name}: over {jobs.MAX_IMAGE_BYTES // 2**20} MiB, resize it first")
    return name, data


def build_bodies(request: dict, hint: str = "", folder: Path | None = None) -> list[tuple[str, dict]]:
    """(label, kami-server job body) for each job a page request asks for. `hint` is the
    printing hint its prompts get (labels and file names keep the prompt as written);
    `folder` is the library its pictures may come from."""
    if not isinstance(request, dict):
        raise ValueError("The request must be a JSON object")
    kind = request.get("kind")
    if kind == "mesh":
        images = request.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("Add at least one image")
        if len(images) > MAX_IMAGES:
            raise ValueError(f"At most {MAX_IMAGES} images at once")
        params = _params(request.get("params"), MESH_PARAMS)
        if params.get("multiview_mode", "stochastic") not in MULTIVIEW_MODES:
            raise ValueError(f"multiview_mode must be one of {', '.join(MULTIVIEW_MODES)}")
        views = request.get("views") is True
        if views and len(images) > MAX_VIEWS:
            raise ValueError(f"At most {MAX_VIEWS} views of one object")
        bodies, parts = [], []
        for image in images:
            name, data = _picture(image, folder)
            view = image.get("view") or None
            if view is not None and view not in jobs.VIEW_ANGLES:
                raise ValueError(f"{name}: the view must be one of {', '.join(jobs.VIEW_ANGLES)}")
            encoded = base64.b64encode(data).decode()
            parts.append({"image": encoded, "view": view, "name": name})
            bodies.append((f"Mesh of {name}",
                           {"kind": "mesh", "image": encoded,
                            "name": jobs.safe_name(Path(name).stem), "params": params}))
        if views:
            # one object: one job with every image as a view of it
            names = [p["name"] for p in parts]
            return [(f"Mesh of {len(parts)} views ({', '.join(names)})",
                     {"kind": "mesh", "name": jobs.safe_name(Path(names[0]).stem), "params": params,
                      "images": [{"image": p["image"], "view": p["view"]} for p in parts]})]
        return bodies
    if kind == "image":
        prompt = _prompt(request)
        sent = _sent(request, prompt, hint)
        count = request.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
            raise ValueError(f"count must be 1 to {MAX_COUNT}")
        params = _params(request.get("params"), IMAGE_PARAMS)
        bodies = []
        for i in range(count):
            p = dict(params)
            if "seed" in p:
                p["seed"] += i
            label = f"Image {i + 1}/{count}: {prompt}" if count > 1 else f"Image: {prompt}"
            bodies.append((label, {"kind": "image", "prompt": sent,
                                   "name": jobs.safe_name(prompt), "params": p}))
        return bodies
    if kind == "text-to-mesh":
        prompt = _prompt(request)
        raw = request.get("params") or {}
        if not isinstance(raw, dict):
            raise ValueError("params must be an object")
        params = {"image": _params(raw.get("image"), IMAGE_PARAMS),
                  "mesh": _params(raw.get("mesh"), MESH_PARAMS)}
        return [(f"Mesh from text: {prompt}",
                 {"kind": "text-to-mesh", "prompt": _sent(request, prompt, hint),
                  "name": jobs.safe_name(prompt), "params": params})]
    if kind == "views":
        raw = request.get("params")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("params must be an object")
        raw = dict(raw or {})
        front_params = _params(raw.pop("image", None), IMAGE_PARAMS)     # of the front made from text
        params = _params(raw, VIEWS_PARAMS)
        if front_params:
            params["image"] = front_params
        angles = request.get("angles")
        if angles is not None:
            if (not isinstance(angles, list) or not angles
                    or not all(a in jobs.VIEW_ANGLES for a in angles)):
                raise ValueError(f"angles must be some of {', '.join(jobs.VIEW_ANGLES)}")
            params["views"] = list(dict.fromkeys(angles))
        front = request.get("front")
        if front is not None:
            name, data = _picture(front, folder)
            return [(f"Views of {name}",
                     {"kind": "views", "image": base64.b64encode(data).decode(),
                      "name": jobs.views_name(Path(name).stem), "params": params})]
        prompt = _prompt(request)
        return [(f"Views: {prompt}", {"kind": "views", "prompt": _sent(request, prompt, hint),
                                       "name": jobs.safe_name(prompt), "params": params})]
    raise ValueError("kind must be mesh, image, text-to-mesh or views")


# --- the pod's state ---

class PodStatus:
    """The backend's pod status, cached for STATUS_TTL seconds: it calls the provider's
    API, and the page asks every few seconds."""

    def __init__(self, backend, ttl: float = STATUS_TTL):
        self.backend = backend
        self.ttl = ttl
        self.lock = threading.Lock()
        self.value = None
        self.at = 0.0

    def get(self, fresh: bool = False) -> dict:
        with self.lock:
            if fresh or self.value is None or time.monotonic() - self.at > self.ttl:
                try:
                    self.value = self.backend.status()
                except Exception as e:          # never let the page break on it
                    self.value = {"pod": None, "error": str(e).splitlines()[0] if str(e) else type(e).__name__}
                self.at = time.monotonic()
            return self.value


class PodJobs:
    """The pod's job listing (GET /v1/jobs), cached for JOBS_TTL seconds however many
    tabs ask. None when there is no ready pod, its server predates the listing, or it
    cannot be reached (the page then lists only its own jobs)."""

    def __init__(self, backend, ttl: float = JOBS_TTL):
        self.backend = backend
        self.ttl = ttl
        self.lock = threading.Lock()
        self.value = None
        self.at = 0.0

    def get(self, ready: bool) -> dict | None:
        if not ready:
            return None
        with self.lock:
            if time.monotonic() - self.at > self.ttl:
                try:
                    self.value = self.backend.jobs()
                except Exception:
                    self.value = None
                self.at = time.monotonic()
            return self.value


# --- the page ---

PAGE_FILE = Path(__file__).with_name("webui.html")


def page(folder: Path, token: str) -> str:
    values = {"MODEL_VIEWER": MODEL_VIEWER, "TOKEN": token,
              "FOLDER": html.escape(str(folder.resolve()))}
    # one pass: a file named "{{TOKEN}}.png" stays a name, it is not filled in again
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)),
                  PAGE_FILE.read_text(encoding="utf-8"))


# --- the server ---

def make_handler(folder: Path, port_ref: list, token: str, work: Work, status: PodStatus,
                 backend):
    pod_jobs = PodJobs(backend)
    printer = Printer(folder)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # quiet: the terminal shows kami's own lines
            pass

        def _send(self, code, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, data):
            self._send(code, json.dumps(data).encode(), "application/json")

        def _origins(self):
            return (f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}")

        def _host_ok(self) -> bool:
            # only this machine's own addresses: a web page elsewhere cannot point a
            # hostname at 127.0.0.1 (DNS rebinding) and read the results
            return (self.headers.get("Host") or "").lower() in self._origins()

        def _may_act(self) -> bool:
            """A POST from this page only: its token, JSON, and our own Origin. A form or
            script on another site cannot set the token header, nor fake the Origin."""
            origin = (self.headers.get("Origin") or "").lower()
            if origin and origin not in tuple(f"http://{o}" for o in self._origins()):
                return False
            sent = self.headers.get("X-Kami-Token") or ""
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            return secrets.compare_digest(sent, token) and kind == "application/json"

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            path = unquote(urlsplit(self.path).path)
            if path in ("/", "/index.html"):
                return self._send(200, page(folder, token).encode(), "text/html; charset=utf-8")
            if path == "/api/state":
                pod = status.get()
                listing = pod_jobs.get(bool(pod.get("pod") and pod.get("ready")))
                preset = getattr(backend, "print_preset", None)
                return self._json(200, {"signature": signature(folder), "pod": pod,
                                        "jobs": work.listing(listing),
                                        "print_hint": bool(getattr(backend, "print_hint", "")),
                                        "print_hint_text": str(getattr(backend, "print_hint", "") or ""),
                                        "print_preset": preset if isinstance(preset, str) else None,
                                        "print_presets": [{"name": name, "label": label, "about": about, "text": text, "front": front,
                                                           "negative": jobs.preset_negative(name)}
                                                          for name, (label, about, text, front) in jobs.PRINT_PRESETS.items()],
                                        "print_colors": jobs.PRINT_COLORS})
            if path == "/api/objects":
                return self._json(200, {"signature": signature(folder), "objects": objects(folder)})
            if path.startswith("/thumbs/"):
                picture = printer.thumbnail(path[len("/thumbs/"):])
                if picture:
                    return self._send(200, picture, "image/png")
            if path.startswith("/preview/"):
                name, body = printer.preview
                if name and path == f"/preview/{name}.glb":
                    return self._send(200, body, TYPES[".glb"])
            if path.startswith("/files/"):
                # a result in the folder, or one folder deep: no other file of the disk
                target = resolve(folder, path[len("/files/"):])
                if target is not None:
                    return self._send(200, target.read_bytes(), TYPES[target.suffix.lower()])
            self._send(404, b"not found", "text/plain")

        do_HEAD = do_GET

        def do_POST(self):
            if not self._host_ok() or not self._may_act():
                return self._json(403, {"error": "forbidden"})
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self._json(413, {"error": f"request over {MAX_BODY // 2**20} MiB"})
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": "the request is not valid JSON"})
            if path == "/api/jobs":
                pod = status.get(fresh=True)
                if not (pod.get("pod") and pod.get("ready")):
                    return self._json(409, {"error": "No ready pod: run `kami up` in a terminal"})
                try:
                    created = work.create(request)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                return self._json(202, {"jobs": created})
            if path == "/api/print":
                try:
                    return self._json(200, printer.run(request))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                except (ConfigError, OSError) as e:
                    return self._json(409, {"error": str(e).splitlines()[0] if str(e) else type(e).__name__})
            if path == "/api/down":
                try:
                    result = backend.down()
                except (ConfigError, OSError) as e:
                    return self._json(409, {"error": str(e).splitlines()[0]})
                status.get(fresh=True)
                return self._json(200, result)
            self._json(404, {"error": "not found"})

    return Handler


def serve(folder: Path, backend, port: int = 0, open_browser: bool = True, on_ready=None):
    """Serve the page for `folder` on 127.0.0.1 until Ctrl+C. `port` 0 picks a free one.
    `backend` has status() -> dict, target() -> (url, api key), down() -> dict and
    jobs() -> the pod's job listing or None.
    `on_ready(url)` runs once it listens."""
    port_ref = [port]
    token = secrets.token_urlsafe(24)
    work = Work(backend, folder)
    status = PodStatus(backend)
    server = ThreadingHTTPServer(("127.0.0.1", port),
                                 make_handler(folder, port_ref, token, work, status, backend))
    server.daemon_threads = True
    port_ref[0] = server.server_address[1]
    url = f"http://127.0.0.1:{port_ref[0]}/"
    if on_ready:
        on_ready(url)
    if open_browser and not os.environ.get("KAMI_NO_BROWSER"):
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return work
