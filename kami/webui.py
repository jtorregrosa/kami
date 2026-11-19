"""`kami ui`: a local web page to make meshes and images and browse the results.

A small HTTP server on 127.0.0.1 serves the page, the files of the output folder, and a
JSON API the page uses: the pod's state, the jobs sent from the page, and the actions
(send jobs, delete the pod). kami relays jobs to the pod itself, so the pod's API key
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
TYPES = {".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg",
         ".jpeg": "image/jpeg", ".webp": "image/webp"}
MODEL_VIEWER = "https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"

MAX_BODY = 200 * 1024 * 1024     # bytes of a POST (several images, base64)
MAX_IMAGES = 20                  # images in one mesh request
MAX_COUNT = 8                    # images of one prompt
MAX_ACTIVE = 16                  # jobs in flight from the page at once
KEPT_JOBS = 40                   # finished jobs listed on the page
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


# --- results (the gallery) ---

def results(folder: Path) -> list[Path]:
    """The meshes and images directly in `folder`, newest first."""
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in (*MESH_SUFFIXES, *IMAGE_SUFFIXES)]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def signature(folder: Path) -> str:
    """Changes when a result is added, replaced or removed (the gallery refreshes then).
    A hash, so no file name ends up in the page's script."""
    listing = ";".join(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in results(folder))
    return hashlib.sha256(listing.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def cards(folder: Path) -> list[dict]:
    """One card per result. A mesh and the image of the same name (text-to-mesh makes
    both) share one card: the image is shown as the mesh's source."""
    files = results(folder)
    meshes = {p.stem: p for p in files if p.suffix.lower() in MESH_SUFFIXES}
    out = []
    for p in files:
        is_mesh = p.suffix.lower() in MESH_SUFFIXES
        if not is_mesh and p.stem in meshes:
            continue                                  # shown with its mesh
        source = None
        if is_mesh:
            source = next((q for q in files if q.stem == p.stem and q is not p
                           and q.suffix.lower() in IMAGE_SUFFIXES), None)
        stat = p.stat()
        out.append({"name": p.name, "kind": "mesh" if is_mesh else "image",
                    "size": stat.st_size, "mtime": stat.st_mtime,
                    "source": source.name if source else None})
    return out


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def _card(c: dict) -> str:
    url = "files/" + quote(c["name"])
    name = html.escape(c["name"])
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["mtime"]))
    if c["kind"] == "mesh":
        poster = f' poster="files/{quote(c["source"])}"' if c["source"] else ""
        media = (f'<model-viewer src="{url}"{poster} camera-controls auto-rotate '
                 f'shadow-intensity="1" exposure="1" loading="lazy" '
                 f'alt="{name}"></model-viewer>')
        badge = "GLB"
    else:
        media = f'<img src="{url}" alt="{name}" loading="lazy">'
        badge = "PNG" if c["name"].lower().endswith(".png") else "IMG"
    source = ""
    if c["source"]:
        source = (f' · <a href="files/{quote(c["source"])}" target="_blank">'
                  f'{html.escape(c["source"])}</a>')
    return (f'<figure class="card {c["kind"]}" data-name="{name}">'
            f'<div class="media">{media}<button type="button" class="grow" title="Full size" '
            f'aria-label="Full size">&#x2922;</button></div>'
            f'<figcaption><span class="badge">{badge}</span>'
            f'<a class="name" href="{url}" download>{name}</a>'
            f'<span class="meta">{_size(c["size"])} · {when}{source}</span></figcaption>'
            f'</figure>')


def gallery(folder: Path) -> str:
    """The cards' HTML (the page swaps it in when the folder changes)."""
    return "".join(_card(c) for c in cards(folder)) or (
        '<p class="empty">No results yet. Make one above, or with <code>kami mesh photo.png</code> '
        'in a terminal: it shows up here.</p>')


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
        bodies = build_bodies(request)
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


def build_bodies(request: dict) -> list[tuple[str, dict]]:
    """(label, kami-server job body) for each job a page request asks for."""
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
            if not isinstance(image, dict):
                raise ValueError("Each image must be {name, data}")
            name = str(image.get("name") or "image")
            suffix = Path(name).suffix.lower()
            if suffix not in jobs.IMAGE_SUFFIXES:
                raise ValueError(f"{name}: not an image kami can send ({', '.join(jobs.IMAGE_SUFFIXES)})")
            try:
                data = base64.b64decode(image.get("data") or "", validate=True)
            except (binascii.Error, ValueError, TypeError):
                raise ValueError(f"{name}: the image data is not valid base64") from None
            if not data:
                raise ValueError(f"{name}: the image is empty")
            if len(data) > jobs.MAX_IMAGE_BYTES:
                raise ValueError(f"{name}: over {jobs.MAX_IMAGE_BYTES // 2**20} MiB, resize it first")
            view = image.get("view") or None
            if view is not None and view not in jobs.VIEW_NAMES:
                raise ValueError(f"{name}: the view must be one of {', '.join(jobs.VIEW_NAMES)}")
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
            bodies.append((label, {"kind": "image", "prompt": prompt,
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
                 {"kind": "text-to-mesh", "prompt": prompt, "name": jobs.safe_name(prompt),
                  "params": params})]
    raise ValueError("kind must be mesh, image or text-to-mesh")


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
              "FOLDER": html.escape(str(folder.resolve())), "GALLERY": gallery(folder),
              "SIGNATURE": signature(folder)}
    # one pass: a file named "{{TOKEN}}.png" stays a name, it is not filled in again
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)),
                  PAGE_FILE.read_text(encoding="utf-8"))


# --- the server ---

def make_handler(folder: Path, port_ref: list, token: str, work: Work, status: PodStatus,
                 backend):
    pod_jobs = PodJobs(backend)

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
                return self._json(200, {"signature": signature(folder), "pod": pod,
                                        "jobs": work.listing(listing)})
            if path == "/api/gallery":
                return self._send(200, gallery(folder).encode(), "text/html; charset=utf-8")
            if path.startswith("/files/"):
                name = path[len("/files/"):]
                target = folder / name
                # a plain file name of a result in the folder: no subpaths, no other files
                if (name and "/" not in name and "\\" not in name and name not in (".", "..")
                        and target.suffix.lower() in TYPES and target.is_file()):
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
