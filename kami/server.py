"""In-pod generation server: image -> 3D mesh (TRELLIS.2) and text -> image (diffusers).

This file is not imported by kami: like guard.py, its source is sent to the pod and run
there (`python3 -c`), under the guard, inside the kami-server container image (see
docker/Dockerfile), which brings torch, TRELLIS.2 and diffusers. Sending the source on
every `kami up` means a change here needs no image rebuild.

Generating a mesh takes from tens of seconds to minutes, longer than the RunPod proxy
keeps a request open (~100 s), so work goes through jobs: POST /v1/jobs answers at once
with an id, GET /v1/jobs/<id> tells its state, and GET /v1/jobs/<id>/files/<name>
downloads what it made. One worker runs the jobs one at a time (one GPU).

Environment (set by `kami up`):
  KAMI_PORT            port to listen on
  KAMI_API_KEY         bearer token every request but /health must carry
  KAMI_MESH_MODEL      Hugging Face repo of the TRELLIS.2 pipeline ("" = no mesh)
  KAMI_IMAGE_MODEL     Hugging Face repo of a diffusers pipeline ("" = no images)
  KAMI_IMAGE_DEFAULTS  JSON of default image parameters (steps, guidance, width, height)
  KAMI_MESH_DEFAULTS   JSON of default mesh parameters (texture_size, decimation_target...)
  KAMI_OFFLOAD         "1": keep the pipeline not in use on the CPU (both on a small GPU)

Endpoints:
  GET  /health                     200 once the pipelines are loaded (503 before)
  GET  /metrics                    Prometheus text: jobs running / queued / finished
  GET  /v1/info                    what this server can do, and the defaults
  POST /v1/jobs                    {"kind": "mesh"|"image"|"text-to-mesh", "image": <base64>,
                                    "prompt": str, "params": {...}} -> {"id": ...}
  GET  /v1/jobs/<id>               {"status": queued|running|done|error, "files": [...], ...}
  GET  /v1/jobs/<id>/files/<name>  the file's bytes
"""

import base64
import gc
import hmac
import inspect
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("KAMI_PORT", "8000"))
API_KEY = os.environ.get("KAMI_API_KEY", "")
MESH_MODEL = os.environ.get("KAMI_MESH_MODEL", "")
IMAGE_MODEL = os.environ.get("KAMI_IMAGE_MODEL", "")
OFFLOAD = os.environ.get("KAMI_OFFLOAD") == "1"
JOBS_DIR = os.environ.get("KAMI_JOBS_DIR", "/tmp/kami-jobs")
MAX_JOBS = 50                   # finished jobs kept (with their files); older ones are dropped
MAX_UPLOAD = 64 * 1024 * 1024   # bytes of a request body

IMAGE_DEFAULTS = {"steps": 8, "guidance": 0.0, "width": 1024, "height": 1024}
IMAGE_DEFAULTS.update(json.loads(os.environ.get("KAMI_IMAGE_DEFAULTS") or "{}"))
MESH_DEFAULTS = {"texture_size": 2048, "decimation_target": 500000, "remesh": True}
MESH_DEFAULTS.update(json.loads(os.environ.get("KAMI_MESH_DEFAULTS") or "{}"))
NVDIFFRAST_LIMIT = 16777216     # faces nvdiffrast can rasterise (TRELLIS.2's example)


def log(msg):
    print(f"[kami-server] {msg}", flush=True)


# --- pipelines ---

class Pipelines:
    """The loaded models. With OFFLOAD, the one not in use waits on the CPU."""

    def __init__(self):
        self.mesh = None
        self.image = None
        self.on_gpu = None
        self.ready = False
        self.error = None

    def load(self):
        import torch    # noqa: F401  (fail early, with a clear message, without torch)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        if MESH_MODEL:
            started = time.monotonic()
            log(f"loading mesh pipeline {MESH_MODEL}")
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            self.mesh = Trellis2ImageTo3DPipeline.from_pretrained(MESH_MODEL)
            if OFFLOAD:
                _to(self.mesh, "cpu")
            else:
                self.mesh.cuda()
                self.on_gpu = "mesh"
            log(f"mesh pipeline ready in {time.monotonic() - started:.0f} s")
        if IMAGE_MODEL:
            started = time.monotonic()
            log(f"loading image pipeline {IMAGE_MODEL}")
            from diffusers import DiffusionPipeline
            self.image = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16)
            if OFFLOAD:
                self.image.to("cpu")
            else:
                self.image.to("cuda")
                self.on_gpu = "image" if not self.mesh else self.on_gpu
            log(f"image pipeline ready in {time.monotonic() - started:.0f} s")
        self.ready = True

    def use(self, which):
        """Have `which` ("mesh" / "image") on the GPU, moving the other one out (OFFLOAD)."""
        if not OFFLOAD or self.on_gpu == which:
            return
        import torch
        other = {"mesh": self.image, "image": self.mesh}[which]
        if other is not None:
            _to(other, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        _to(self.mesh if which == "mesh" else self.image, "cuda")
        self.on_gpu = which


def _to(pipe, device):
    if hasattr(pipe, "to"):
        pipe.to(device)
    elif device == "cuda":
        pipe.cuda()
    else:
        pipe.cpu()


PIPES = Pipelines()


def _accepted(fn, kwargs):
    """The kwargs `fn` takes (pipelines differ between versions: pass what each knows)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def make_image(prompt, params, out_path):
    import torch
    p = {**IMAGE_DEFAULTS, **params}
    PIPES.use("image")
    kwargs = {"prompt": prompt, "num_inference_steps": int(p["steps"]),
              "guidance_scale": float(p["guidance"]),
              "width": int(p["width"]), "height": int(p["height"])}
    if p.get("negative"):
        kwargs["negative_prompt"] = p["negative"]
    if p.get("seed") is not None:
        kwargs["generator"] = torch.Generator("cuda").manual_seed(int(p["seed"]))
    image = PIPES.image(**_accepted(PIPES.image.__call__, kwargs)).images[0]
    image.save(out_path)
    return image


def make_mesh(image, params, out_path):
    import o_voxel
    p = {**MESH_DEFAULTS, **params}
    PIPES.use("mesh")
    kwargs = {"seed": int(p["seed"]) if p.get("seed") is not None else 42}
    if p.get("pipeline_type"):
        kwargs["pipeline_type"] = p["pipeline_type"]
    mesh = PIPES.mesh.run(image, **_accepted(PIPES.mesh.run, kwargs))[0]
    mesh.simplify(NVDIFFRAST_LIMIT)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(p["decimation_target"]), texture_size=int(p["texture_size"]),
        remesh=bool(p["remesh"]), remesh_band=1, remesh_project=0, verbose=False)
    glb.export(out_path, extension_webp=True)


# --- jobs ---

class Job:
    def __init__(self, kind, prompt, image_bytes, params, name):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.prompt, self.image_bytes, self.params = kind, prompt, image_bytes, params
        self.name = name or self.id
        self.status = "queued"
        self.error = None
        self.files = []
        self.created = time.time()
        self.started = self.finished = None
        self.dir = os.path.join(JOBS_DIR, self.id)

    def info(self):
        now = time.time()
        return {"id": self.id, "kind": self.kind, "status": self.status, "error": self.error,
                "files": self.files, "queued_seconds": round((self.started or now) - self.created, 1),
                "run_seconds": round((self.finished or now) - self.started, 1) if self.started else None,
                "position": QUEUE_ORDER.index(self.id) + 1 if self.id in QUEUE_ORDER else None}


JOBS = {}
QUEUE = queue.Queue()
QUEUE_ORDER = []                # ids of queued jobs, in order (for "position")
LOCK = threading.Lock()
COUNTS = {"finished": 0, "failed": 0, "running": 0}


def run(job):
    from PIL import Image
    os.makedirs(job.dir, exist_ok=True)
    if job.kind == "image":
        path = os.path.join(job.dir, f"{job.name}.png")
        make_image(job.prompt, job.params, path)
        job.files.append(os.path.basename(path))
        return
    if job.kind == "text-to-mesh":
        path = os.path.join(job.dir, f"{job.name}.png")
        image = make_image(job.prompt, job.params.get("image", {}), path)
        job.files.append(os.path.basename(path))
    else:
        image = Image.open(io.BytesIO(job.image_bytes))
        image.load()
    path = os.path.join(job.dir, f"{job.name}.glb")
    make_mesh(image, job.params.get("mesh", job.params), path)
    job.files.append(os.path.basename(path))


def worker():
    while True:
        job = QUEUE.get()
        with LOCK:
            if job.id in QUEUE_ORDER:
                QUEUE_ORDER.remove(job.id)
            job.status, job.started = "running", time.time()
            COUNTS["running"] = 1
        log(f"job {job.id} ({job.kind}) started")
        try:
            run(job)
            job.status = "done"
            COUNTS["finished"] += 1
            log(f"job {job.id} done in {time.time() - job.started:.0f} s: {', '.join(job.files)}")
        except Exception as e:
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            COUNTS["failed"] += 1
            log(f"job {job.id} failed: {job.error}")
            traceback.print_exc()
            _free_gpu()
        finally:
            job.finished = time.time()
            job.image_bytes = None
            COUNTS["running"] = 0
            prune()


def _free_gpu():
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def prune():
    with LOCK:
        done = sorted((j for j in JOBS.values() if j.finished), key=lambda j: j.finished)
        for job in done[:-MAX_JOBS]:
            JOBS.pop(job.id, None)
            shutil.rmtree(job.dir, ignore_errors=True)


KINDS = ("mesh", "image", "text-to-mesh")


def submit(body):
    """A new job from a request body, or raise ValueError with what is wrong with it."""
    kind = body.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind in ("mesh", "text-to-mesh") and PIPES.mesh is None:
        raise ValueError("this pod serves no mesh model (profile without mesh_model)")
    if kind in ("image", "text-to-mesh") and PIPES.image is None:
        raise ValueError("this pod serves no image model (profile without image_model)")
    image_bytes = None
    if kind == "mesh":
        try:
            image_bytes = base64.b64decode(body.get("image") or "", validate=True)
        except ValueError:
            raise ValueError("image must be base64") from None
        if not image_bytes:
            raise ValueError("a mesh job needs an image")
    prompt = body.get("prompt") or ""
    if kind != "mesh" and not prompt.strip():
        raise ValueError(f"a {kind} job needs a prompt")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    name = "".join(c for c in str(body.get("name") or "") if c.isalnum() or c in "-_")[:60]
    job = Job(kind, prompt, image_bytes, params, name)
    with LOCK:
        JOBS[job.id] = job
        QUEUE_ORDER.append(job.id)
    QUEUE.put(job)
    log(f"job {job.id} ({kind}) queued")
    return job


# --- HTTP ---

class Handler(BaseHTTPRequestHandler):
    server_version = "kami-server"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # one line per request is noise in `kami logs`
        pass

    def send(self, code, body=b"", content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        given = self.headers.get("Authorization", "")
        if API_KEY and hmac.compare_digest(given, f"Bearer {API_KEY}"):
            return True
        self.send(401, {"error": "missing or wrong API key"})
        return False

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            if PIPES.error:
                return self.send(500, {"status": "error", "error": PIPES.error})
            return self.send(200 if PIPES.ready else 503,
                             {"status": "ok" if PIPES.ready else "loading"})
        if not self.authorized():
            return
        if path == "/metrics":
            queued = sum(1 for j in JOBS.values() if j.status == "queued")
            text = (f"kami:jobs_running {COUNTS['running']}\n"
                    f"kami:jobs_queued {queued}\n"
                    f"kami:jobs_finished_total {COUNTS['finished']}\n"
                    f"kami:jobs_failed_total {COUNTS['failed']}\n")
            return self.send(200, text.encode(), "text/plain; version=0.0.4")
        if path == "/v1/info":
            return self.send(200, {"mesh_model": MESH_MODEL or None, "image_model": IMAGE_MODEL or None,
                                   "mesh_defaults": MESH_DEFAULTS, "image_defaults": IMAGE_DEFAULTS,
                                   "offload": OFFLOAD, "ready": PIPES.ready})
        parts = path.split("/")
        # /v1/jobs/<id> and /v1/jobs/<id>/files/<name>
        if len(parts) >= 4 and parts[1:3] == ["v1", "jobs"]:
            job = JOBS.get(parts[3])
            if job is None:
                return self.send(404, {"error": "no such job (finished jobs are kept for a while only)"})
            if len(parts) == 4:
                return self.send(200, job.info())
            if len(parts) == 6 and parts[4] == "files" and parts[5] in job.files:
                with open(os.path.join(job.dir, parts[5]), "rb") as f:
                    data = f.read()
                kind = "image/png" if parts[5].endswith(".png") else "model/gltf-binary"
                return self.send(200, data, kind)
            return self.send(404, {"error": "no such file"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path.rstrip("/") != "/v1/jobs":
            return self.send(404, {"error": "not found"})
        if not PIPES.ready:
            return self.send(503, {"error": "the pipelines are still loading"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            return self.send(413, {"error": f"request over {MAX_UPLOAD // 2**20} MiB"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            job = submit(body if isinstance(body, dict) else {})
        except (ValueError, json.JSONDecodeError) as e:
            return self.send(400, {"error": str(e)})
        self.send(202, job.info())


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)
    if not API_KEY:
        log("KAMI_API_KEY is not set: refusing to serve without authentication")
        sys.exit(2)
    if not MESH_MODEL and not IMAGE_MODEL:
        log("neither KAMI_MESH_MODEL nor KAMI_IMAGE_MODEL is set: nothing to serve")
        sys.exit(2)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"listening on :{PORT} (health 503 until the pipelines load)")
    try:
        PIPES.load()
    except Exception as e:
        PIPES.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        log(f"could not load the pipelines: {PIPES.error}")
        sys.exit(1)
    threading.Thread(target=worker, daemon=True).start()
    log("server is ready")
    while True:
        time.sleep(3600)


if __name__ == "__main__":  # also true under `python3 -c`
    main()
