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
  KAMI_DINO_MODEL      repo replacing the DINOv3 model TRELLIS.2's pipeline.json names
  KAMI_REMBG_MODEL     repo replacing its background remover (BiRefNet weights)
  KAMI_OFFLOAD         "1": keep the pipeline not in use on the CPU (both on a small GPU)
  KAMI_MESH_BACKEND    "trellis2" (default) or "hunyuan" (KAMI_MESH_MODEL is a Hunyuan3D-2 repo)
  KAMI_MESH_SUBFOLDER  Hunyuan3D-2: the repo folder of the shape model (-mv: named views)
  KAMI_PAINT_MODEL     Hunyuan3D-2: the repo of Hunyuan3D-Paint (texture)

Endpoints:
  GET  /health                     200 once the pipelines are loaded (503 before)
  GET  /metrics                    Prometheus text: jobs running / queued / finished
  GET  /v1/info                    what this server can do, and the defaults
  POST /v1/jobs                    {"kind": "mesh"|"image"|"text-to-mesh", "image": <base64>,
                                    "prompt": str, "params": {...}} -> {"id": ...}; or, for
                                    several views of one object, "images": [{"image":
                                    <base64>, "view": "front"|"left"|"back"|"right"|null}]
  GET  /v1/jobs                    every job kept: running, queue (in order), finished
  GET  /v1/jobs/<id>               {"status": queued|running|done|error, "stage", "step",
                                    "steps", "progress" (0..1), "position", "files", ...}
  GET  /v1/jobs/<id>/files/<name>  the file's bytes
"""

import base64
import contextlib
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
DINO_MODEL = os.environ.get("KAMI_DINO_MODEL", "")
REMBG_MODEL = os.environ.get("KAMI_REMBG_MODEL", "")
JOBS_DIR = os.environ.get("KAMI_JOBS_DIR", "/tmp/kami-jobs")
MAX_JOBS = 50                   # finished jobs kept (with their files); older ones are dropped
MAX_UPLOAD = 200 * 1024 * 1024  # bytes of a request body (several views, base64)
# `kami reload`: the new source is saved here and the server exits with RELOAD_EXIT; the
# guard starts it again, and the boot command runs this file instead of KAMI_SERVER_SRC.
RELOAD_PATH = os.environ.get("KAMI_RELOAD_PATH", "/tmp/kami-server.py")
RELOAD_EXIT = 75
STARTED_AT = time.time()

IMAGE_DEFAULTS = {"steps": 8, "guidance": 0.0, "width": 1024, "height": 1024}
IMAGE_DEFAULTS.update(json.loads(os.environ.get("KAMI_IMAGE_DEFAULTS") or "{}"))
# "trellis2" (TRELLIS.2) or "hunyuan" (Hunyuan3D-2: shape model + Hunyuan3D-Paint texture)
MESH_BACKEND = os.environ.get("KAMI_MESH_BACKEND") or "trellis2"
MESH_SUBFOLDER = os.environ.get("KAMI_MESH_SUBFOLDER") or "hunyuan3d-dit-v2-0"
PAINT_MODEL = os.environ.get("KAMI_PAINT_MODEL") or "tencent/Hunyuan3D-2"
HUNYUAN_MV = MESH_BACKEND == "hunyuan" and "-mv" in MESH_SUBFOLDER
if MESH_BACKEND == "hunyuan":
    # steps, octree resolution and guidance of Hunyuan3D-2's examples; the faces kept
    # before texturing as in its demo app (Hunyuan3D-Paint unwraps the UVs of all of them)
    MESH_DEFAULTS = {"steps": 50, "octree_resolution": 380, "guidance": 5.0,
                     "decimation_target": 40000, "texture": True}
else:
    MESH_DEFAULTS = {"texture_size": 2048, "decimation_target": 500000, "remesh": True,
                     "multiview_mode": "stochastic"}
MESH_DEFAULTS.update(json.loads(os.environ.get("KAMI_MESH_DEFAULTS") or "{}"))
NVDIFFRAST_LIMIT = 16777216     # faces nvdiffrast can rasterise (TRELLIS.2's example)
# Views of one object a mesh job may carry: Hunyuan3D-2mv knows these four (its
# conditioner embeds which one each image is); TRELLIS.2 takes any views, unnamed.
HUNYUAN_VIEWS = ("front", "left", "back", "right")
MAX_VIEWS = 8
MULTIVIEW_MODES = ("stochastic", "multidiffusion")


def log(msg):
    print(f"[kami-server] {msg}", flush=True)


# --- pipelines ---

def fit_transformers(pipeline):
    """Bridge TRELLIS.2 and transformers 5. TRELLIS.2's DINOv3 feature extractor walks
    `model.layer`; transformers 5 moved the layers to `model.model.layer` (same layers,
    same call). Alias them, outside the module tree so nothing is registered twice."""
    dino = getattr(getattr(pipeline, "image_cond_model", None), "model", None)
    if dino is None or "layer" in dino.__dict__ or "layer" in getattr(dino, "_modules", {}):
        return
    inner = getattr(dino, "_modules", {}).get("model")
    layers = getattr(inner, "_modules", {}).get("layer") if inner is not None else None
    if layers is not None:
        object.__setattr__(dino, "layer", layers)
        log("DINOv3: transformers 5 layout, layers aliased for TRELLIS.2")


def override_model(module, name, repo):
    """Make `module.<name>(model_name=...)` load `repo` instead. TRELLIS.2 builds its
    sub-models with getattr(module, name)(**pipeline.json args), so this swaps the weights
    a pipeline.json names without editing it."""
    cls = getattr(module, name, None)
    if cls is None:
        log(f"cannot replace {name}: TRELLIS.2 has no such class (left as it is)")
        return
    def make(*args, **kwargs):
        kwargs["model_name"] = repo
        return cls(*args, **kwargs)
    setattr(module, name, make)
    log(f"{name} loads {repo}")


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
        if MESH_MODEL and MESH_BACKEND == "hunyuan":
            started = time.monotonic()
            log(f"loading Hunyuan3D-2 {MESH_MODEL} ({MESH_SUBFOLDER}) + {PAINT_MODEL} (texture)")
            self.mesh = HunyuanMesh()
            if OFFLOAD:
                _to(self.mesh, "cpu")
            else:
                self.on_gpu = "mesh"
            log(f"mesh pipeline ready in {time.monotonic() - started:.0f} s")
        elif MESH_MODEL:
            started = time.monotonic()
            log(f"loading mesh pipeline {MESH_MODEL}")
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            if DINO_MODEL:
                from trellis2.modules import image_feature_extractor
                override_model(image_feature_extractor, "DinoV3FeatureExtractor", DINO_MODEL)
            if REMBG_MODEL:
                from trellis2.pipelines import rembg
                override_model(rembg, "BiRefNet", REMBG_MODEL)
            self.mesh = Trellis2ImageTo3DPipeline.from_pretrained(MESH_MODEL)
            fit_transformers(self.mesh)
            track_mesh_progress(self.mesh)
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
        stage("to_gpu")
        other = {"mesh": self.image, "image": self.mesh}[which]
        if other is not None:
            _to(other, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        _to(self.mesh if which == "mesh" else self.image, "cuda")
        self.on_gpu = which


# transformers 5 renamed DINOv2's weights; its from_pretrained converts old checkpoints,
# but hy3dgen builds Dinov2Model(config) and calls load_state_dict with the transformers 4
# names. The same renames as transformers' conversion_mapping ("Dinov2Model").
DINOV2_RENAMES = (("attention.attention.query.", "attention.q_proj."),
                  ("attention.attention.key.", "attention.k_proj."),
                  ("attention.attention.value.", "attention.v_proj."),
                  ("attention.output.dense.", "attention.o_proj."),
                  ("mlp.weights_out.", "mlp.down_proj."))


def dinov2_v5_state(state, expected):
    """`state` (a transformers 4 DINOv2 checkpoint, possibly inside a bigger model) with the
    transformers 5 names, when `expected` (the module's own keys) uses them; else as it is.
    mlp.weights_in is split in two: the first half is gate_proj, the second up_proj."""
    if not any(".q_proj." in k or ".gate_proj." in k for k in expected) or not any(
            "attention.attention.query." in k or "mlp.weights_in." in k for k in state):
        return state
    out = type(state)()
    for key, value in state.items():
        if "mlp.weights_in." in key:
            gate, up = value.chunk(2, dim=0)
            out[key.replace("mlp.weights_in.", "mlp.gate_proj.")] = gate
            out[key.replace("mlp.weights_in.", "mlp.up_proj.")] = up
            continue
        for old, new in DINOV2_RENAMES:
            key = key.replace(old, new)
        out[key] = value
    return out


@contextlib.contextmanager
def dinov2_v4_checkpoints():
    """While loading hy3dgen: load_state_dict converts transformers 4 DINOv2 weights."""
    import torch
    original = torch.nn.Module.load_state_dict

    def load_state_dict(module, state_dict, *args, **kwargs):
        converted = dinov2_v5_state(state_dict, module.state_dict().keys())
        if converted is not state_dict:
            log(f"{type(module).__name__}: DINOv2 weights renamed for transformers 5")
        return original(module, converted, *args, **kwargs)
    torch.nn.Module.load_state_dict = load_state_dict
    try:
        yield
    finally:
        torch.nn.Module.load_state_dict = original


HUNYUAN_CODE = "/opt/Hunyuan3D-2"      # the image's clone, at a pinned commit


@contextlib.contextmanager
def trust_hunyuan_pipelines():
    """While loading Hunyuan3D-Paint: newer diffusers runs a local `custom_pipeline` only
    with trust_remote_code=True. Given only for hy3dgen's own pipeline code, which is the
    image's pinned clone, not code downloaded at run time."""
    from diffusers import DiffusionPipeline
    original = DiffusionPipeline.__dict__["from_pretrained"]
    load = original.__func__

    def from_pretrained(cls, *args, **kwargs):
        custom = kwargs.get("custom_pipeline")
        if isinstance(custom, str) and os.path.realpath(custom).startswith(HUNYUAN_CODE + "/"):
            kwargs.setdefault("trust_remote_code", True)
        return load(cls, *args, **kwargs)
    DiffusionPipeline.from_pretrained = classmethod(from_pretrained)
    try:
        yield
    finally:
        DiffusionPipeline.from_pretrained = original


class HunyuanMesh:
    """Hunyuan3D-2: its shape model (one image, or named views with -2mv), Hunyuan3D-Paint
    for the texture, and RMBG-2.0 for the background (the BiRefNet TRELLIS.2 brings)."""

    def __init__(self):
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from trellis2.pipelines.rembg import BiRefNet
        with dinov2_v4_checkpoints(), trust_hunyuan_pipelines():
            self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                MESH_MODEL, subfolder=MESH_SUBFOLDER, variant="fp16")
            self.paint = Hunyuan3DPaintPipeline.from_pretrained(PAINT_MODEL)
        self.rembg = BiRefNet(model_name=REMBG_MODEL or "briaai/RMBG-2.0")
        self.rembg.cuda()

    def to(self, device):
        self.shape.to(device)
        for model in self.paint.models.values():
            pipe = getattr(model, "pipeline", None)
            if pipe is not None:
                pipe.to(device)
        self.rembg.to(device)

    def cutout(self, image):
        """The image with its background removed (kept as it is if it already has one)."""
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            return image
        return self.rembg(image.convert("RGB"))


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
    stage("image")
    steps = int(p["steps"])

    def on_step(pipe, i, t, tensors):
        step(i + 1, steps)
        return tensors
    kwargs = {"prompt": prompt, "num_inference_steps": steps,
              "guidance_scale": float(p["guidance"]),
              "width": int(p["width"]), "height": int(p["height"]),
              "callback_on_step_end": on_step}
    if p.get("negative"):
        kwargs["negative_prompt"] = p["negative"]
    if p.get("seed") is not None:
        kwargs["generator"] = torch.Generator("cuda").manual_seed(int(p["seed"]))
    image = PIPES.image(**_accepted(PIPES.image.__call__, kwargs)).images[0]
    image.save(out_path)
    return image


def make_mesh(views, params, out_path):
    """`views`: [(view name or None, PIL image)], one or several views of one object."""
    p = {**MESH_DEFAULTS, **params}
    PIPES.use("mesh")
    if MESH_BACKEND == "hunyuan":
        return make_mesh_hunyuan(views, p, out_path)
    return make_mesh_trellis(views, p, out_path)


@contextlib.contextmanager
def trellis_multiview(pipe, views, mode):
    """TRELLIS.2 conditioned on several views of one object -- the tuning-free trick of
    TRELLIS (1)'s run_multi_image, which TRELLIS.2 does not ship: the image features of
    every view go through the samplers, and at each step the flow model sees one view
    in turn ("stochastic") or all of them, averaged ("multidiffusion", N times slower).
    The model was trained on single images: consistent views help, mismatched ones hurt."""
    if mode not in MULTIVIEW_MODES:
        raise ValueError(f"multiview_mode must be one of {', '.join(MULTIVIEW_MODES)}")
    n = len(views)
    get_cond = pipe.get_cond
    own = "get_cond" in vars(pipe)        # the progress wrapper sets one on the instance

    def multi_cond(image, resolution, include_neg_cond=True):
        cond = get_cond(views, resolution, include_neg_cond)
        if "neg_cond" in cond:
            cond["neg_cond"] = cond["neg_cond"][:1]      # one unconditional branch
        return cond

    samplers = {}
    for name in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
        sampler = getattr(pipe, name, None)
        if sampler is not None:
            samplers[id(sampler)] = sampler
    for sampler in samplers.values():
        inner = sampler._inference_model                 # the class's (guidance mixins)
        if mode == "stochastic":
            calls = [0]

            def model(flow, x_t, t, cond=None, _inner=inner, _calls=calls, **kwargs):
                i = _calls[0] % n
                _calls[0] += 1
                return _inner(flow, x_t, t, cond[i:i + 1], **kwargs)
        else:
            def model(flow, x_t, t, cond=None, _inner=inner, **kwargs):
                # guidance is linear: the mean of each view's guided prediction is the
                # guided mean of their predictions
                return sum(_inner(flow, x_t, t, cond[i:i + 1], **kwargs) for i in range(n)) / n
        sampler._inference_model = model
    object.__setattr__(pipe, "get_cond", multi_cond)
    try:
        yield
    finally:
        for sampler in samplers.values():
            sampler.__dict__.pop("_inference_model", None)
        if own:
            object.__setattr__(pipe, "get_cond", get_cond)
        else:
            pipe.__dict__.pop("get_cond", None)


def make_mesh_trellis(views, p, out_path):
    import o_voxel
    pipe = PIPES.mesh
    kwargs = {"seed": int(p["seed"]) if p.get("seed") is not None else 42}
    if p.get("pipeline_type"):
        kwargs["pipeline_type"] = p["pipeline_type"]
    if len(views) == 1:
        mesh = pipe.run(views[0][1], **_accepted(pipe.run, kwargs))[0]
    else:
        images = [pipe.preprocess_image(image) for _, image in views]
        log(f"{len(images)} views, multiview mode {p['multiview_mode']}")
        with trellis_multiview(pipe, images, p["multiview_mode"]):
            mesh = pipe.run(images[0], preprocess_image=False, **_accepted(pipe.run, kwargs))[0]
    mesh.simplify(NVDIFFRAST_LIMIT)
    stage("glb")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(p["decimation_target"]), texture_size=int(p["texture_size"]),
        remesh=bool(p["remesh"]), remesh_band=1, remesh_project=0, verbose=False)
    # PNG textures, not EXT_texture_webp: the image's Pillow (pillow-simd, from TRELLIS.2's
    # setup.sh) cannot encode them as WebP ("unsupported image mode"), and many viewers and
    # slicers do not read WebP textures in a GLB anyway.
    glb.export(out_path)


def make_mesh_hunyuan(views, p, out_path):
    import torch
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover
    h = PIPES.mesh
    stage("prepare")
    views = [(name, h.cutout(image)) for name, image in views]
    image = dict(views) if HUNYUAN_MV else views[0][1]
    steps = int(p["steps"])
    stage("shape")

    def on_step(i, t, outputs):
        step(i + 1, steps)
        if i + 1 >= steps:
            stage("decode")                  # then the volume is decoded into a surface
    seed = int(p["seed"]) if p.get("seed") is not None else 42
    mesh = h.shape(image=image, num_inference_steps=steps,
                   octree_resolution=int(p["octree_resolution"]), num_chunks=20000,
                   guidance_scale=float(p["guidance"]), generator=torch.manual_seed(seed),
                   output_type="trimesh", callback=on_step, callback_steps=1)[0]
    stage("decode")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=int(p["decimation_target"]))
    if p.get("texture", True):
        stage("texture")
        # Hunyuan3D-Paint paints from one image: the front view when there are several
        front = dict(views).get("front") or views[0][1]
        mesh = h.paint(mesh, image=front)
    stage("glb")
    mesh.export(out_path)


# --- jobs ---

# --- progress ---
# A job's stages, with the share of its time each one takes (rough, from a 4090): the
# progress is the stages done plus the part of the current one its sampling loop has done.
STAGES = {
    "image": "Generating the image",
    "prepare": "Removing the background",
    "condition": "Reading the image (DINOv3)",
    "structure": "Sampling the sparse structure",
    "shape": "Sampling the shape",
    "texture": "Sampling the texture",
    "decode": "Decoding the mesh",
    "glb": "Baking the GLB (remesh, texture)",
    "to_gpu": "Moving the model to the GPU",       # offload only; no share of its own
}
MESH_PLAN = (("prepare", .04), ("condition", .04), ("structure", .12), ("shape", .34),
             ("texture", .22), ("decode", .08), ("glb", .16))
# Hunyuan3D-2: shape diffusion, surface extraction (+ clean-up), then Hunyuan3D-Paint
HUNYUAN_PLAN = (("prepare", .03), ("shape", .40), ("decode", .12), ("texture", .40), ("glb", .05))
PLANS = {
    "image": (("image", 1.0),),
    "mesh": MESH_PLAN,
    "text-to-mesh": (("image", .2),) + tuple((k, w * .8) for k, w in MESH_PLAN),
}


HUNYUAN_STAGES = {"shape": "Sampling the shape (Hunyuan3D-2)",
                  "decode": "Extracting and cleaning the surface",
                  "texture": "Painting the texture (Hunyuan3D-Paint)", "glb": "Saving the GLB"}


def stage_label(key):
    if MESH_BACKEND == "hunyuan" and key in HUNYUAN_STAGES:
        return HUNYUAN_STAGES[key]
    return STAGES.get(key)


def plan_for(kind):
    mesh = HUNYUAN_PLAN if MESH_BACKEND == "hunyuan" else MESH_PLAN
    if kind == "mesh":
        return mesh
    if kind == "text-to-mesh":
        return (("image", .2),) + tuple((k, w * .8) for k, w in mesh)
    return PLANS.get(kind, ())
CURRENT = [None]                # the job the worker runs (one at a time)


def stage(key, loops=1):
    """The running job entered `key`; `loops` sampling loops are expected in it."""
    job = CURRENT[0]
    if job is not None:
        job.enter(key, loops)


def step(n, total):
    """The running job's current sampling loop has done n of total steps."""
    job = CURRENT[0]
    if job is not None and total:
        job.advance(n, total)


def tracked_tqdm(orig):
    """TRELLIS.2's samplers loop over `tqdm(steps, ...)`: the same bars in the log, and
    each step reported to the running job."""
    def tqdm(iterable=None, *args, **kwargs):
        bar = orig(iterable, *args, **kwargs)
        if iterable is None:
            return bar
        try:
            total = len(iterable)
        except TypeError:
            total = kwargs.get("total")

        def steps():
            for i, item in enumerate(bar, 1):
                yield item
                step(i, total)
        return steps()
    tqdm.kami_tracked = True
    return tqdm


# pipeline method -> (stage, sampling loops in it); TRELLIS.2's run() calls them in order
MESH_METHODS = (("preprocess_image", "prepare", 1), ("get_cond", "condition", 1),
                ("sample_sparse_structure", "structure", 1), ("sample_shape_slat", "shape", 1),
                ("sample_shape_slat_cascade", "shape", 2), ("sample_tex_slat", "texture", 1),
                ("decode_latent", "decode", 1))


def track_mesh_progress(pipeline):
    """Report TRELLIS.2's stages and sampling steps. Methods a version lacks are skipped:
    the progress then moves in bigger jumps, nothing breaks."""
    for name, key, loops in MESH_METHODS:
        method = getattr(pipeline, name, None)
        if not callable(method):
            continue

        def wrapped(*args, _method=method, _key=key, _loops=loops, **kwargs):
            stage(_key, _loops)
            return _method(*args, **kwargs)
        object.__setattr__(pipeline, name, wrapped)
    # only the samplers: they loop over the bar and nothing else (other code may call the
    # bar's methods, which the generator returned in its place does not have)
    for module in [m for n, m in list(sys.modules.items())
                   if n.startswith("trellis2.pipelines.samplers") and m]:
        orig = getattr(module, "tqdm", None)
        if callable(orig) and not getattr(orig, "kami_tracked", False):
            module.tqdm = tracked_tqdm(orig)


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
        self.plan = plan_for(kind)
        # views of the object (image_bytes is dropped once the job ends)
        self.view_count = (len(image_bytes) if isinstance(image_bytes, list)
                           else 1 if kind == "mesh" else None)
        self.stage = None
        self.step = self.steps = None
        self.progress = 0.0
        self._base = self._share = 0.0
        self._loops = 1
        self._loops_done = 0

    def enter(self, key, loops=1):
        if key == self.stage:
            return                      # get_cond runs twice (512 and 1024): one stage
        self.stage, self.step, self.steps = key, None, None
        keys = [k for k, _ in self.plan]
        if key in keys:                 # a stage of the plan: what came before is done
            i = keys.index(key)
            self._base = sum(w for _, w in self.plan[:i])
            self._share = self.plan[i][1]
            self._loops, self._loops_done = max(1, loops), 0
            self.progress = max(self.progress, self._base)

    def advance(self, n, total):
        self.step, self.steps = n, total
        done = min(1.0, (self._loops_done + n / total) / self._loops)
        self.progress = max(self.progress, min(1.0, self._base + self._share * done))
        if n >= total:
            self._loops_done += 1

    def info(self):
        now = time.time()
        return {"id": self.id, "kind": self.kind, "name": self.name, "status": self.status,
                "views": self.view_count,
                "error": self.error, "files": self.files,
                "prompt": self.prompt[:200] if self.prompt else None,
                "stage": self.stage if self.status == "running" else None,
                "stage_label": stage_label(self.stage) if self.status == "running" else None,
                "step": self.step if self.status == "running" else None,
                "steps": self.steps if self.status == "running" else None,
                "progress": 1.0 if self.status == "done" else round(self.progress, 3),
                "created": self.created, "started": self.started, "finished": self.finished,
                "queued_seconds": round((self.started or now) - self.created, 1),
                "run_seconds": round((self.finished or now) - self.started, 1) if self.started else None,
                "position": QUEUE_ORDER.index(self.id) + 1 if self.id in QUEUE_ORDER else None}


def listing():
    """Every job the pod keeps: the running one, the queue in order, then the finished
    ones, newest first."""
    with LOCK:
        jobs = list(JOBS.values())
        order = list(QUEUE_ORDER)
    running = [j for j in jobs if j.status == "running"]
    queued = sorted((j for j in jobs if j.status == "queued"),
                    key=lambda j: order.index(j.id) if j.id in order else len(order))
    finished = sorted((j for j in jobs if j.status in ("done", "error")),
                      key=lambda j: j.finished or 0, reverse=True)
    return {"jobs": [j.info() for j in running + queued + finished],
            "running": len(running), "queued": len(queued)}


JOBS = {}
QUEUE = queue.Queue()
QUEUE_ORDER = []                # ids of queued jobs, in order (for "position")
LOCK = threading.Lock()
COUNTS = {"finished": 0, "failed": 0, "running": 0}


def open_image(data):
    """The uploaded image as an RGB / RGBA PIL image, whatever it came as: PNG, JPEG, WebP,
    AVIF, HEIC (pillow-heif), TIFF, BMP, GIF (its first frame). A photo is turned as its
    EXIF says; palettes, grayscale and 16-bit images are converted. If Pillow cannot
    read it, OpenCV decodes it instead (older images mixed pillow and pillow-simd, and
    that Pillow failed on WebP: "too many values to unpack")."""
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as e:
        image = opencv_image(data, e)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass                              # a broken EXIF block: keep it as it is
    return plain_mode(image)


# An alpha channel counts as a cut-out when at least this share of the image is
# (nearly) transparent. TRELLIS.2 skips background removal for any image with alpha,
# so an almost opaque one (a WebP whose alpha only fades a few edge pixels) made a mesh
# of the whole square, background included.
CUTOUT_SHARE = 0.01
CUTOUT_ALPHA = 26                         # alpha below ~10 %: transparent


def plain_mode(image):
    """RGB, or RGBA when the image is really cut out (see CUTOUT_SHARE)."""
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        image = image.point(lambda v: v / 257).convert("L")
    elif image.mode == "F":
        image = image.convert("L")
    if image.mode != "RGBA":
        transparent = "A" in image.getbands() or "transparency" in image.info
        if not transparent:
            return image if image.mode == "RGB" else image.convert("RGB")
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    clear = sum(alpha.histogram()[:CUTOUT_ALPHA])
    if clear < CUTOUT_SHARE * image.width * image.height:
        return image.convert("RGB")       # not a cut-out: background removal runs
    return image


def opencv_image(data, error):
    """Decode with OpenCV what Pillow could not (`error`, raised again if OpenCV cannot)."""
    from PIL import Image
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise error from None
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError(f"cannot read the image ({type(error).__name__}: {error})") from None
    if pixels.dtype != np.uint8:          # 16-bit PNG / TIFF
        pixels = (pixels / 257).astype(np.uint8)
    log(f"Pillow could not read an image ({type(error).__name__}): decoded with OpenCV")
    if pixels.ndim == 2:
        return Image.fromarray(pixels, "L")
    code = cv2.COLOR_BGRA2RGBA if pixels.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return Image.fromarray(cv2.cvtColor(pixels, code))


def run(job):
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
        views = [("front", image)]
    else:
        # image_bytes: [(view, bytes)] (submit's), or the bytes of one image
        given = job.image_bytes if isinstance(job.image_bytes, list) else [(None, job.image_bytes)]
        views = [(view, open_image(data)) for view, data in given]
    path = os.path.join(job.dir, f"{job.name}.glb")
    make_mesh(views, job.params.get("mesh", job.params), path)
    job.files.append(os.path.basename(path))


def worker():
    while True:
        job = QUEUE.get()
        with LOCK:
            if job.id in QUEUE_ORDER:
                QUEUE_ORDER.remove(job.id)
            job.status, job.started = "running", time.time()
            COUNTS["running"] = 1
            CURRENT[0] = job
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
            CURRENT[0] = None
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


def view_limits():
    """(how many views of one object a mesh job may carry, their names or None for any)."""
    if MESH_BACKEND == "hunyuan":
        return (len(HUNYUAN_VIEWS), HUNYUAN_VIEWS) if HUNYUAN_MV else (1, None)
    return MAX_VIEWS, None


def mesh_views(body):
    """[(view name or None, image bytes)] of a mesh job: "image" (one), or "images"
    ([{"image": <base64>, "view": "front"}, ...]: several views of one object)."""
    if body.get("images") is not None:
        items = body["images"]
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise ValueError('images must be a list of {"image": <base64>, "view": <name>}')
    else:
        items = [{"image": body.get("image")}]
    if not items:
        raise ValueError("a mesh job needs an image")
    most, names = view_limits()
    if len(items) > most:
        if most == 1:
            raise ValueError(f"{MESH_MODEL} makes a mesh from one image: several views need "
                             "TRELLIS.2 or tencent/Hunyuan3D-2mv")
        raise ValueError(f"at most {most} views of one object")
    views = []
    for item in items:
        try:
            data = base64.b64decode(item.get("image") or "", validate=True)
        except ValueError:
            raise ValueError("image must be base64") from None
        if not data:
            raise ValueError("a mesh job needs an image")
        view = item.get("view") or None
        if view is not None and not isinstance(view, str):
            raise ValueError("view must be a name")
        views.append([view, data])
    if names:
        # Hunyuan3D-2mv tells views apart by name: unnamed ones take the free names in
        # order (front, left, back, right)
        given = [v for v, _ in views if v]
        unknown = [v for v in given if v not in names]
        if unknown:
            raise ValueError(f"unknown view {unknown[0]!r}: one of {', '.join(names)}")
        if len(set(given)) != len(given):
            raise ValueError("each view may appear once")
        free = [n for n in names if n not in given]
        for item in views:
            if not item[0]:
                item[0] = free.pop(0)
    return [tuple(v) for v in views]


def submit(body):
    """A new job from a request body, or raise ValueError with what is wrong with it."""
    kind = body.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind in ("mesh", "text-to-mesh") and PIPES.mesh is None:
        raise ValueError("this pod serves no mesh model (profile without mesh_model)")
    if kind in ("image", "text-to-mesh") and PIPES.image is None:
        raise ValueError("this pod serves no image model (profile without image_model)")
    image_bytes = mesh_views(body) if kind == "mesh" else None
    prompt = body.get("prompt") or ""
    if kind != "mesh" and not prompt.strip():
        raise ValueError(f"a {kind} job needs a prompt")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    mode = (params.get("mesh", params) if isinstance(params.get("mesh", params), dict) else {}).get(
        "multiview_mode")
    if mode is not None and mode not in MULTIVIEW_MODES:
        raise ValueError(f"multiview_mode must be one of {', '.join(MULTIVIEW_MODES)}")
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
                                   "offload": OFFLOAD, "ready": PIPES.ready,
                                   "reload": True, "started_at": STARTED_AT,
                                   "job_listing": True, "mesh_backend": MESH_BACKEND if MESH_MODEL else None,
                                   "max_views": view_limits()[0] if MESH_MODEL else 0,
                                   "view_names": list(view_limits()[1] or []) or None,
                                   "multiview_modes": list(MULTIVIEW_MODES)
                                   if MESH_BACKEND == "trellis2" else None})
        if path == "/v1/jobs":
            return self.send(200, listing())
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

    def reload(self):
        """Swap in new server code without a new pod: check it compiles, save it, answer,
        then exit with RELOAD_EXIT so the guard starts it. The weights stay on disk; only
        the pipelines load again. Refused while a job runs or waits, unless forced."""
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(min(length, MAX_UPLOAD)) or b"{}")
            source = body["source"]
            compile(source, "kami-server", "exec")
        except (ValueError, KeyError, TypeError, SyntaxError) as e:
            return self.send(400, {"error": f"not valid server code: {type(e).__name__}: {e}"})
        busy = COUNTS["running"] + len(QUEUE_ORDER)
        if busy and not body.get("force"):
            return self.send(409, {"error": f"{busy} job(s) running or queued; wait, or force"})
        tmp = RELOAD_PATH + ".new"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(source)
        os.replace(tmp, RELOAD_PATH)
        log("new server code received: restarting to load it")
        self.send(202, {"reloading": True})
        threading.Timer(0.5, os._exit, args=(RELOAD_EXIT,)).start()

    def do_POST(self):
        if not self.authorized():
            return
        if self.path.rstrip("/") == "/v1/reload":
            return self.reload()
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
