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
  KAMI_MESH_BACKEND    "trellis2" (default), "hunyuan" (a Hunyuan3D-2 repo) or "pixal3d"
                       (TencentARC/Pixal3D: TRELLIS.2 with pixel-aligned conditioning, posed views)
  KAMI_MESH_SUBFOLDER  Hunyuan3D-2: the repo folder of the shape model (-mv: named views)
  KAMI_PAINT_MODEL     Hunyuan3D-2: the repo of Hunyuan3D-Paint (texture)
  KAMI_VIEWS_MODEL     diffusers image-editing repo that turns the camera around a front
                       picture ("views" jobs): Qwen-Image-Edit-2511 or FLUX.2 [dev]
  KAMI_VIEWS_LORA      its camera LoRA, "org/name/file.safetensors"
  KAMI_VIEWS_DEFAULTS  JSON of default views parameters (steps, guidance, lora_scale...)

Endpoints:
  GET  /health                     200 once the pipelines are loaded (503 before)
  GET  /metrics                    Prometheus text: jobs running / queued / finished
  GET  /v1/info                    what this server can do, and the defaults
  POST /v1/jobs                    {"kind": "mesh"|"image"|"text-to-mesh"|"views", "image": <base64>,
                                    "prompt": str, "params": {...}} -> {"id": ...}; or, for
                                    several views of one object, "images": [{"image":
                                    <base64>, "view": "front"|"left"|"back"|"right"|null}]
                                   "views": a prompt (or a front "image") -> <name>-front.png and
                                    the views params["views"] names (left, back, right...)
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
elif MESH_BACKEND == "pixal3d":
    # Pixal3D's inference defaults (1536 cascade); fov: the camera of a single picture in
    # degrees (Pixal3D estimates it with MoGe-2 for a photo; 20 is its multi-view rig's,
    # and suits product shots and `kami views` pictures)
    # resolution None: 1536, or 1024 in its low-VRAM mode (as its inference scripts pick)
    MESH_DEFAULTS = {"texture_size": 2048, "decimation_target": 500000, "remesh": True,
                     "resolution": None, "fov": 20.0}
else:
    # multidiffusion: every view has its say at every step. From the same four pictures
    # of a gecko it kept the spots on its back round and apart and its eye whole, where
    # stochastic (one view a step) blurred them into each other; the texture came darker.
    MESH_DEFAULTS = {"texture_size": 2048, "decimation_target": 500000, "remesh": True,
                     "multiview_mode": "multidiffusion"}
MESH_DEFAULTS.update(json.loads(os.environ.get("KAMI_MESH_DEFAULTS") or "{}"))
NVDIFFRAST_LIMIT = 16777216     # faces nvdiffrast can rasterise (TRELLIS.2's example)
# Views of one object a mesh job may carry: Hunyuan3D-2mv knows these four (its
# conditioner embeds which one each image is); TRELLIS.2 takes any views, unnamed.
HUNYUAN_VIEWS = ("front", "left", "back", "right")
MAX_VIEWS = 8
MULTIVIEW_MODES = ("stochastic", "multidiffusion")

# Pixal3D: views are posed by name around its training rig (the camera at azimuth 0 sees
# the front; at 90 the object faces the picture's left, kami's and Hunyuan3D-2mv's
# "left"), 20 degrees of fov at 3.12 from the object. The front comes first: every other
# view is placed relative to it. Unnamed views take the free names in PIXAL_VIEW_ORDER.
PIXAL_AZIMUTHS = {"front": 0, "front-left": 45, "left": 90, "back-left": 135, "back": 180,
                  "back-right": 225, "right": 270, "front-right": 315}
PIXAL_VIEW_ORDER = ("front", "left", "back", "right", "front-left", "back-left",
                    "back-right", "front-right")
PIXAL_FOV = 0.3490658503988659            # radians: transforms.json of its example
# GPUs smaller than this run Pixal3D's low-VRAM mode (models on the CPU, a stage at a
# time, 1024). It was 40 until four posed views at 1536 ran out of memory on an L40S too
# (45 GiB, 40.6 of them allocated): only the 80 GB cards are known to hold it.
PIXAL_LOW_VRAM_BELOW_GIB = 60
PIXAL_DISTANCE = 3.1192049980163574
PIXAL3D_CODE = "/opt/Pixal3D"             # the -pixal3d image's clone, at a pinned commit
NAF_CODE = "/opt/NAF"                     # its feature upsampler, pinned, weights included
# the four image-conditioning extractors of inference.py / inference_mv.py
PIXAL_COND = {
    "ss": {"image_size": 512, "grid_resolution": 16},
    "shape_512": {"image_size": 512, "grid_resolution": 32, "use_naf_upsample": True,
                  "naf_target_size": 512},
    "shape_1024": {"image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True,
                   "naf_target_size": 512},
    "tex_1024": {"image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True,
                 "naf_target_size": 1024},
}

# "views" jobs: an image-editing pipeline with a camera LoRA turns the camera around a
# front picture. Both LoRAs kami knows (fal's for Qwen-Image-Edit-2511, lovis93's for
# FLUX.2 [dev]) take "<sks> <azimuth> view <elevation> <distance>".
VIEWS_MODEL = os.environ.get("KAMI_VIEWS_MODEL", "")
VIEWS_LORA = os.environ.get("KAMI_VIEWS_LORA", "")
VIEWS_FAMILY = "flux2" if "flux.2" in VIEWS_MODEL.lower() else "qwen-edit"
VIEWS_DEFAULTS = {"steps": 28 if VIEWS_FAMILY == "flux2" else 40, "guidance": 4.0,
                  "lora_scale": 0.9, "elevation": "eye-level shot", "distance": "medium shot",
                  "views": ["left", "back", "right"]}
VIEWS_DEFAULTS.update(json.loads(os.environ.get("KAMI_VIEWS_DEFAULTS") or "{}"))
# view name -> the LoRA's azimuth. kami names views as Hunyuan3D-2mv does, by where the
# camera goes: "left" = the camera moved to your left, the object facing the picture's
# left (its example left.png). The LoRAs name the object's side instead: their "left
# side view" shows the object facing right (seen on views-qwen). So left and right swap.
VIEW_ANGLES = {"front": "front view", "front-right": "front-left quarter view",
               "right": "left side view", "back-right": "back-left quarter view",
               "back": "back view", "back-left": "back-right quarter view",
               "left": "right side view", "front-left": "front-right quarter view"}
# the front picture of a concept: what a mesh model wants to see
FRONT_HINT = ("front view of the whole object, centred, full view, on a plain white "
              "background, soft even studio light")
# a default like the others: `front_hint` in views_defaults or in a job's params replaces
# it, "" leaves the prompt alone
VIEWS_DEFAULTS.setdefault("front_hint", FRONT_HINT)


def front_prompt(prompt, p):
    """The prompt of a concept's front picture: the concept, then how to show it."""
    hint = str(p.get("front_hint", FRONT_HINT) or "").strip()
    return f"{prompt}, {hint}" if hint else prompt


def log(msg):
    print(f"[kami-server] {msg}", flush=True)


# --- pipelines ---

def alias_dino_layers(dino, label="DINOv3"):
    """transformers 5 moved DINOv3's layers from `model.layer` to `model.model.layer`;
    TRELLIS.2 and Pixal3D walk `model.layer`. Alias them, outside the module tree."""
    if dino is None or "layer" in dino.__dict__ or "layer" in getattr(dino, "_modules", {}):
        return False
    inner = getattr(dino, "_modules", {}).get("model")
    layers = getattr(inner, "_modules", {}).get("layer") if inner is not None else None
    if layers is None:
        return False
    object.__setattr__(dino, "layer", layers)
    log(f"{label}: transformers 5 layout, layers aliased")
    return True


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
        self.views = None
        self.on_gpu = None
        self.ready = False
        self.error = None

    def load(self):
        import torch    # noqa: F401  (fail early, with a clear message, without torch)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        if MESH_MODEL and MESH_BACKEND == "pixal3d":
            started = time.monotonic()
            log(f"loading Pixal3D {MESH_MODEL} (single view and posed views)")
            self.mesh = PixalMesh()
            self.on_gpu = "mesh"
            log(f"mesh pipeline ready in {time.monotonic() - started:.0f} s")
        elif MESH_MODEL and MESH_BACKEND == "hunyuan":
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
        if VIEWS_MODEL:
            started = time.monotonic()
            log(f"loading views pipeline {VIEWS_MODEL}" + (f" + LoRA {VIEWS_LORA}" if VIEWS_LORA else ""))
            from diffusers import DiffusionPipeline
            # a pre-quantised repo (diffusers/FLUX.2-dev-bnb-4bit) brings its own
            # quantisation config: its 4-bit parts load straight onto the GPU
            self.views = DiffusionPipeline.from_pretrained(VIEWS_MODEL, torch_dtype=torch.bfloat16)
            if VIEWS_LORA:
                repo, weight = lora_parts(VIEWS_LORA)
                self.views.load_lora_weights(repo, weight_name=weight, adapter_name="views")
                log(f"views LoRA: {lora_layers(self.views)} layers carry it; adapters "
                    f"{getattr(self.views, 'get_list_adapters', lambda: '?')()}")
            if OFFLOAD:
                _to(self.views, "cpu")
            else:
                try:
                    self.views.to("cuda")
                except (ValueError, RuntimeError, NotImplementedError) as e:
                    # some diffusers versions refuse .to() on 4-bit (bitsandbytes) parts;
                    # their FLUX.2 4-bit example offloads instead
                    log(f"views pipeline: .to('cuda') refused ({e}): model CPU offload instead")
                    self.views.enable_model_cpu_offload()
                self.on_gpu = self.on_gpu or "views"
            log(f"views pipeline ready in {time.monotonic() - started:.0f} s")
        self.ready = True

    def use(self, which):
        """Have `which` ("mesh" / "image" / "views") on the GPU, moving the others out
        (OFFLOAD)."""
        if not OFFLOAD or self.on_gpu == which:
            return
        import torch
        stage("to_gpu")
        loaded = {"mesh": self.mesh, "image": self.image, "views": self.views}
        for name, pipe in loaded.items():
            if name != which and pipe is not None:
                _to(pipe, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        _to(loaded[which], "cuda")
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


def naf_upsampler(device):
    """NAF from the image's pinned clone: only its naf.py (src/model/__init__ imports a
    dozen other upsamplers and their dependencies), with the release weights baked in,
    instead of the torch.hub.load of valeoai/NAF that Pixal3D does (code fetched on boot)."""
    import importlib.util
    import torch
    if NAF_CODE not in sys.path:
        sys.path.insert(0, NAF_CODE)          # src/ is a namespace package: src.layers
    spec = importlib.util.spec_from_file_location("kami_naf", os.path.join(NAF_CODE, "src/model/naf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NAF().to(device)
    state = torch.load(os.path.join(NAF_CODE, "naf_release.pth"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


class PixalMesh:
    """Pixal3D: one pipeline for single pictures and one for posed views (separate
    weights in the same repo). Both load at boot onto the CPU; the one a job needs goes
    to the GPU, the other waits in RAM (each takes ~18 GB on the GPU)."""

    def __init__(self):
        import torch
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as proj
        proj.DinoV3ProjFeatureExtractor._load_naf = _load_naf
        proj.DinoV3ProjMultiViewFeatureExtractor._load_naf = _load_naf
        # Four posed views at 1536 ran out of memory on a 24 GB 4090: below 40 GB its own
        # low-VRAM mode keeps the models on the CPU and moves each stage's in and out.
        gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.low_vram = gib < PIXAL_LOW_VRAM_BELOW_GIB
        self.pipes = {kind: self._build(kind) for kind in ("single", "mv")}
        self.on_gpu = None
        if self.low_vram:
            log(f"Pixal3D: {gib:.0f} GiB GPU, low-VRAM mode (models on the CPU, a stage at a time)")
            for pipe in self.pipes.values():
                pipe.low_vram = True
                pipe._device = torch.device("cuda")
                for model in self._cond_models(pipe):
                    if getattr(model, "use_naf_upsample", False):
                        model._load_naf()          # on the CPU, as its low-VRAM mode does
        else:
            self.use("mv")

    def _build(self, kind):
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline, Pixal3DMVImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as proj
        if kind == "mv":
            pipe = Pixal3DMVImageTo3DPipeline.from_pretrained(MESH_MODEL, "pipeline_mv.json")
            extractor = proj.DinoV3ProjMultiViewFeatureExtractor
        else:
            pipe = Pixal3DImageTo3DPipeline.from_pretrained(MESH_MODEL, "pipeline.json")
            extractor = proj.DinoV3ProjFeatureExtractor
        dino = DINO_MODEL or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        for stage_name, config in PIXAL_COND.items():
            config = dict(config, model_name=dino)
            if kind == "mv":
                config["multiview_fusion"] = "average"
            model = extractor(**config).eval()
            alias_dino_layers(getattr(model, "model", None), f"Pixal3D {kind} {stage_name}")
            setattr(pipe, f"image_cond_model_{stage_name}", model)
        pipe.low_vram = False
        track_mesh_progress(pipe)
        return pipe

    def _cond_models(self, pipe):
        return [getattr(pipe, f"image_cond_model_{name}") for name in PIXAL_COND]

    def use(self, kind):
        """The pipeline for `kind` ("single" / "mv"), on the GPU."""
        import torch
        if self.low_vram:
            return self.pipes[kind]             # it moves its own stages
        if self.on_gpu != kind:
            if self.on_gpu is not None:
                self._move(self.pipes[self.on_gpu], "cpu")
                gc.collect()
                torch.cuda.empty_cache()
            self._move(self.pipes[kind], "cuda")
            for model in self._cond_models(self.pipes[kind]):
                if getattr(model, "use_naf_upsample", False):
                    model._load_naf()
            self.on_gpu = kind
        return self.pipes[kind]

    def _move(self, pipe, device):
        pipe.to(device)
        for model in self._cond_models(pipe):
            model.to(device)

    def to(self, device):                     # OFFLOAD: the whole of it out of the way
        if self.on_gpu is not None and device == "cpu":
            self._move(self.pipes[self.on_gpu], "cpu")
            self.on_gpu = None
        elif device == "cuda":
            self.use("mv")


def _load_naf(self):
    """Pixal3D's extractors call this to get their NAF upsampler (see naf_upsampler)."""
    if self.naf_model is None:
        self.naf_model = naf_upsampler(next(self.model.parameters()).device)


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


def lora_parts(entry):
    """("org/name", "file.safetensors") of a LoRA entry "org/name/file.safetensors"."""
    parts = entry.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def lora_layers(pipe):
    """How many layers of the pipeline's denoiser got LoRA weights (0: the LoRA's keys
    matched nothing, and it does nothing)."""
    model = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if model is None:
        return 0
    return sum(1 for _, module in model.named_modules() if hasattr(module, "lora_A"))


def on_white(image):
    """An RGB picture: a cut-out one on white (the editing models want a plain backdrop)."""
    from PIL import Image
    if image.mode != "RGBA":
        return image.convert("RGB")
    backdrop = Image.new("RGB", image.size, (255, 255, 255))
    backdrop.paste(image, mask=image.getchannel("A"))
    return backdrop


def editing_size(image, longest=1024):
    """The picture at most `longest` px on its long side, both sides a multiple of 16."""
    from PIL import Image
    scale = min(1.0, longest / max(image.size))
    w, h = (max(16, int(v * scale) // 16 * 16) for v in image.size)
    return image if (w, h) == image.size else image.resize((w, h), Image.Resampling.LANCZOS)


def view_prompt(name, p):
    """The camera LoRA's prompt for view `name`: '<sks> back view eye-level shot medium shot'."""
    return f"<sks> {VIEW_ANGLES[name]} {p['elevation']} {p['distance']}"


def _lora(enabled, scale=1.0):
    """Turn the views pipeline's camera LoRA on (at `scale`) or off."""
    pipe = PIPES.views
    if not VIEWS_LORA:
        return
    if enabled:
        pipe.enable_lora()
        pipe.set_adapters(["views"], adapter_weights=[float(scale)])
    else:
        pipe.disable_lora()


def views_call(prompt, p, image=None, width=1024, height=1024):
    """One picture from the views pipeline: an edit of `image`, or (FLUX.2) from text."""
    import torch
    steps = int(p["steps"])

    def on_step(pipe, i, t, tensors):
        step(i + 1, steps)
        return tensors
    kwargs = {"prompt": prompt, "num_inference_steps": steps, "callback_on_step_end": on_step}
    if p.get("seed") is not None:
        kwargs["generator"] = torch.Generator("cuda").manual_seed(int(p["seed"]))
    if image is not None:
        kwargs["image"] = [image]
    if VIEWS_FAMILY == "flux2":
        kwargs.update(guidance_scale=float(p["guidance"]), width=width, height=height)
    else:
        # Qwen-Image-Edit: true CFG with an empty negative prompt, as in its model card
        kwargs.update(true_cfg_scale=float(p["guidance"]), negative_prompt=" ", guidance_scale=1.0)
    pipe = PIPES.views
    return pipe(**_accepted(pipe.__call__, kwargs)).images[0]


def make_views(job):
    """A front picture (given, or made from the prompt) and the camera turned around it:
    <name>-front.png, <name>-left.png... for `kami mesh front=... left=...`."""
    p = {**VIEWS_DEFAULTS, **job.params}
    names = [n for n in p["views"] if n != "front"]

    def save(image, view):
        path = os.path.join(job.dir, f"{job.name}-{view}.png")
        image.save(path)
        job.files.append(os.path.basename(path))
    if job.image_bytes:
        front = open_image(job.image_bytes[0][1])
    elif PIPES.image is not None:
        image_params = dict(job.params.get("image") or {})
        if p.get("seed") is not None:
            image_params.setdefault("seed", p["seed"])
        front = make_image(front_prompt(job.prompt, p), image_params,
                           os.path.join(job.dir, "front-raw.png"))
    else:                                    # FLUX.2: the same model makes the front picture
        PIPES.use("views")
        stage("image")
        _lora(False)
        front = views_call(front_prompt(job.prompt, p), p)
    front = editing_size(on_white(front))
    save(front, "front")
    PIPES.use("views")
    stage("views", loops=len(names))
    _lora(True, p["lora_scale"])
    for name in names:
        save(views_call(view_prompt(name, p), p, front, *front.size), name)


def make_mesh(views, params, out_path):
    """`views`: [(view name or None, PIL image)], one or several views of one object."""
    p = {**MESH_DEFAULTS, **params}
    PIPES.use("mesh")
    if MESH_BACKEND == "hunyuan":
        return make_mesh_hunyuan(views, p, out_path)
    if MESH_BACKEND == "pixal3d":
        return make_mesh_pixal3d(views, p, out_path)
    return make_mesh_trellis(views, p, out_path)


@contextlib.contextmanager
def trellis_multiview(pipe, views, mode):
    """TRELLIS.2 conditioned on several views of one object -- the tuning-free trick of
    TRELLIS (1)'s run_multi_image, which TRELLIS.2 does not ship: the image features of
    every view go through the samplers, and at each step the flow model sees one view
    in turn ("stochastic") or all of them, averaged ("multidiffusion", the default; N
    model calls a step).
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


def pixal_camera(azimuth_degrees):
    """Pixal3D's camera-to-world matrix (Z-up, the camera looking down its -Z) for a view
    at that azimuth around the object: its front camera turned about the world's Z."""
    import math
    a = math.radians(azimuth_degrees)
    c, s = math.cos(a), math.sin(a)
    front = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -PIXAL_DISTANCE], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
    turn = [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return [[sum(turn[i][k] * front[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def same_framing(images, margin=1.1):
    """RGBA views cropped to squares of one size, each centred on its object: Pixal3D's
    posed views are used as framed (no crop of its own), and its training renders show
    the object at one scale in every view. The square is the largest object extent of
    all the views times `margin`, as its single-view preprocessing frames a picture."""
    from PIL import Image
    boxes = []
    for image in images:
        mask = image.getchannel("A").point(lambda a: 255 if a > 204 else 0)
        boxes.append(mask.getbbox() or (0, 0, image.width, image.height))
    side = int(max(max(r - l, b - t) for l, t, r, b in boxes) * margin) or 1
    out = []
    for image, (l, t, r, b) in zip(images, boxes):
        cx, cy = (l + r) // 2, (t + b) // 2
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(image, (side // 2 - cx, side // 2 - cy))
        out.append(square)
    return out


def pixal_cutout(pipe, image):
    """RGBA with the object cut out (an image already cut out is kept as it is)."""
    if image.mode == "RGBA":
        return image
    if getattr(pipe, "low_vram", False):
        pipe.rembg_model.to(pipe.device)        # low-VRAM: it waits on the CPU
    try:
        return pipe.rembg_model(image.convert("RGB")).convert("RGBA")
    finally:
        if getattr(pipe, "low_vram", False):
            pipe.rembg_model.cpu()


def pixal_views(pipe, views):
    """The bundle run_mv takes, out of [(name, PIL)]: front first, then the others."""
    import numpy as np
    import torch
    from PIL import Image
    order = sorted(views, key=lambda v: PIXAL_VIEW_ORDER.index(v[0]))
    images = same_framing([pixal_cutout(pipe, image) for _, image in order])

    def cond(image, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        alpha = torch.tensor(np.array(image.getchannel("A"))).float() / 255.0
        rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        return rgb * alpha.unsqueeze(0)          # premultiplied: black background
    count = len(images)
    return {
        "images": {size: torch.stack([cond(im, size) for im in images])[None] for size in (512, 1024)},
        "camera_angle_x": torch.full((1, count), PIXAL_FOV),
        "camera_distance": torch.full((1, count), PIXAL_DISTANCE),
        "transform_matrix": torch.tensor([[pixal_camera(PIXAL_AZIMUTHS[n]) for n, _ in order]]),
        "mesh_scale": 1.0,
        "view_names": [n for n, _ in order],
    }


def pixal_distance(fov, image_resolution=512, mesh_scale=1.0):
    """Pixal3D's distance_from_fov (inference.py) for its default framing: the camera
    distance at which the object fills a picture seen with that horizontal fov."""
    import math
    gp_x, gp_y = -1.0 / mesh_scale / 2, 0.0       # grid point (-1, 0, 0), turned and halved
    f_pixels = (16.0 / math.tan(fov / 2.0)) * image_resolution / 32.0
    x_ndc = 0 - image_resolution / 2.0
    return f_pixels * gp_x / x_ndc - gp_y


def make_mesh_pixal3d(views, p, out_path):
    import math
    import numpy as np
    import o_voxel
    seed = int(p["seed"]) if p.get("seed") is not None else 42
    resolution = int(p["resolution"] or (1024 if PIPES.mesh.low_vram else 1536))
    kwargs = {"seed": seed, "pipeline_type": f"{resolution}_cascade"}
    if len(views) == 1:
        pipe = PIPES.mesh.use("single")
        fov = math.radians(float(p["fov"]))
        mesh = pipe.run(views[0][1], camera_params={"camera_angle_x": fov, "distance": pixal_distance(fov),
                                                    "mesh_scale": 1.0}, **kwargs)[0]
    else:
        pipe = PIPES.mesh.use("mv")
        stage("prepare")
        bundle = pixal_views(pipe, views)
        log(f"{len(views)} posed views: {', '.join(bundle['view_names'])}")
        mesh = pipe.run_mv(bundle, **kwargs)[0]
    if hasattr(mesh, "simplify"):
        mesh.simplify(NVDIFFRAST_LIMIT)
    stage("glb")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(p["decimation_target"]), texture_size=int(p["texture_size"]),
        remesh=bool(p["remesh"]), remesh_band=1, remesh_project=0, verbose=False)
    # Pixal3D's frame (front camera on -Y) to the GLB's, as its inference scripts do
    glb.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                                 dtype=np.float64))
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
    "views": "Turning the camera around the object",
}
MESH_PLAN = (("prepare", .04), ("condition", .04), ("structure", .12), ("shape", .34),
             ("texture", .22), ("decode", .08), ("glb", .16))
# Hunyuan3D-2: shape diffusion, surface extraction (+ clean-up), then Hunyuan3D-Paint
HUNYUAN_PLAN = (("prepare", .03), ("shape", .40), ("decode", .12), ("texture", .40), ("glb", .05))
PLANS = {
    "image": (("image", 1.0),),
    "mesh": MESH_PLAN,
    "text-to-mesh": (("image", .2),) + tuple((k, w * .8) for k, w in MESH_PLAN),
    "views": (("image", .2), ("views", .8)),
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
                   if n.startswith(("trellis2.pipelines.samplers", "pixal3d.pipelines.samplers")) and m]:
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
    if job.kind == "views":
        return make_views(job)
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


KINDS = ("mesh", "image", "text-to-mesh", "views")


def view_limits():
    """(how many views of one object a mesh job may carry, their names or None for any)."""
    if MESH_BACKEND == "hunyuan":
        return (len(HUNYUAN_VIEWS), HUNYUAN_VIEWS) if HUNYUAN_MV else (1, None)
    if MESH_BACKEND == "pixal3d":
        return len(PIXAL_VIEW_ORDER), PIXAL_VIEW_ORDER
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
        if MESH_BACKEND == "pixal3d" and len(views) > 1 and "front" not in [v for v, _ in views]:
            raise ValueError("Pixal3D places every view relative to the front one: name one front")
    return [tuple(v) for v in views]


def check_views_params(params, has_front):
    """Raise ValueError if a views job's params are wrong."""
    names = params.get("views", VIEWS_DEFAULTS["views"])
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n in VIEW_ANGLES for n in names)):
        raise ValueError(f"views must be a list of: {', '.join(VIEW_ANGLES)}")
    if len(set(names)) != len(names):
        raise ValueError("each view may appear once")
    for key in ("elevation", "distance", "front_hint"):
        if key in params and not isinstance(params[key], str):
            raise ValueError(f"{key} must be text, like {VIEWS_DEFAULTS[key]!r}")
    if not has_front and PIPES.image is None and VIEWS_FAMILY != "flux2":
        raise ValueError(f"{VIEWS_MODEL} edits pictures: give a front image, or use a "
                         "profile with an image_model to make one from the prompt")


def submit(body):
    """A new job from a request body, or raise ValueError with what is wrong with it."""
    kind = body.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind in ("mesh", "text-to-mesh") and PIPES.mesh is None:
        raise ValueError("this pod serves no mesh model (profile without mesh_model)")
    if kind in ("image", "text-to-mesh") and PIPES.image is None:
        raise ValueError("this pod serves no image model (profile without image_model)")
    if kind == "views" and PIPES.views is None:
        raise ValueError("this pod serves no views model (profile without views_model)")
    prompt = body.get("prompt") or ""
    if kind == "mesh":
        image_bytes = mesh_views(body)
    elif kind == "views" and body.get("image"):
        image_bytes = mesh_views({"image": body["image"]})[:1]
    else:
        image_bytes = None
    if kind != "mesh" and not image_bytes and not prompt.strip():
        raise ValueError(f"a {kind} job needs a prompt" + (" or a front image" if kind == "views" else ""))
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if kind == "views":
        check_views_params(params, bool(image_bytes))
    mode = (params.get("mesh", params) if isinstance(params.get("mesh", params), dict) else {}).get(
        "multiview_mode")
    if mode is not None and mode not in MULTIVIEW_MODES:
        raise ValueError(f"multiview_mode must be one of {', '.join(MULTIVIEW_MODES)}")
    name = "".join(c for c in str(body.get("name") or "") if c.isalnum() or c in "-_")[:60]
    job = Job(kind, prompt, image_bytes, params, name)
    if kind == "views" and image_bytes:
        job.plan = (("views", 1.0),)            # no picture to make first
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
                                   if MESH_BACKEND == "trellis2" else None,
                                   "views_model": VIEWS_MODEL or None,
                                   "views_lora": VIEWS_LORA or None,
                                   "views_defaults": VIEWS_DEFAULTS if VIEWS_MODEL else None,
                                   "view_angles": list(VIEW_ANGLES) if VIEWS_MODEL else None,
                                   "views_from_prompt": bool(VIEWS_MODEL) and (
                                       bool(IMAGE_MODEL) or VIEWS_FAMILY == "flux2")})
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


# Packages the pipelines need newer than an older kami-server image has (the image pins
# them now): upgraded on boot, before anything imports them. Qwen-Image-Edit's text
# encoder renders its chat template with transformers, which wants jinja2 >= 3.1.6.
MIN_PACKAGES = {"jinja2": (3, 1, 6)}


def _version(text):
    return tuple(int(part) for part in (text.split(".") + ["0", "0"])[:3] if part.isdigit())


def ensure_packages():
    """pip-upgrade what is older than MIN_PACKAGES; a failure is logged, not fatal."""
    from importlib import metadata
    import subprocess
    for name, least in MIN_PACKAGES.items():
        try:
            have = _version(metadata.version(name))
        except metadata.PackageNotFoundError:
            have = ()
        if have >= least:
            continue
        wanted = f"{name}>={'.'.join(map(str, least))}"
        log(f"{name} {'.'.join(map(str, have)) or 'missing'} is too old: installing {wanted}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", wanted],
                           check=True, timeout=300)
        except Exception as e:
            log(f"could not install {wanted}: {e}")


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)
    if not API_KEY:
        log("KAMI_API_KEY is not set: refusing to serve without authentication")
        sys.exit(2)
    if not MESH_MODEL and not IMAGE_MODEL and not VIEWS_MODEL:
        log("none of KAMI_MESH_MODEL, KAMI_IMAGE_MODEL, KAMI_VIEWS_MODEL is set: nothing to serve")
        sys.exit(2)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"listening on :{PORT} (health 503 until the pipelines load)")
    try:
        ensure_packages()
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
