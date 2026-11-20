"""Configuration (kami.toml), secrets (.env) and local state (.kami/state.json).

The project is the directory holding kami.toml. It is found from, in order: the
--config flag, the KAMI_CONFIG variable, the current directory and its parents, and
finally the source checkout kami runs from. `.env` and `.kami/` live next to it, so
each project has its own secrets and its own pod.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "kami.toml"
# The checkout kami is installed from (editable install); a fallback, not a default.
SOURCE_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    pass


# Written by `kami init` when the project has no kami.toml yet.
STARTER_CONFIG = """\
# kami configuration. Secrets (RUNPOD_API_KEY, VAST_API_KEY, HF_TOKEN) live in .env, not here.

[defaults]
profile = "mesh"
# Where pods run: "runpod" or "vast" (vast.ai). `kami up --provider vast` overrides it.
provider = "runpod"
# The kami-server container image: build it from docker/Dockerfile and push it
# (see the README), then put its name here.
container = "ghcr.io/CHANGE-ME/kami-server:0.1.0"
# "ephemeral": weights downloaded on every `kami up` (no fixed cost).
# "volume": persistent network volume (fast boots, ~$0.07/GB/month, pinned datacenter).
storage = "ephemeral"
datacenter = ""
volume_name = "kami-models"
volume_size_gb = 100
cloud = "SECURE"
port = 8000
# Where `kami mesh` / `kami image` save what they make (relative to the current directory).
output_dir = "kami-out"
# Prompts get a hint added so the picture is of something to print: standing on a flat
# base, one thick piece, {colors} flat colours, matte, evenly lit. Write your own here
# ({colors} becomes the number of filaments), or "" for prompts exactly as written.
# print_hint = "..."
# The pod deletes itself after max_hours of uptime, or after idle_minutes without
# jobs, even if your PC is off. 0 disables each one.
max_hours = 4
idle_minutes = 30

# A profile = models + GPUs (tried in order until one has stock).
# mesh_model: a TRELLIS.2 pipeline (image -> textured GLB).
# image_model: any diffusers text-to-image pipeline.
[profiles.mesh]
mesh_model = "microsoft/TRELLIS.2-4B"
gpus = ["RTX 4090", "L40S", "RTX A6000"]
min_vram_gb = 24
min_ram_gb = 48          # host RAM per GPU: TRELLIS.2 keeps its models in RAM
"""

STARTER_ENV = """\
# RunPod API key: https://console.runpod.io/user/settings
RUNPOD_API_KEY=
# vast.ai API key (provider = "vast"): https://cloud.vast.ai/manage-keys/
VAST_API_KEY=
# Hugging Face token: TRELLIS.2 needs gated models (DINOv3, RMBG-2.0); accept their terms first
HF_TOKEN=
"""

# The kami-server image before the README's build step: `up` and `doctor` stop on it.
PLACEHOLDER = "CHANGE-ME"


class Project:
    """Paths of the active project; set once by `use_project()`."""

    def __init__(self, config: Path):
        self.config = config.resolve()
        self.root = self.config.parent
        self.env = self.root / ".env"
        self.state_dir = self.root / ".kami"
        self.state = self.state_dir / "state.json"


def find_config(explicit: str | None = None, start: Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    if os.environ.get("KAMI_CONFIG"):
        return Path(os.environ["KAMI_CONFIG"])
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).is_file():
            return folder / CONFIG_NAME
    if (SOURCE_ROOT / CONFIG_NAME).is_file():
        return SOURCE_ROOT / CONFIG_NAME
    return None


def use_project(explicit: str | None = None) -> Project:
    """Select the project (see the module docstring). Without a kami.toml anywhere,
    the current directory is used, so `kami init` can create one there."""
    global project
    found = find_config(explicit)
    project = Project(found if found else Path.cwd() / CONFIG_NAME)
    return project


project = Project(find_config() or Path.cwd() / CONFIG_NAME)


@dataclass
class Profile:
    name: str
    gpus: list[str]
    # Hugging Face repos of the pipelines the pod loads; at least one of the two.
    mesh_model: str = ""             # TRELLIS.2 (image -> GLB)
    image_model: str = ""            # diffusers text-to-image
    # Views of one object (`kami views`): a diffusers image-editing pipeline that turns the
    # camera around a front picture (Qwen-Image-Edit-2511, FLUX.2 [dev]), with a camera
    # LoRA ("org/name/file.safetensors"). The front picture of a concept comes from
    # image_model, or from views_model itself when it also makes images from text (FLUX.2).
    views_model: str = ""
    views_lora: str = ""
    # Other repos the pipelines download on load (gated ones must be accepted first);
    # TRELLIS.2's are added by default, see MESH_DEPENDENCIES.
    extra_models: list[str] = field(default_factory=list)
    # Replace the sub-models TRELLIS.2's pipeline.json names (a Hugging Face repo, in the
    # transformers format): the image conditioning model (DINOv3) and the background
    # remover. Empty = the pipeline's own (DINO_MODEL, REMBG_MODEL).
    dino_model: str = ""
    rembg_model: str = ""
    # Hunyuan3D-2 (a mesh_model of tencent/Hunyuan3D-2*): the folder of the repo with the
    # shape model (empty: the repo's usual one, see HUNYUAN_SUBFOLDERS), the repo of the
    # texture model (Hunyuan3D-Paint), and the acknowledgement of its licence, which
    # does not apply in the EU, the UK and South Korea (see README).
    mesh_subfolder: str = ""
    paint_model: str = ""
    accept_hunyuan_license: bool = False
    # Defaults for the jobs (the server's own when empty): see server.py.
    image_defaults: dict = field(default_factory=dict)
    mesh_defaults: dict = field(default_factory=dict)
    views_defaults: dict = field(default_factory=dict)
    # Keep the pipeline not in use on the CPU: both fit a smaller GPU, a switch costs seconds.
    offload: bool = False
    # Container disk in GB. In ephemeral mode every model must fit here.
    disk_gb: int = 80
    gpu_count: int = 1
    # Smallest GPU memory the profile needs: GPUs below it are skipped (0 = no check).
    min_vram_gb: int = 0
    # Smallest host RAM per GPU (GB): only machines with at least this (0 = any). TRELLIS.2
    # keeps its models in RAM between steps, so a small host runs out of memory.
    min_ram_gb: int = 0
    # Container image override (default: [defaults] container).
    container: str | None = None
    # Lowest host CUDA version ("12.4"): the pod is only placed on machines with at least this.
    min_cuda: str | None = None

    @property
    def model(self) -> str:
        """The models, for messages: 'microsoft/TRELLIS.2-4B + Tongyi-MAI/Z-Image-Turbo'."""
        return " + ".join(m for m in (self.mesh_model, self.image_model, self.views_model) if m)

    @property
    def mesh_backend(self) -> str:
        return mesh_backend(self.mesh_model)

    @property
    def shape_subfolder(self) -> str:
        """Hunyuan3D-2: the repo folder with the shape model."""
        return self.mesh_subfolder or HUNYUAN_SUBFOLDERS.get(self.mesh_model.lower(), HUNYUAN_SUBFOLDER)

    @property
    def repos(self) -> list[str]:
        """Every Hugging Face repo the pod downloads on boot. "org/name/folder" is the
        part of a repo under that folder (Hunyuan3D-2 downloads only its folders)."""
        image = [self.image_model] if self.image_model else []
        image += [m for m in (self.views_model, self.views_lora) if m]
        if self.mesh_backend == "hunyuan":
            paint = self.paint_model or HUNYUAN_PAINT_MODEL
            repos = [f"{self.mesh_model}/{self.shape_subfolder}", *image,
                     *(f"{paint}/{folder}" for folder in HUNYUAN_PAINT_FOLDERS),
                     self.rembg_model or REMBG_MODEL]
        elif self.mesh_backend == "pixal3d":
            # its own repo holds every stage (single-view and posed-view weights); DINOv3
            # and RMBG-2.0 as TRELLIS.2's (no TRELLIS-image-large)
            repos = [self.mesh_model, *image, self.dino_model or DINO_MODEL,
                     self.rembg_model or REMBG_MODEL]
        elif self.mesh_model:
            repos = [self.mesh_model, *image, self.dino_model or DINO_MODEL,
                     self.rembg_model or REMBG_MODEL, *MESH_DEPENDENCIES]
        else:
            repos = image
        return list(dict.fromkeys([*repos, *self.extra_models]))


# Repos the TRELLIS.2 pipeline loads besides its own (its pipeline.json): the image
# conditioning model and the background remover are gated on Hugging Face; a profile can
# replace them (dino_model, rembg_model).
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_MODEL = "briaai/RMBG-2.0"
MESH_DEPENDENCIES = ("microsoft/TRELLIS-image-large",)

# Hunyuan3D-2 (tencent/Hunyuan3D-2, -2mv, -2mini): one repo holds several models, each in
# its folder. The texture model (Hunyuan3D-Paint) and its delighter come from
# tencent/Hunyuan3D-2 whatever the shape model; the background goes with RMBG-2.0 (the
# BiRefNet TRELLIS.2 uses).
HUNYUAN_SUBFOLDER = "hunyuan3d-dit-v2-0"
HUNYUAN_SUBFOLDERS = {"tencent/hunyuan3d-2mv": "hunyuan3d-dit-v2-mv",
                      "tencent/hunyuan3d-2mini": "hunyuan3d-dit-v2-mini"}
HUNYUAN_PAINT_MODEL = "tencent/Hunyuan3D-2"
HUNYUAN_PAINT_FOLDERS = ("hunyuan3d-paint-v2-0-turbo", "hunyuan3d-delight-v2-0")
HUNYUAN_LICENSE = ("Hunyuan3D-2's licence (Tencent Hunyuan Community) does not apply in the "
                   "European Union, the United Kingdom and South Korea: read it at "
                   "https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE")


def lora_repo(lora: str) -> str:
    """The repo of a LoRA entry: "org/name" out of "org/name/file.safetensors"."""
    return "/".join(lora.split("/")[:2]) if lora else ""


def views_family(views_model: str) -> str:
    """How a views_model is called: "flux2" (FLUX.2, also text -> image) or "qwen-edit"."""
    return "flux2" if "flux.2" in views_model.lower() else "qwen-edit"


def mesh_backend(mesh_model: str) -> str:
    """Which pipeline a mesh_model runs on: "hunyuan" (Hunyuan3D-2), "pixal3d"
    (TencentARC/Pixal3D) or "trellis2"."""
    name = mesh_model.lower()
    if "hunyuan3d" in name:
        return "hunyuan"
    return "pixal3d" if "pixal3d" in name else "trellis2"


@dataclass
class Config:
    container: str
    storage: str          # "ephemeral" (download on each boot) or "volume" (network volume)
    datacenter: str
    volume_name: str
    volume_size_gb: int
    cloud: str
    port: int
    output_dir: str
    # In-pod guard (0 disables each): the pod deletes itself after this many hours
    # of uptime, or once the server has run no job for this many minutes.
    max_hours: float
    idle_minutes: float
    default_profile: str
    profiles: dict[str, Profile]
    # Where pods run: "runpod" or "vast" (vast.ai).
    provider: str = "runpod"
    # Added to every prompt so the picture is of something printable (jobs.PRINT_HINT
    # when None; "" for none).
    print_hint: str | None = None

    def profile(self, name: str | None) -> Profile:
        name = name or self.default_profile
        if name not in self.profiles:
            raise ConfigError(f"Unknown profile '{name}'\n"
                              f"Available: {', '.join(self.profiles)} (`kami profiles`)")
        return self.profiles[name]


def resolve_profile(name: str, raw: dict[str, dict], seen: tuple[str, ...] = ()) -> dict:
    """The profile's table with `extends` applied: keys of the base profile, then its own."""
    if name in seen:
        raise ConfigError(f"Profiles extend each other in a loop: {' -> '.join((*seen, name))}")
    own = dict(raw[name])
    base_name = own.pop("extends", None)
    if base_name is None:
        return own
    if base_name not in raw:
        raise ConfigError(f"Profile '{name}' extends unknown profile '{base_name}'")
    return {**resolve_profile(base_name, raw, (*seen, name)), **own}


# What kami.toml may hold, and the kind of value each key takes. Anything else is
# reported (a typo, or a key from another kami version) instead of silently ignored.
TEXT, INT, NUMBER, NAMES, TABLE, BOOL = ("text", "an integer", "a number", "a list of names",
                                         "a table", "true or false")
VERSION = "a version like \"12.4\""

DEFAULT_KEYS = {
    "profile": TEXT, "provider": TEXT, "container": TEXT, "storage": TEXT, "datacenter": TEXT,
    "volume_name": TEXT, "volume_size_gb": INT, "cloud": TEXT, "port": INT,
    "output_dir": TEXT, "max_hours": NUMBER, "idle_minutes": NUMBER, "print_hint": TEXT,
}

PROFILE_KEYS = {
    "extends": TEXT, "mesh_model": TEXT, "image_model": TEXT, "extra_models": NAMES,
    "views_model": TEXT, "views_lora": TEXT, "views_defaults": TABLE,
    "dino_model": TEXT, "rembg_model": TEXT, "mesh_subfolder": TEXT, "paint_model": TEXT,
    "accept_hunyuan_license": BOOL,
    "image_defaults": TABLE, "mesh_defaults": TABLE, "offload": BOOL, "gpus": NAMES,
    "disk_gb": INT, "gpu_count": INT, "min_vram_gb": INT, "min_ram_gb": INT, "container": TEXT,
    "min_cuda": VERSION,
}

# Allowed values, and numeric ranges (low, high; None = unbounded).
CHOICES = {"provider": ("runpod", "vast"), "storage": ("ephemeral", "volume"),
           "cloud": ("SECURE", "COMMUNITY")}
RANGES = {"port": (1, 65535), "volume_size_gb": (1, None), "disk_gb": (1, None),
          "gpu_count": (1, 8), "max_hours": (0, None), "idle_minutes": (0, None),
          "min_vram_gb": (0, None), "min_ram_gb": (0, None)}

# Job parameters a profile's defaults may set (see server.py).
IMAGE_PARAMS = ("steps", "guidance", "width", "height", "negative", "seed")
# views: steps / guidance of the editing pipeline, the LoRA's weight, and the camera's
# elevation and distance in the LoRA's words ("eye-level shot", "medium shot")
VIEWS_PARAMS = ("steps", "guidance", "seed", "lora_scale", "elevation", "distance")
# Both backends: TRELLIS.2 (texture_size, remesh, pipeline_type, multiview_mode) and
# Hunyuan3D-2 (steps, octree_resolution, guidance, texture); each ignores the others.
MESH_PARAMS = ("texture_size", "decimation_target", "remesh", "seed", "pipeline_type",
               "multiview_mode", "steps", "octree_resolution", "guidance", "texture",
               "resolution", "fov")


def _kind_ok(value, kind: str) -> bool:
    if kind == TEXT:
        return isinstance(value, str)
    if kind == VERSION:
        return isinstance(value, (str, float))       # "12.8" or 12.8
    if kind == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == NAMES:
        return isinstance(value, list) and all(isinstance(a, str) and a.strip() for a in value)
    if kind == TABLE:
        return isinstance(value, dict)
    if kind == BOOL:
        return isinstance(value, bool)
    return True


def _unknown(key: str, where: str, known: dict, other: dict, other_where: str) -> str:
    msg = f"{where}: unknown key '{key}'"
    if key in other:
        return f"{msg} (it belongs in {other_where})"
    close = difflib.get_close_matches(key, known, n=1)
    return f"{msg} (did you mean '{close[0]}'?)" if close else msg


def _check_table(table: dict, where: str, known: dict, other: dict, other_where: str) -> list[str]:
    problems = []
    for key, value in table.items():
        if key not in known:
            problems.append(_unknown(key, where, known, other, other_where))
            continue
        kind = known[key]
        if not _kind_ok(value, kind):
            problems.append(f"{where}: {key} must be {kind}, not {json.dumps(value)}")
            continue
        if key in CHOICES and value not in CHOICES[key]:
            options = " or ".join(f"'{c}'" for c in CHOICES[key])
            problems.append(f"{where}: {key} must be {options}, not '{value}'")
        if key in RANGES:
            low, high = RANGES[key]
            if value < low or (high is not None and value > high):
                bound = f"between {low} and {high}" if high is not None else f"at least {low}"
                problems.append(f"{where}: {key} must be {bound}, not {value}")
        if key == "min_cuda" and not re.fullmatch(r"\d+\.\d+", str(value)):
            problems.append(f"{where}: min_cuda must look like \"12.4\" (major.minor), not '{value}'")
        if kind == TABLE:
            known_params = {"image_defaults": IMAGE_PARAMS,
                            "views_defaults": VIEWS_PARAMS}.get(key, MESH_PARAMS)
            for param in value:
                if param not in known_params:
                    problems.append(f"{where}: {key} has unknown parameter '{param}' "
                                    f"(known: {', '.join(known_params)})")
    return problems


def validate(raw: dict) -> list[str]:
    """Every problem in a parsed kami.toml (empty when it is sound), checked before
    anything is built from it, so a typo is reported up front, not as a failed pod."""
    problems = []
    for key in raw:
        if key not in ("defaults", "profiles"):
            close = difflib.get_close_matches(key, ("defaults", "profiles"), n=1)
            hint = f" (did you mean [{close[0]}]?)" if close else " (only [defaults] and [profiles.*])"
            problems.append(f"unknown table [{key}]{hint}")
    d = raw.get("defaults", {})
    if not isinstance(d, dict):
        return problems + ["defaults must be a table: [defaults]"]
    problems += _check_table(d, "[defaults]", DEFAULT_KEYS, PROFILE_KEYS, "a [profiles.<name>] table")
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict) or not all(isinstance(p, dict) for p in profiles.values()):
        return problems + ["profiles must be tables: [profiles.<name>]"]
    if not profiles:
        problems.append("no profiles: add at least one [profiles.<name>] with a model and gpus")
    for name, table in profiles.items():
        where = f"[profiles.{name}]"
        problems += _check_table(table, where, PROFILE_KEYS, DEFAULT_KEYS, "[defaults]")
        base = table.get("extends")
        if isinstance(base, str) and base not in profiles:
            problems.append(f"{where}: extends unknown profile '{base}'")
    if problems:
        return problems       # the rest needs well-formed tables
    # what only the resolved profiles (extends applied) can tell
    for name in profiles:
        where = f"[profiles.{name}]"
        try:
            p = resolve_profile(name, profiles)
        except ConfigError as e:
            problems.append(str(e))
            continue
        if not p.get("gpus"):
            problems.append(f"{where}: needs gpus")
        if not any(p.get(k) for k in ("mesh_model", "image_model", "views_model")):
            problems.append(f"{where}: needs mesh_model, image_model or views_model")
        if p.get("views_lora") and not p.get("views_model"):
            problems.append(f"{where}: views_lora needs a views_model")
        if p.get("views_lora") and p["views_lora"].count("/") < 2:
            problems.append(f"{where}: views_lora must be \"org/name/file.safetensors\", "
                            f"not '{p['views_lora']}'")
        if mesh_backend(p.get("mesh_model", "")) == "hunyuan" and not p.get("accept_hunyuan_license"):
            problems.append(f"{where}: {HUNYUAN_LICENSE}. If it applies to you, set "
                            "accept_hunyuan_license = true")
        if d.get("provider") == "vast":
            from . import catalog      # catalog imports this module
            for gpu in p.get("gpus", []):
                try:
                    catalog.provider_gpu(gpu, "vast")
                except ConfigError as e:
                    problems.append(f"{where}: {str(e).splitlines()[0]}")
    default = d.get("profile")
    if default is not None and default not in profiles:
        problems.append(f"[defaults]: profile = '{default}' is not a defined profile "
                        f"(available: {', '.join(profiles)})")
    if d.get("provider") == "vast" and d.get("storage") == "volume":
        problems.append('[defaults]: storage = "volume" is RunPod only (vast.ai has no network volumes)')
    return problems


def load_config(path: Path | None = None) -> Config:
    path = path or project.config
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}\n"
                          "Run `kami init` here to create one, or pass --config / set KAMI_CONFIG.")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name} is not valid TOML\n{e}") from None
    problems = validate(raw)
    if problems:
        count = f"{len(problems)} problem{'s' if len(problems) > 1 else ''}"
        raise ConfigError(f"{path.name} has {count}\n" + "\n".join(f"- {p}" for p in problems))
    d = raw.get("defaults", {})
    raw_profiles = raw.get("profiles", {})
    profiles = {}
    for name in raw_profiles:
        p = resolve_profile(name, raw_profiles)
        min_cuda = p.get("min_cuda")
        profiles[name] = Profile(name=name, gpus=list(p["gpus"]),
                                 mesh_model=p.get("mesh_model", ""),
                                 image_model=p.get("image_model", ""),
                                 views_model=p.get("views_model", ""),
                                 views_lora=p.get("views_lora", ""),
                                 views_defaults=dict(p.get("views_defaults", {})),
                                 extra_models=list(p.get("extra_models", [])),
                                 dino_model=p.get("dino_model", ""),
                                 rembg_model=p.get("rembg_model", ""),
                                 mesh_subfolder=p.get("mesh_subfolder", ""),
                                 paint_model=p.get("paint_model", ""),
                                 accept_hunyuan_license=bool(p.get("accept_hunyuan_license", False)),
                                 image_defaults=dict(p.get("image_defaults", {})),
                                 mesh_defaults=dict(p.get("mesh_defaults", {})),
                                 offload=bool(p.get("offload", False)),
                                 disk_gb=int(p.get("disk_gb", 80)),
                                 gpu_count=int(p.get("gpu_count", 1)),
                                 min_vram_gb=int(p.get("min_vram_gb", 0)),
                                 min_ram_gb=int(p.get("min_ram_gb", 0)),
                                 container=p.get("container"),
                                 min_cuda=str(min_cuda) if min_cuda is not None else None)
    storage = d.get("storage", "ephemeral")
    default_profile = d.get("profile", next(iter(profiles)))
    provider = d.get("provider", "runpod")
    check_provider(provider, storage)
    return Config(
        container=d.get("container", DEFAULT_CONTAINER),
        storage=storage,
        datacenter=d.get("datacenter", ""),
        volume_name=d.get("volume_name", "kami-models"),
        volume_size_gb=int(d.get("volume_size_gb", 100)),
        cloud=d.get("cloud", "SECURE"),
        port=int(d.get("port", 8000)),
        output_dir=d.get("output_dir", "kami-out"),
        max_hours=float(d.get("max_hours", 4)),
        idle_minutes=float(d.get("idle_minutes", 30)),
        default_profile=default_profile,
        profiles=profiles,
        provider=provider,
        print_hint=d.get("print_hint"),
    )


DEFAULT_CONTAINER = f"ghcr.io/{PLACEHOLDER}/kami-server:0.1.0"


def fallback_config() -> Config:
    """The default settings and no profiles: what `down` / `status` / `logs` use when
    kami.toml is broken, so a typo never keeps a billing pod out of reach."""
    return Config(container=DEFAULT_CONTAINER, storage="ephemeral", datacenter="",
                  volume_name="kami-models", volume_size_gb=100, cloud="SECURE", port=8000,
                  output_dir="kami-out", max_hours=4, idle_minutes=30, default_profile="",
                  profiles={})


PROVIDERS = ("runpod", "vast")


def check_provider(provider: str, storage: str) -> None:
    if provider not in PROVIDERS:
        raise ConfigError(f"provider must be 'runpod' or 'vast', not '{provider}'")
    if provider == "vast" and storage == "volume":
        raise ConfigError('storage = "volume" is RunPod only: vast.ai has no network volumes\n'
                          'Use storage = "ephemeral" with vast (the weights download on each boot).')


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ without overriding existing vars."""
    path = path or project.env
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(f"Missing {name}\nAdd it to {project.env} (`kami init` creates the file).")
    return value


def load_state() -> dict:
    if not project.state.exists():
        return {}
    try:
        return json.loads(project.state.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ConfigError(
            f"State file is unreadable: {project.state}\n"
            f"{e}\n"
            "If a pod is still running, delete it from the provider's console, then remove this file."
        ) from None


def save_state(state: dict) -> None:
    """Write state atomically so an interrupted write cannot corrupt the file."""
    project.state_dir.mkdir(exist_ok=True)
    tmp = project.state.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(project.state)
