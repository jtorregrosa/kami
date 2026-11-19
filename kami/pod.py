"""Pod plumbing shared by the commands: the provider client (RunPod or vast.ai), the
server endpoint, the pod definition (kami-server under the in-pod guard), log streaming
and termination."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from . import USER_AGENT, catalog, config, ui, vast
from .api import ProviderError
from .config import HUNYUAN_PAINT_MODEL, Config, ConfigError, load_state, require_env, save_state
from .runpod import CONTAINER_DISK_PRICE_GB_MONTH, RunPod
from .vast import Vast

VOLUME_MOUNT = "/models"


EPHEMERAL_CACHE = "/root/.cache/huggingface"


GUARD_SOURCE = Path(__file__).with_name("guard.py")


SERVER_SOURCE = Path(__file__).with_name("server.py")


VOLUME_PRICE_GB_MONTH = 0.07     # RunPod network volume, $/GB/month (approximate)




VOLUME_POD_DISK_GB = 30          # container disk of a pod whose weights live on the volume


# Pod states where the container is no longer serving (it may still hold its disk).
DEAD_STATES = ("TERMINATED", "EXITED", "ERROR")


# State keys that describe the registered pod.
POD_KEYS = ("pod_id", "api_key", "profile", "model", "deadline", "idle_minutes",
            "created_at", "balance_at_up", "provider", "url")


# Pods kami created, newest last (state["history"]), so `kami cost` can name the
# profile behind each billed pod id. Kept across pods; capped.
HISTORY_MAX = 100


# Log lines that mean kami-server will not start.
FATAL_MARKERS = ("[kami-server] could not load the pipelines",
                 "[kami-server] neither KAMI_MESH_MODEL", "[kami-server] KAMI_API_KEY is not set")


# Log lines that mean the host's GPU is unusable (seen on community hosts): torch would
# carry on in CPU (or fail late), so kami stops at once.
NO_GPU_MARKERS = ("failed to initialize CUDA", "no usable GPU found",
                  "No CUDA GPUs are available", "CUDA driver initialization failed")


# RunPod restarts the container when the process dies; seeing it twice = crash loop.
CONTAINER_START = "start container for"


ERROR_LINE = re.compile(r"\b\w+(Error|Exception): ")


# Exceptions that only wrap the real cause; used only when there is nothing better.
GENERIC_ERRORS = ("RuntimeError: CUDA error: an illegal memory access",)


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def endpoint(pod: dict, port: int) -> str | None:
    """The server's URL for a pod as the provider returns it (None while vast.ai
    has not published the port yet)."""
    if pod.get("provider") == "vast":
        return vast.endpoint(pod, port)
    return proxy_url(pod["id"], port)


def pod_url(cfg: Config, state: dict) -> str | None:
    """The registered pod's URL: recorded by `up`, or RunPod's proxy for older states.
    None while a vast.ai pod has not published its port yet."""
    if state.get("url"):
        return state["url"]
    return proxy_url(state["pod_id"], cfg.port) if pod_provider(state) == "runpod" else None


def require_url(cfg: Config, state: dict) -> str:
    url = pod_url(cfg, state)
    if not url:
        raise ConfigError("The pod has no endpoint yet\n"
                          "vast.ai publishes the port once the container runs: `kami status`.")
    return url


def server_metrics(url: str, api_key: str, timeout: float = 5) -> dict[str, float]:
    """kami-server's Prometheus /metrics, summed over labels ({} when unavailable)."""
    req = urllib.request.Request(url + "/metrics", headers={
        "Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return {}
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0].strip()
        try:
            metrics[name] = metrics.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return metrics


def server_usage(metrics: dict[str, float]) -> dict | None:
    """Jobs since the server started (running, queued, finished, failed), or None."""
    if not any(k.startswith("kami:") for k in metrics):
        return None
    m = metrics.get
    return {"running": int(m("kami:jobs_running", 0)), "queued": int(m("kami:jobs_queued", 0)),
            "finished": int(m("kami:jobs_finished_total", 0)),
            "failed": int(m("kami:jobs_failed_total", 0))}


def healthy(url: str, timeout: float = 10) -> bool:
    req = urllib.request.Request(url + "/health", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


# Each provider's client and the .env variable holding its API key.
PROVIDERS = {"runpod": (RunPod, "RUNPOD_API_KEY"), "vast": (Vast, "VAST_API_KEY")}


def client(provider: str = "runpod") -> RunPod | Vast:
    cls, key = PROVIDERS[provider]
    return cls(require_env(key))


def pod_provider(state: dict) -> str:
    """The provider the registered pod runs on (RunPod for states written before kami
    knew other providers) -- never the configured one, which may differ."""
    return state.get("provider") or "runpod"


def pod_client(state: dict) -> RunPod | Vast:
    return client(pod_provider(state))


class LogStream:
    """Read the pod logs on a daemon thread and hand them off through a queue.

    On Windows a blocking socket read is not interrupted by Ctrl+C, so the stream is
    never read from the main thread. With reconnect=True, a dropped stream is reopened
    without backfill (so lines are not duplicated).
    """

    DONE = object()

    def __init__(self, rp, pod_id: str, tail: int, reconnect: bool = False):
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(rp, pod_id, tail, reconnect), daemon=True).start()

    def _run(self, rp, pod_id: str, tail: int, reconnect: bool) -> None:
        while True:
            try:
                for event in rp.stream_logs(pod_id, tail=tail):
                    self.lines.put(event.get("line", ""))
            except Exception as e:
                if not reconnect:
                    self.lines.put(e)
                    break
            if not reconnect:
                break
            tail = 0
            time.sleep(5)
        self.lines.put(self.DONE)

    def drain(self) -> list[str]:
        """Lines available right now, without blocking."""
        out = []
        while True:
            try:
                item = self.lines.get_nowait()
            except queue.Empty:
                return out
            if isinstance(item, str):
                out.append(item)


class LogSnapshot:
    """The pod's latest log lines, read off a fresh stream on a daemon thread.

    A long-lived stream can deliver the container's lines minutes late, while a new one
    backfills them at once; the stream never ends by itself, so it is closed after
    `wait` seconds. Poll `done` / `lines` from the main thread (never blocks)."""

    def __init__(self, rp, pod_id: str, tail: int = 200, wait: float = 4.0):
        self.lines: list[str] = []
        self.done = False
        threading.Thread(target=self._run, args=(rp, pod_id, tail, wait), daemon=True).start()

    def _run(self, rp, pod_id: str, tail: int, wait: float) -> None:
        resp = None
        try:
            resp = rp.open_logs(pod_id, tail)
            timer = threading.Timer(wait, resp.close)
            timer.daemon = True
            timer.start()
            for event in rp.stream_logs(pod_id, resp=resp):
                self.lines.append(event.get("line", ""))
        except Exception:
            pass            # a snapshot is best effort; the main stream still runs
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            self.done = True


def terminate(rp, state: dict) -> str | None:
    """Delete the registered pod and clear it from state. Returns its id."""
    pod_id = state.get("pod_id")
    if not pod_id:
        return None
    try:
        rp.delete_pod(pod_id)
    except ProviderError as e:
        if e.status != 404:
            raise
    for entry in state.get("history", []):
        if entry.get("pod_id") == pod_id and not entry.get("deleted_at"):
            entry["deleted_at"] = time.time()
    clear_pod(state)
    return pod_id


def terminate_own(rp, pod_id: str) -> None:
    """Delete `pod_id` -- the pod this long-running process (`up`, `watchdog`) created or
    guards -- re-reading the state first. Its in-memory copy may be old: if `kami down`
    and a new `up` ran meanwhile, saving it would unregister the new pod, which would then
    bill without `kami status` seeing it. The state changes only where it is about `pod_id`."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        terminate(rp, state)
        return
    try:
        rp.delete_pod(pod_id)
    except ProviderError as e:
        if e.status != 404:
            raise
    marked = False
    for entry in state.get("history", []):
        if entry.get("pod_id") == pod_id and not entry.get("deleted_at"):
            entry["deleted_at"] = time.time()
            marked = True
    if marked:
        save_state(state)


def forget_own(pod_id: str) -> None:
    """Clear `pod_id` from the state if it is still the registered pod (see terminate_own)."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        clear_pod(state)


def orphans(state: dict) -> list[tuple[object, dict]]:
    """Pods named kami-* on any provider with a key set that are not this project's
    registered pod: (client, pod). They bill unseen -- a crashed `up`, or another
    project's pod. Providers that cannot be reached are skipped."""
    found = []
    for provider, (_, key) in PROVIDERS.items():
        if not os.environ.get(key):
            continue
        rp = client(provider)
        own = state.get("pod_id") if pod_provider(state) == provider else None
        try:
            pods = rp.list_pods()
        except ProviderError:
            continue
        found += [(rp, p) for p in pods
                  if str(p.get("name") or "").startswith("kami-") and str(p["id"]) != own]
    return found


def record_boot(pod_id: str, **fields) -> None:
    """Add how the pod booted (seconds, download...) to its history entry."""
    state = load_state()
    entry = next((h for h in reversed(state.get("history", [])) if h.get("pod_id") == pod_id), None)
    if entry is not None:
        entry.update(fields)
        save_state(state)


BOOT_SAMPLES = 5                # the typical boot is the median of this many recent ones


def typical_boot(state: dict, profile: str, provider: str, storage: str,
                 gpu: str | None = None) -> tuple[float, int] | None:
    """(median seconds, boots counted) of this profile's recent boots on the same provider
    and storage (a volume skips the download), on `gpu` when it has some; None if none."""
    boots = [h for h in state.get("history", [])
             if h.get("boot_seconds") and h.get("profile") == profile
             and h.get("provider", "runpod") == provider and h.get("storage") == storage]
    same_gpu = [h for h in boots if gpu and h.get("gpu") == gpu]
    recent = sorted(h["boot_seconds"] for h in (same_gpu or boots)[-BOOT_SAMPLES:])
    if not recent:
        return None
    middle = len(recent) // 2
    median = recent[middle] if len(recent) % 2 else (recent[middle - 1] + recent[middle]) / 2
    return median, len(recent)


def remember_pod(state: dict, pod: dict, profile_name: str, provider: str = "runpod") -> None:
    """Add a just-created pod to the history (saved with the next save_state)."""
    history = state.setdefault("history", [])
    history.append({"pod_id": pod["id"], "profile": profile_name, "provider": provider,
                    "gpu": (pod.get("gpu") or {}).get("id"),
                    "cost_per_hour": pod.get("cost"), "created_at": time.time()})
    del history[:-HISTORY_MAX]


def clear_pod(state: dict) -> None:
    """Forget the registered pod (the volume id is kept)."""
    for key in POD_KEYS:
        state.pop(key, None)
    save_state(state)


def root_cause(lines) -> str | None:
    """Last exception in the log (e.g. 'ValueError: ...'), preferring the real cause."""
    errors = []
    for line in lines:
        match = ERROR_LINE.search(line)
        if match:
            errors.append(line[match.start():].strip())
    specific = [e for e in errors if not any(g in e for g in GENERIC_ERRORS)]
    return (specific or errors or [None])[-1]


def guard_text(state: dict) -> str | None:
    """'deletes itself at 02:30 (in 7h 58m) · or after 60m idle', or None if unguarded."""
    parts = []
    deadline = state.get("deadline") or 0
    if deadline:
        left = max(0, deadline - time.time())
        parts.append(f"deletes itself at {time.strftime('%H:%M', time.localtime(deadline))} "
                     f"(in {ui.duration(left)})")
    idle = state.get("idle_minutes") or 0
    if idle:
        parts.append(f"after {ui.duration(idle * 60)} idle")
    return " · or ".join(parts) or None


def session_cost(pod: dict, since: float | None = None) -> str | None:
    """'1h 12m · ≈ $1.31 (estimate)' from the hourly cost and the time billed so far.

    Providers bill from the pod's creation, so `since` (the creation time kami recorded)
    is used when known; the container's uptime leaves out the image pull and the model
    download, and under-counts. Either way it is an estimate: see `kami cost`."""
    seconds = time.time() - since if since else (pod.get("runtime") or {}).get("uptime")
    cost = pod.get("cost")
    if seconds is None:
        return None
    text = ui.duration(seconds)
    if isinstance(cost, (int, float)):
        text += f" · ≈ ${cost * seconds / 3600:.2f} (estimate)"
    return text


def real_session_cost(rp, state: dict) -> float | None:
    """Money the account lost since `kami up` (balance then minus balance now): the
    real figure, available at once -- billing per pod is only published ~1 h later.
    It includes anything else billing on the account meanwhile (other pods, volumes), and
    lags behind: the balance is debited in batches. None when unknown -- also when the
    balance grew (a top-up meanwhile), which would otherwise read as $0.00."""
    before = state.get("balance_at_up")
    if before is None:
        return None
    try:
        spent = before - rp.account()["balance"]
    except ProviderError:
        return None
    return spent if spent >= 0 else None


def ensure_volume(rp: RunPod, cfg: Config, state: dict) -> str:
    """Return the network volume id, creating it if needed."""
    if not cfg.datacenter:
        raise ConfigError("Missing datacenter for the volume\n"
                          "Set it in kami.toml (`kami gpus` shows where there is stock).")
    volumes = rp.list_volumes()
    for vol in volumes:
        if vol["name"] == cfg.volume_name and vol["dataCenter"] == cfg.datacenter:
            state["volume_id"] = vol["id"]
            save_state(state)
            return vol["id"]
    ui.step(f"Creating network volume {cfg.volume_name} ({cfg.volume_size_gb} GB) in {cfg.datacenter} "
            f"{ui.dim('· ≈ ' + monthly(cfg.volume_size_gb))}")
    vol = rp.create_volume(cfg.volume_name, cfg.volume_size_gb, cfg.datacenter)
    state["volume_id"] = vol["id"]
    save_state(state)
    # A changed datacenter leaves the old volume behind, still billing every month.
    for old in volumes:
        if old["name"] == cfg.volume_name:
            ui.warn(f"Volume {old['id']} ({old['dataCenter']}) has the same name and still bills "
                    f"≈ {monthly(old.get('size', 0))}")
            ui.hint(f"`kami volume rm {old['id']}` if you no longer need it")
    return vol["id"]


def monthly(size_gb: float) -> str:
    return f"${size_gb * VOLUME_PRICE_GB_MONTH:.2f}/month"


# Lowest host CUDA version, unless the profile sets min_cuda: the kami-server image is
# built on CUDA 12.4 (TRELLIS.2's toolkit).
DEFAULT_MIN_CUDA = "12.4"


# Where the pipelines download their weights (the guard reports its growth while the
# server boots).
def _hf_home(volume_id: str | None) -> str:
    return f"{VOLUME_MOUNT}/hf" if volume_id else EPHEMERAL_CACHE


def pack(path: Path) -> str:
    """A source file as a `python3 -c` program, compressed: it travels in the pod's command
    line (or env), which providers may cap in size."""
    packed = base64.b64encode(zlib.compress(path.read_bytes(), 9)).decode()
    return f"import base64,zlib;exec(zlib.decompress(base64.b64decode('{packed}')))"


def guard_program() -> str:
    return pack(GUARD_SOURCE)


# The server's source arrives in KAMI_SERVER_SRC: the command line carries the guard only.
# After a `kami reload` the newer code saved at KAMI_RELOAD_PATH runs instead.
SERVER_BOOT = ("import os;p=os.environ.get('KAMI_RELOAD_PATH','/tmp/kami-server.py');"
               "exec(open(p).read() if os.path.exists(p) else os.environ['KAMI_SERVER_SRC'])")


def server_spec(cfg: Config, profile, volume_id: str | None, api_key: str,
                deadline: float = 0, idle_seconds: float = 0, guard_env: dict | None = None):
    """(env, entrypoint, cmd, image) of the pod, whatever the provider: kami-server
    (server.py) under the in-pod guard (guard.py), which reports the boot and, with a
    deadline or an idle limit, deletes the pod from inside. `guard_env` adds provider
    settings for it."""
    env = {
        "HF_HOME": _hf_home(volume_id),
        # parallel downloads with hf_xet: matters mostly in ephemeral mode
        "HF_XET_HIGH_PERFORMANCE": "1",
        # an env var keeps the key out of the container command line
        "KAMI_API_KEY": api_key,
        "KAMI_MESH_MODEL": profile.mesh_model,
        "KAMI_IMAGE_MODEL": profile.image_model,
        "KAMI_SERVER_SRC": pack(SERVER_SOURCE),
    }
    if profile.image_defaults:
        env["KAMI_IMAGE_DEFAULTS"] = json.dumps(profile.image_defaults)
    if profile.mesh_defaults:
        env["KAMI_MESH_DEFAULTS"] = json.dumps(profile.mesh_defaults)
    if profile.offload:
        env["KAMI_OFFLOAD"] = "1"
    if profile.mesh_model and profile.dino_model:
        env["KAMI_DINO_MODEL"] = profile.dino_model
    if profile.mesh_model and profile.rembg_model:
        env["KAMI_REMBG_MODEL"] = profile.rembg_model
    if profile.mesh_model and profile.mesh_backend == "hunyuan":
        env["KAMI_MESH_BACKEND"] = "hunyuan"
        env["KAMI_MESH_SUBFOLDER"] = profile.shape_subfolder
        env["KAMI_PAINT_MODEL"] = profile.paint_model or HUNYUAN_PAINT_MODEL
    if volume_id:
        # torch.compile / kernel caches survive on the volume between pods
        env["TORCH_HOME"] = f"{VOLUME_MOUNT}/torch"
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    # "utility" has the NVIDIA runtime mount nvidia-smi, which the guard reads GPU memory with
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
    env.update(KAMI_DEADLINE=str(int(deadline)), KAMI_IDLE=str(int(idle_seconds)),
               KAMI_PORT=str(cfg.port), KAMI_WATCH=env["HF_HOME"])
    env.update(guard_env or {})
    entrypoint = ["python3", "-c", guard_program(), "python3", "-c", SERVER_BOOT]
    return env, entrypoint, [], profile.container or cfg.container


def min_cuda(profile) -> str | None:
    return getattr(profile, "min_cuda", None) or DEFAULT_MIN_CUDA


def pod_body(cfg: Config, profile, gpu_id: str, volume_id: str | None, api_key: str,
             deadline: float = 0, idle_seconds: float = 0, cloud: str | None = None) -> dict:
    """The RunPod pod definition (POST /pods)."""
    env, entrypoint, cmd, image = server_spec(cfg, profile, volume_id, api_key, deadline, idle_seconds)
    gpu = {"id": catalog.provider_gpu(gpu_id, "runpod"), "count": profile.gpu_count}
    cuda = min_cuda(profile)
    if cuda:
        # only hosts whose driver supports this CUDA version (RunPod filters on create)
        gpu["minCudaVersion"] = cuda
    if getattr(profile, "min_ram_gb", 0):
        # only hosts with this much system RAM per GPU (a placement filter)
        gpu["minRamPerGpu"] = profile.min_ram_gb
    body = {
        "name": f"kami-{profile.name}",
        "image": image,
        "cloud": cloud or cfg.cloud,
        "gpu": gpu,
        # with a volume the pod must go to its datacenter; in ephemeral only if one is set
        "dataCenterIds": [cfg.datacenter] if cfg.datacenter else [],
        "disk": VOLUME_POD_DISK_GB if volume_id else profile.disk_gb,
        "ports": [f"{cfg.port}/http"],
        "env": env,
        "entrypoint": entrypoint,
        "cmd": cmd,
    }
    if volume_id:
        body["mounts"] = {"network": [{"volumeId": volume_id, "path": VOLUME_MOUNT}]}
    return body


# The guard settings of a vast.ai pod: how to delete itself, and an HTTPS tunnel, since
# vast.ai publishes the model port as plain HTTP on the host's IP (see guard.py).
VAST_GUARD_ENV = {"KAMI_PROVIDER": "vast", "KAMI_TUNNEL": "1"}


def vast_offer(cfg: Config, profile, gpu: str, cloud: str | None = None) -> dict:
    """The vast.ai offer search filters for `profile` on `gpu`: what `up` would rent and
    what `prices` quotes. Raises ConfigError for a GPU vast.ai has no name for."""
    name = catalog.provider_gpu(gpu, "vast")
    row = catalog.find(gpu)
    count = profile.gpu_count
    offer = {
        **catalog.VAST_BASE_FILTERS,
        **catalog.VAST_CLOUDS[cloud or cfg.cloud],
        "gpu_name": {"eq": name},
        "num_gpus": {"eq": count},
        "disk_space": {"gte": profile.disk_gb},
        # the same name covers several memory sizes on vast (A100 40/80 GB): MiB per card
        "gpu_ram": {"gte": catalog.min_ram_mib(row)},
    }
    if getattr(profile, "min_ram_gb", 0):
        offer["cpu_ram"] = {"gte": profile.min_ram_gb * 1024 * count}     # MiB, the offer's share
    if row.vast_max_dph:
        offer["dph_base"] = {"lte": row.vast_max_dph * count}
    cuda = min_cuda(profile)
    if cuda:
        offer["cuda_max_good"] = {"gte": float(cuda)}
    return offer


def vast_body(cfg: Config, profile, gpu: str, api_key: str, deadline: float = 0,
              idle_seconds: float = 0, cloud: str | None = None) -> dict:
    """What vast.ai's client rents: {'offer': search filters, 'instance': PUT /asks body}.
    The guard always runs on vast.ai, for its tunnel even without limits."""
    offer = vast_offer(cfg, profile, gpu, cloud)
    env, entrypoint, cmd, image = server_spec(cfg, profile, None, api_key, deadline,
                                              idle_seconds, guard_env=dict(VAST_GUARD_ENV))
    instance = {
        "image": image,
        "disk": profile.disk_gb,
        # "args": the container runs as is (no ssh/jupyter injected), with this entrypoint
        "runtype": "args",
        "entrypoint": entrypoint[0],
        "args": [*entrypoint[1:], *cmd],
        # docker options go in env: publish the model port
        "env": {**env, f"-p {cfg.port}:{cfg.port}": "1"},
        "label": f"kami-{profile.name}",
        # fail instead of leaving a stopped instance (billing storage) when the host is busy
        "cancel_unavail": True,
    }
    return {"offer": offer, "instance": instance}


def make_body(provider: str):
    """The pod-definition builder of a provider (same arguments for both)."""
    if provider == "vast":
        return lambda cfg, profile, gpu, volume_id, api_key, deadline, idle, cloud: \
            vast_body(cfg, profile, gpu, api_key, deadline, idle, cloud)
    return pod_body


def pod_problem(cfg: Config, state: dict) -> str | None:
    """Why the registered pod cannot run jobs, or None if the server answers.
    A pod that no longer exists (or is dead and then deleted) is reported as 'gone…'."""
    url = pod_url(cfg, state)
    if url and healthy(url, timeout=5):
        return None
    rp = pod_client(state)
    try:
        pod = rp.get_pod(state["pod_id"])
    except ProviderError as e:
        if e.status != 404:
            raise
        clear_pod(state)
        return f"gone (it no longer exists on {rp.label}; the guard may have deleted it)"
    if pod["status"] in DEAD_STATES:
        terminate(rp, state)
        return f"gone (it was {pod['status']}; deleted)"
    return f"{pod['status']} but the server is not ready yet (still loading the models?)"
