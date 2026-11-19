"""kami: rent a GPU only while you use it, to turn images into 3D meshes (TRELLIS.2)
and prompts into images (diffusers)."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import __version__, catalog, config, fit, jobs, mascot, ui, webui
from .api import ProviderError
from .config import Config, ConfigError, load_config, load_env, load_state, save_state
from .pod import (CONTAINER_DISK_PRICE_GB_MONTH, CONTAINER_START, DEAD_STATES, FATAL_MARKERS,
                  NO_GPU_MARKERS, PROVIDERS, VOLUME_POD_DISK_GB, LogSnapshot, LogStream,
                  clear_pod, client, endpoint, ensure_volume, guard_text, healthy,
                  make_body, monthly, orphans, pod_problem,
                  pod_provider, real_session_cost, record_boot, remember_pod, root_cause,
                  SERVER_SOURCE, server_metrics, server_usage, session_cost, terminate,
                  terminate_own, typical_boot, vast_offer)
from .watchdog import WATCHDOG_POLL, cmd_watchdog

# Waiting for the container to come up. vast.ai hosts pull the server image themselves
# (RunPod's are cached), which can take a while on a slow host.
POD_READY_TIMEOUT = {"runpod": 10 * 60, "vast": 30 * 60}


TUNNEL_WAIT = 90                 # seconds to wait for a vast.ai pod's HTTPS tunnel once ready


# The guard logs the URL of its HTTPS tunnel (vast.ai pods; see guard.py).
TUNNEL_LINE = re.compile(r"\[kami-guard\] tunnel (https://\S+)")


MODEL_READY_TIMEOUT = 45 * 60    # weight download (always in ephemeral, first time with a volume)
                                 # plus a vast.ai host pulling the ~15 GB server image


WARN_AFTER_HOURS = 4             # status warns about a long-running (billing) session


SNAPSHOT_EVERY = 20              # seconds between fresh log snapshots while `up` waits


FATAL_GRACE = 15                 # seconds to collect the traceback after the marker


# Startup phase from the latest recognised log line (in order of appearance).
PHASES = (
    ("Pulling from", "Pulling the server image"),
    (CONTAINER_START, "Starting the server"),
    ("[kami-server] listening", "Starting kami-server"),
    ("[kami-server] loading mesh pipeline", "Loading the mesh model"),
    ("[kami-server] loading image pipeline", "Loading the image model"),
    ("[kami-server] server is ready", "Opening the server"),
)


# The guard's boot reports (see guard.py): host checks, then the bytes downloaded and the
# GPU memory in use every few seconds, until the server answers.
GUARD_CHECK = re.compile(r"\[kami-guard\] check (ok|warn) (.+)")
GUARD_PROGRESS = re.compile(r"\[kami-guard\] progress t=(\d+) downloaded=(\d+) rate=(\d+)"
                            r"(?: vram=(\d+)/(\d+))?")
DOWNLOAD_PHASE = "Downloading the model weights"
LOAD_PHASE = "Loading the weights onto the GPU"
MIB = 1024 ** 2
SLOW_DOWNLOAD = 25e6            # bytes/s: below this for SLOW_AFTER s, suggest another host
SLOW_AFTER = 90


class BootReport:
    """Turns the guard's boot reports into `up`'s output: a line per host check, and on
    the progress line a bar for the download (size, speed, time left) and for the
    weights reaching the GPU. `total` is the download size in bytes, if known; with
    `gpu_bar`, the weights take that much GPU memory too (not so with online quantisation)."""

    def __init__(self, progress: ui.Progress, total: float | None,
                 gpu_bar: bool = True, hourly: float | None = None):
        self.progress, self.total, self.hourly = progress, total, hourly
        self.slow_since: int | None = None          # pod time the download turned slow
        self.slow_warned = False
        self.download_phase = DOWNLOAD_PHASE
        self.gpu_bar = gpu_bar
        self.shown: set[str] = set()
        self.last_t = 0
        self.got = 0
        self.span: tuple[int, int] | None = None     # pod time of the first / last growth
        self.vram_base: float | None = None
        self.summarised = False

    def line(self, line: str) -> bool:
        """Read a log line; True if it was a progress report (noise for failure logs)."""
        match = GUARD_CHECK.search(line)
        if match:
            kind, text = match.groups()
            if text not in self.shown:            # snapshots repeat lines
                self.shown.add(text)
                self.progress.stop()
                (ui.step if kind == "ok" else ui.warn)(text)
            return False
        match = GUARD_PROGRESS.search(line)
        if not match:
            return False
        t, got, rate = (int(g) for g in match.groups()[:3])
        if t < self.last_t:                     # a late line from a stale stream
            return True
        self.last_t = t
        if got > self.got:
            self.span = (self.span[0] if self.span else t, t)
            self.got = got
            if self.total and got > self.total:
                # more than the size kami expected (a file it did not count): grow the bar
                self.total = got
            if self.downloading and self.progress.text != self.download_phase:
                self.progress.phase(self.download_phase)
        used, total_mib = (float(g) if g else None for g in match.groups()[3:])
        if used is not None and self.vram_base is None:
            self.vram_base = used               # the CUDA context, before any weights
        if self.progress.text == self.download_phase:
            self.slow_check(t, rate)
            self.progress.bar(*self.download_bar(rate))
        elif self.progress.text == LOAD_PHASE and used is not None:
            self.summary()
            self.progress.bar(*self.gpu_bar_of(used, total_mib))
        else:
            self.summary()
        return True

    def slow_check(self, t: int, rate: int) -> None:
        """Suggest another host once the download stays slow (the pod bills meanwhile);
        a suggestion only: the download keeps going."""
        if rate >= SLOW_DOWNLOAD or not self.downloading:
            self.slow_since = None
            return
        self.slow_since = self.slow_since or t
        if self.slow_warned or t - self.slow_since < SLOW_AFTER:
            return
        self.slow_warned = True
        text = f"Slow download from this host: {rate / 1e6:.0f} MB/s"
        if self.total and rate > 0:
            left = (self.total - self.got) / rate
            text += f", ~{ui.duration(left)} left"
            if self.hourly:
                text += f" (≈ ${self.hourly * left / 3600:.2f} of pod time)"
        self.progress.stop()
        ui.warn(text)
        ui.hint("Hosts vary a lot: `kami down` and `kami up` again usually lands on a faster "
                "one. This one keeps downloading meanwhile.")

    @property
    def downloading(self) -> bool:
        return not self.total or self.got < self.total

    def download_bar(self, rate: int) -> tuple[float | None, str]:
        size = f"{self.got / 1e9:.1f}"
        if self.total:
            size += f"/{self.total / 1e9:.1f}"
        detail = f"{size} GB · {rate / 1e6:.0f} MB/s"
        if self.total and rate > 0 and self.downloading:
            detail += f" · {ui.duration((self.total - self.got) / rate)} left"
        return (self.got / self.total if self.total else None), detail

    def gpu_bar_of(self, used: float, total_mib: float) -> tuple[float | None, str]:
        detail = f"{used / 1024:.1f}/{total_mib / 1024:.0f} GiB on the GPU"
        if not (self.gpu_bar and self.total):
            return None, detail
        # never 100 % before the server answers: the KV cache and the graphs follow
        return min(0.99, (used - (self.vram_base or 0)) * MIB / self.total), detail

    def summary(self) -> None:
        """Once the download is over: its size, time and average speed."""
        if self.summarised or not self.span or self.progress.text == self.download_phase:
            return
        self.summarised = True
        seconds = max(1, self.span[1] - self.span[0])
        self.progress.stop()
        ui.ok(f"Downloaded {self.got / 1e9:.1f} GB in {ui.duration(seconds)} "
              f"{ui.dim(f'· {self.got / seconds / 1e6:.0f} MB/s')}")


# --- commands ---

def cmd_init(args, cfg: Config | None) -> None:
    """Set up the project in the current directory (or the one --config points at)."""
    # Not the discovered project: a kami.toml up the tree or in kami's own checkout
    # is not where a new project should be created.
    target = args.config or os.environ.get("KAMI_CONFIG") or Path.cwd() / config.CONFIG_NAME
    project = config.project = config.Project(Path(target))
    load_env()
    if project.config.exists():
        ui.ok(f"Config found: {project.config}")
    else:
        project.config.write_text(config.STARTER_CONFIG, encoding="utf-8")
        ui.ok(f"Created {project.config}")
        ui.hint("Set `container` to your kami-server image (README: Build the server image).")
    if project.env.exists():
        ui.ok(f"Secrets file found: {project.env}")
    else:
        project.env.write_text(config.STARTER_ENV, encoding="utf-8")
        ui.ok(f"Created {project.env}")
    ignore = project.root / ".gitignore"
    if ignore.exists():
        text = ignore.read_text(encoding="utf-8")
        missing = [entry for entry in (".env", ".kami/") if entry not in text.splitlines()]
        if missing:
            with ignore.open("a", encoding="utf-8") as f:
                f.write(("" if not text or text.endswith("\n") else "\n") + "\n".join(missing) + "\n")
            ui.ok(f"Added {', '.join(missing)} to .gitignore (secrets and local state)")
    cfg = load_config()
    key = PROVIDERS[cfg.provider][1]
    if not os.environ.get(key):
        print()
        ui.warn(f"{key} is empty")
        ui.hint(f"Paste your key into {project.env} ({KEY_PAGES[cfg.provider]}), "
                "then run `kami doctor`.")
        return
    if cfg.storage == "volume":
        vol_id = ensure_volume(client(), cfg, load_state())
        ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")
    print()
    ui.hint("Next: `kami doctor` to check everything, then `kami up`.")


KEY_PAGES = {"runpod": "https://console.runpod.io/user/settings",
             "vast": "https://cloud.vast.ai/manage-keys/"}


def cmd_doctor(args, cfg: Config | None) -> None:
    """Check the setup end to end and report every problem, not just the first."""
    problems = 0

    def bad(msg: str, hint: str | None = None) -> None:
        nonlocal problems
        problems += 1
        ui.error(f"{msg}\n{hint}" if hint else msg)

    project = config.project
    try:
        cfg = load_config()
    except ConfigError as e:
        bad(str(e))
        cfg = None
    else:
        ui.ok(f"Config: {project.config} {ui.dim(f'· {len(cfg.profiles)} profile(s) · default {cfg.default_profile}')}")
        problem = container_problem(cfg, None)
        if problem:
            bad(*problem)
        else:
            ui.ok(f"Server image: {cfg.container}")
        access = {p.name: fit.access_problems(p) for p in cfg.profiles.values()}
        for name, repo_problems in access.items():
            for problem in repo_problems:
                bad(f"Profile {name}: {problem}")
        if not any(access.values()):
            ui.ok("Hugging Face: every profile's weights can be downloaded")
        if not os.environ.get("HF_TOKEN") and any(p.mesh_model for p in cfg.profiles.values()):
            ui.warn("HF_TOKEN is not set: TRELLIS.2 loads gated models (DINOv3, RMBG-2.0)")
        if cfg.storage == "volume" and not cfg.datacenter:
            bad('storage = "volume" needs a datacenter', "`kami gpus` shows where there is stock.")
        guard = [f"{cfg.max_hours:g} h max" if cfg.max_hours > 0 else "",
                 f"{cfg.idle_minutes:g} min idle" if cfg.idle_minutes > 0 else ""]
        if any(guard):
            ui.ok(f"Guard: the pod deletes itself after {' or '.join(g for g in guard if g)}")
        else:
            ui.warn("Guard disabled (max_hours and idle_minutes are 0): a forgotten pod bills forever")

    provider = cfg.provider if cfg else "runpod"
    key = PROVIDERS[provider][1]
    label = PROVIDERS[provider][0].label
    rp = None
    if not os.environ.get(key):
        bad(f"{key} is not set", f"Add it to {project.env} (`kami init` creates the file).")
    else:
        rp = client(provider)
        try:
            gpus = rp.list_gpus(cfg.cloud if cfg else "SECURE")
        except ProviderError as e:
            hint = "Check the key in .env." if e.status in (401, 403) else None
            bad(f"{label} API: {e.detail}", hint)
            rp = None
        else:
            ui.ok(f"{label} API key works")
            try:
                balance = rp.account()["balance"]
            except ProviderError:
                balance = None
            if balance is not None and balance <= 0:
                bad(f"{label} balance is ${balance:.2f}: it will not start a pod",
                    "Add credit in the provider's billing page.")
            if cfg and provider == "vast":
                for p in cfg.profiles.values():
                    for gpu in p.gpus:
                        try:
                            catalog.provider_gpu(gpu, "vast")
                        except ConfigError as e:
                            bad(f"Profile {p.name}: {e}")
            if cfg:
                profile = cfg.profile(None)
                # the datacenter is RunPod's; vast.ai lists countries there
                dc = cfg.datacenter if provider == "runpod" else ""
                stock = {catalog.key(g["id"]): g for g in gpus}
                available = [g for g in profile.gpus
                             if stock.get(catalog.key(g), {}).get("availability") not in (None, "NONE")
                             and (not dc or dc in [d["id"] for d in
                                                   stock.get(catalog.key(g), {}).get("dataCenters", [])])]
                where = dc or ("any datacenter" if provider == "runpod" else label)
                if available:
                    ui.ok(f"Stock for profile {profile.name}: {', '.join(available)} {ui.dim('in ' + where)}")
                else:
                    ui.warn(f"No stock right now for profile {profile.name}'s GPUs in {where}")
                    ui.hint("`kami gpus` to see alternatives.")
                for p in cfg.profiles.values():
                    doctor_memory(p, gpus, bad)

    state = load_state()
    if state.get("pod_id"):
        ui.warn(f"A pod is registered: {state['pod_id']} · `kami status` to see what it costs")
    doctor_orphans(state, bad)

    print()
    if problems:
        raise ConfigError(f"{problems} problem{'s' if problems > 1 else ''} found")
    ui.ok("All good. `kami up` to start.")


CONSOLES = {"runpod": "https://www.runpod.io/console/pods",
            "vast": "https://cloud.vast.ai/instances/"}


def doctor_orphans(state: dict, bad) -> None:
    """Pods kami named that this project does not know: they bill unseen."""
    for rp, p in orphans(state):
        cost = p.get("cost")
        bad(f"{rp.label} pod {p['id']} ({p.get('name')}, {p.get('status')}) is not registered "
            f"here and may be billing ≈ ${cost if isinstance(cost, (int, float)) else 0:.2f}/h",
            f"Delete it at {CONSOLES[rp.name]} if it is not another project's.")


def container_problem(cfg: Config, profile) -> tuple[str, str] | None:
    """(problem, hint) when the server image is still the starter's placeholder."""
    image = (profile.container if profile else None) or cfg.container
    if config.PLACEHOLDER in image:
        return (f"The server image is not set: container = \"{image}\"",
                "Build and push docker/Dockerfile (README: Build the server image), "
                "then set `container` in kami.toml.")
    return None


def doctor_memory(profile, catalog: list[dict], bad) -> None:
    if not profile.min_vram_gb:
        return
    results = fit.fits(profile, catalog)
    if not any(f.fits for f in results):
        bad(f"Profile {profile.name}: none of its GPUs has the {profile.min_vram_gb} GB it needs")
        return
    for f in results:
        if not f.fits:
            ui.warn(f"Profile {profile.name}: {f.gpu} is too small ({f.vram_gb:g} GB < "
                    f"{profile.min_vram_gb} GB); `kami up` skips it")


def memory_check(rp, cfg: Config, profile) -> list[fit.Fit]:
    """The profile's GPUs with the memory it needs (min_vram_gb), in order; warns about
    the rest. Stops `kami up` before any pod is created (and billed) when none has."""
    unchecked = [fit.Fit(g, None, 0) for g in profile.gpus]
    if not profile.min_vram_gb:
        return unchecked
    try:
        catalog = rp.list_gpus(cfg.cloud)
    except ProviderError as e:
        ui.warn(f"Memory check skipped: {e.detail}")
        return unchecked
    results = fit.fits(profile, catalog)
    usable = [f for f in results if f.fits]
    if not usable:
        raise ConfigError(f"No GPU of profile {profile.name} has the {profile.min_vram_gb} GB it needs\n"
                          "Add a bigger GPU to its `gpus` (`kami gpus`), or turn on `offload`.\n"
                          "No pod was created. `kami up --skip-memory-check` to try anyway.")
    for f in results:
        if not f.fits:
            ui.warn(f"Skipping {f.gpu}: {f.vram_gb:g} GB is under the {profile.min_vram_gb} GB it needs")
    return usable


def cmd_volume(args, cfg: Config) -> None:
    {"ls": volume_ls, "create": volume_create, "rm": volume_rm}[args.action](args, cfg)


def volume_ls(args, cfg: Config) -> None:
    volumes = client("runpod").list_volumes()
    if not volumes:
        ui.step("No network volumes.")
        if cfg.storage == "volume":
            ui.hint("`kami volume create` to create the one in kami.toml")
        return
    current = load_state().get("volume_id")
    w_id = max(len(v["id"]) for v in volumes)
    w_name = max(len(v["name"]) for v in volumes)
    print("  " + ui.dim(f"{'id':<{w_id}}  {'name':<{w_name}}  {'datacenter':<10}  {'size':>7}  cost"))
    for v in volumes:
        size = v.get("size", 0)
        row = (f"{v['id']:<{w_id}}  {v['name']:<{w_name}}  {v['dataCenter']:<10}  "
               f"{str(size) + ' GB':>7}  ≈ {monthly(size)}")
        print(f"  {ui.bold(row) if v['id'] == current else row}")
    total = sum(v.get("size", 0) for v in volumes)
    print()
    ui.hint(f"Total ≈ {monthly(total)} · bold: this project's volume · `kami volume rm <id>` to delete one")


def volume_create(args, cfg: Config) -> None:
    if cfg.storage != "volume":
        ui.warn('storage = "ephemeral" in kami.toml: the volume would not be used')
        ui.hint('Set storage = "volume" first.')
        return
    vol_id = ensure_volume(client("runpod"), cfg, load_state())
    ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")


def volume_rm(args, cfg: Config) -> None:
    rp = client("runpod")
    vol = next((v for v in rp.list_volumes() if v["id"] == args.id), None)
    if vol is None:
        raise ConfigError(f"No network volume with id {args.id}\n`kami volume ls` lists them.")
    what = f"{vol['name']} ({vol['dataCenter']}, {vol.get('size', '?')} GB)"
    if not args.yes:
        if not sys.stdin.isatty():
            raise ConfigError("Refusing to delete a volume without confirmation\nPass --yes.")
        ui.warn(f"This deletes volume {args.id} {what} and every model stored on it.")
        if input("    Type the volume id to confirm: ").strip() != args.id:
            ui.step("Cancelled.")
            return
    rp.delete_volume(args.id)
    state = load_state()
    if state.get("volume_id") == args.id:
        state.pop("volume_id")
        save_state(state)
    ui.ok(f"Volume {args.id} deleted: {what}")


def cmd_profiles(args, cfg: Config) -> None:
    rows = []
    for name, p in cfg.profiles.items():
        gpu = p.gpus[0] + (f" x{p.gpu_count}" if p.gpu_count > 1 else "")
        rows.append((name + (" *" if name == cfg.default_profile else ""), gpu,
                     p.mesh_model or "-", p.image_model or "-", p.views_model or "-"))
    head = ("profile", "GPU", "mesh", "image", "views")
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    print("  " + ui.dim("  ".join(h.ljust(w[i]) if i < 4 else h for i, h in enumerate(head))))
    for name, gpu, mesh, image, views in rows:
        # pad before colouring: ANSI codes would count towards the width
        print(f"  {ui.bold(name.ljust(w[0]))}  {gpu:<{w[1]}}  {mesh:<{w[2]}}  {image:<{w[3]}}  {views}")
    print()
    ui.hint("* default profile · `kami up -p <profile>` to pick another")


def cmd_gpus(args, cfg: Config) -> None:
    cloud_name = "SECURE" if args.secure else cfg.cloud
    cloud = cloud_name.lower()
    rp = client(args.provider or cfg.provider)
    gpus = [g for g in rp.list_gpus(cloud_name) if g.get("memory", 0) >= args.min_vram]
    gpus.sort(key=lambda g: (g.get("price", {}).get(cloud) or 0))
    if not gpus:
        ui.warn(f"No GPUs with {args.min_vram} GB or more in {cloud_name}.")
        return

    stock_color = {"HIGH": ui.green, "MEDIUM": ui.yellow, "LOW": ui.yellow}
    rows = []
    for g in gpus:
        price = g.get("price", {}).get(cloud)
        dcs = [dc["id"] for dc in g.get("dataCenters", [])]
        rows.append((g["id"], f"{g['memory']} GB", f"${price:.2f}" if price is not None else "-",
                     g.get("availability", "-"), dcs))
    w_id = max(len(r[0]) for r in rows)

    where = "datacenters" if rp.name == "runpod" else "countries"
    print(f"  {ui.bold(f'GPUs with {args.min_vram} GB or more')} "
          f"{ui.dim(f'· {rp.label} · {cloud_name} cloud · price per hour')}\n")
    print("  " + ui.dim(f"{'GPU':<{w_id}}  {'VRAM':>6}  {'$/h':>6}  {'stock':<6}  {where}"))
    for gpu_id, vram, price, stock, dcs in rows:
        color = stock_color.get(stock, ui.dim)
        dc_text = ", ".join(ui.bold(d) if d == cfg.datacenter else d for d in dcs) or ui.dim("-")
        print(f"  {gpu_id:<{w_id}}  {vram:>6}  {price:>6}  {color(f'{stock:<6}')}  {dc_text}")
    if rp.name == "vast":
        print()
        ui.hint("Stock = verified offers (HIGH 10+, MEDIUM 3+) · GPU names as profiles write them")
    elif cfg.datacenter:
        print()
        ui.hint(f"Bold: your datacenter ({cfg.datacenter}).")


@dataclass
class Quote:
    """What one GPU of a profile costs on a provider right now."""
    gpu: str                        # as the profile names it
    stock: str | None = None        # None: cannot be had now (no offers / not offered)
    gpu_h: float = 0                # $/h of the GPUs (gpu_count of them)
    disk_h: float = 0               # $/h of the pod's disk
    per_gb: float = 0               # $/GB of the weight download
    note: str = ""
    hourly_max: float = 0           # vast.ai: the dearest offer `up` may fall back to
    per_gb_max: float = 0

    @property
    def hourly(self) -> float:
        return self.gpu_h + self.disk_h

    def session(self, hours: float, download_gb: float | None) -> float:
        """$/h × hours plus one download of the weights (unknown size counts as 0)."""
        return self.hourly * hours + (download_gb or 0) * self.per_gb


QUOTE_WORKERS = 4                # parallel offer searches for `kami prices`


HOURS_PER_MONTH = 30 * 24        # how providers turn $/GB/month into $/h


def runpod_quotes(rp, cfg: Config, profiles, cloud: str) -> dict[str, list[Quote]]:
    """RunPod: the catalog's per-GPU price × gpu_count, plus the container disk.
    Ingress (the weight download) is free on RunPod."""
    gpus = {catalog.key(g["id"]): g for g in rp.list_gpus(cloud)}
    quotes = {}
    for p in profiles:
        disk = VOLUME_POD_DISK_GB if cfg.storage == "volume" else p.disk_gb
        rows = []
        for gpu in p.gpus:
            g = gpus.get(catalog.key(gpu), {})
            q = Quote(gpu, disk_h=disk * CONTAINER_DISK_PRICE_GB_MONTH / HOURS_PER_MONTH)
            price = (g.get("price") or {}).get(cloud.lower())
            stock = g.get("availability")
            dcs = [d["id"] for d in g.get("dataCenters", [])]
            if price is not None:
                q.gpu_h = price * p.gpu_count       # the list price, with stock or not
            if price is None or stock in (None, "NONE"):
                q.note = "no stock"
            elif cfg.datacenter and cfg.datacenter not in dcs:
                q.note = f"no stock in {cfg.datacenter}"
            else:
                q.stock = stock
            rows.append(q)
        quotes[p.name] = rows
    return quotes


def vast_quotes(rp, cfg: Config, profiles, cloud: str) -> dict[str, list[Quote]]:
    """vast.ai: the offer `up` would rent now (kami's filters, the profile's disk), with
    its storage and download prices. One search per distinct GPU/count/disk/CUDA."""
    asks = {}                       # (profile, gpu) -> filters, or the ConfigError text
    for p in profiles:
        for gpu in p.gpus:
            try:
                asks[(p.name, gpu)] = vast_offer(cfg, p, gpu, cloud)
            except ConfigError as e:
                asks[(p.name, gpu)] = str(e).splitlines()[0]
    disks = {p.name: p.disk_gb for p in profiles}
    searches = {json.dumps([f, disks[name]], sort_keys=True): (f, disks[name])
                for (name, _), f in asks.items() if isinstance(f, dict)}
    with ThreadPoolExecutor(QUOTE_WORKERS) as pool:
        found = dict(zip(searches, pool.map(lambda fd: rp.quote(*fd), searches.values())))
    quotes = {}
    for p in profiles:
        rows = []
        for gpu in p.gpus:
            ask = asks[(p.name, gpu)]
            if not isinstance(ask, dict):
                rows.append(Quote(gpu, note="not on vast.ai"))
                continue
            offer = found[json.dumps([ask, p.disk_gb], sort_keys=True)]
            if not offer:
                rows.append(Quote(gpu, note="no offers"))
                continue
            n = offer["offers"]
            rows.append(Quote(gpu, stock=f"{n}{'+' if n >= 50 else ''} offer{'s' if n > 1 else ''}",
                              gpu_h=offer["gpu"], disk_h=offer["disk"], per_gb=offer["download"],
                              hourly_max=offer["hourly_max"], per_gb_max=offer["download_max"]))
        quotes[p.name] = rows
    return quotes


def cmd_prices(args, cfg: Config) -> None:
    """Current price of each profile's GPUs on a provider: GPU, disk, the weight download
    and the total for a session of --hours."""
    provider = args.provider or cfg.provider
    cloud = "SECURE" if args.secure else cfg.cloud
    names = args.profiles or list(cfg.profiles)
    profiles = [cfg.profile(n) for n in names]
    rp = client(provider)
    quotes = (vast_quotes if provider == "vast" else runpod_quotes)(rp, cfg, profiles, cloud)
    cached = provider == "runpod" and cfg.storage == "volume"   # weights already on the volume
    sizes = {}
    for p in profiles:
        try:
            sizes[p.name] = 0.0 if cached else fit.download_gb(p)
        except LookupError:
            sizes[p.name] = None

    hours = args.hours
    head = (f"{'profile':<16}  {'GPU':<28}  {'GPU/h':>7}  {'disk/h':>6}  {'$/h':>7}  "
            f"{'download':>16}  {f'{hours:g} h total':>10}  stock")
    print(f"  {ui.bold('Prices for your profiles')} "
          f"{ui.dim(f'· {rp.label} · {cloud} cloud · now')}\n")
    print("  " + ui.dim(head))
    for p in profiles:
        picked = next((q for q in quotes[p.name] if q.stock), None)
        gb = sizes[p.name]
        for i, q in enumerate(quotes[p.name]):
            # pad before colouring: ANSI codes would count towards the width
            name = (p.name if i == 0 else "").ljust(16)
            mark = ui.SYM["pick"] + " " if q is picked else "  "
            gpu = (catalog.key(q.gpu) + (f" x{p.gpu_count}" if p.gpu_count > 1 else ""))[:26].ljust(26)
            if not q.gpu_h:                  # not even a price: nothing to show
                print(f"  {ui.bold(name)}  {mark}{ui.dim(gpu)}  {ui.dim(q.note)}")
                continue
            if gb is None:
                download = "? GB".rjust(16)
            elif cached:
                download = "on volume".rjust(16)
            else:
                cost = "free" if gb * q.per_gb < 0.005 else f"${gb * q.per_gb:.2f}"
                download = f"{cost} · {gb:.0f} GB".rjust(16)
            total = f"${q.session(hours, gb):.2f}" + ("+" if gb is None and q.per_gb else "")
            cells = (f"{gpu}  {'$' + format(q.gpu_h, '.2f'):>7}  {'$' + format(q.disk_h, '.3f'):>6}  "
                     f"{'$' + format(q.hourly, '.2f'):>7}  {download}  ")
            if q.stock:
                print(f"  {ui.bold(name)}  {mark}{cells}"
                      f"{ui.bold(total.rjust(10)) if q is picked else total.rjust(10)}  {ui.dim(q.stock)}")
            else:                             # priced, but cannot be had right now
                print(f"  {ui.bold(name)}  {mark}{ui.dim(cells + total.rjust(10))}  {ui.yellow(q.note)}")
    print()
    ui.hint(f"{ui.SYM['pick']} the GPU `kami up` takes: the first in the profile's order with stock")
    ui.hint(f"Total = $/h × {hours:g} h (--hours) + one download of the weights; the "
            "download minutes also bill at $/h")
    state = load_state()
    boots = [f"{p.name} ~{ui.duration(t[0])}" for p in profiles
             if (t := typical_boot(state, p.name, provider, cfg.storage))]
    if boots:
        ui.hint("Typical boot (your history; it bills at $/h too): " + ", ".join(boots))
    if provider == "vast":
        ui.hint("vast.ai: the offer `up` would rent now; the host bills the download per GB")
    elif cached:
        ui.hint(f"Weights on the network volume: no download, but the volume bills "
                f"≈ {monthly(cfg.volume_size_gb)} whether a pod runs or not")
    else:
        ui.hint("RunPod: container disk ≈ $0.10/GB/month while running; downloads are free")


def cmd_up(args, cfg: Config) -> None:
    profile = cfg.profile(args.profile)
    provider = getattr(args, "provider", None) or cfg.provider
    config.check_provider(provider, cfg.storage)
    if provider == "vast":
        # kami.toml was checked for its own provider: with --provider vast, check the
        # GPU names now, before anything is created
        for gpu in profile.gpus:
            catalog.provider_gpu(gpu, "vast")
    rp = client(provider)
    state = load_state()

    waiting = waiting_for_stock(state)
    if waiting:
        # two waiting `up`s could each create a pod, and both would bill
        ui.warn("Another " + waiting_text(waiting))
        ui.hint("Wait for it, or Ctrl+C it in its terminal first.")
        return

    if state.get("pod_id"):
        # the registered pod may be on another provider than this one
        old_rp = client(pod_provider(state))
        try:
            pod = old_rp.get_pod(state["pod_id"])
        except ProviderError as e:
            if e.status != 404:
                raise
            clear_pod(state)
        else:
            if pod["status"] not in DEAD_STATES:
                ui.warn(f"A pod is already active: {pod['id']} ({pod['status']}) "
                        f"· profile {state.get('profile')} · {old_rp.label}")
                ui.hint("`kami status` to see it · `kami down` to delete it")
                return
            # A stopped pod still exists (and keeps its disk): delete it, do not orphan it.
            ui.step(f"Deleting the previous pod {pod['id']} ({pod['status']})")
            terminate(old_rp, state)

    problem = container_problem(cfg, profile)
    if problem:
        raise ConfigError("\n".join((*problem, "No pod was created.")))
    ui.step(f"Profile {ui.bold(profile.name)} {ui.dim('· ' + profile.model + ' · ' + rp.label)}")
    if args.skip_memory_check:
        candidates = [fit.Fit(g, None, 0) for g in profile.gpus]
    else:
        candidates = memory_check(rp, cfg, profile)
    problems = fit.access_problems(profile)
    if problems:
        raise ConfigError("\n".join((*problems, "No pod was created: the pod could not download "
                                                 "the weights.")))
    volume_id = ensure_volume(rp, cfg, state) if cfg.storage == "volume" else None
    api_key = secrets.token_urlsafe(32)
    max_hours = cfg.max_hours if args.max_hours is None else args.max_hours
    idle_minutes = cfg.idle_minutes if args.idle is None else args.idle
    deadline = time.time() + max_hours * 3600 if max_hours > 0 else 0
    clouds = [cfg.cloud] + (["SECURE"] if args.secure and cfg.cloud != "SECURE" else [])
    budget = budget_plan(rp, cfg, profile, candidates, clouds, args.budget) if args.budget else None
    if budget:
        candidates = [c for c in candidates if any((c.gpu, cl) in budget for cl in clouds)]

    def guard_deadline(gpu: str, cloud: str) -> float:
        """max_hours, or sooner when --budget runs out first on this GPU."""
        if not budget or (gpu, cloud) not in budget:
            return deadline
        end = time.time() + budget[(gpu, cloud)]["hours"] * 3600
        return min(deadline, end) if deadline else end
    idle_seconds = max(0, idle_minutes) * 60
    # The balance before the pod exists: `kami down` subtracts the balance then, which
    # gives the real cost at once (per-pod billing is published about an hour later).
    try:
        balance_at_up = rp.account()["balance"]
    except ProviderError:
        balance_at_up = None

    waiting_since = time.time()

    def heartbeat() -> None:
        # visible to `kami status` / `down` / another `up` in other terminals; re-read
        # so a long wait never writes back an old copy of the state
        fresh = load_state()
        fresh["waiting"] = {"profile": profile.name, "clouds": clouds,
                            "since": waiting_since, "heartbeat": time.time()}
        save_state(fresh)

    build = make_body(provider)
    deadlines = {}                  # (gpu, cloud) -> the deadline its pod body carries

    def body(gpu: str, cloud: str) -> dict:
        deadlines[(gpu, cloud)] = guard_deadline(gpu, cloud)
        return build(cfg, profile, gpu, volume_id, api_key, deadlines[(gpu, cloud)],
                     idle_seconds, cloud)

    try:
        pod, candidate, cloud = create_with_stock(
            rp, cfg, candidates, clouds, body,
            max_wait=(args.max_wait or 0) * 60, on_wait=heartbeat,
            poll=max(MIN_STOCK_POLL, args.poll) if args.poll else STOCK_POLL)
    finally:
        state = load_state()
        if state.pop("waiting", None):
            save_state(state)

    deadline = deadlines.get((candidate.gpu, cloud), deadline)
    state.update(pod_id=pod["id"], profile=profile.name, api_key=api_key, model=profile.model,
                 deadline=deadline, idle_minutes=idle_seconds / 60,
                 created_at=time.time(), balance_at_up=balance_at_up, provider=provider)
    remember_pod(state, pod, profile.name, provider)
    save_state(state)
    ui.ok(f"Pod {pod['id']} created {ui.dim('·')} {pod.get('gpu', {}).get('id', '?')} "
          f"{ui.dim('·')} {hourly_text(pod)} {ui.dim('· ' + cloud.lower() + ' cloud')}")
    if pod.get("where"):
        ui.step(f"Host in {pod['where']}")
    typical = typical_boot(state, profile.name, provider, cfg.storage, (pod.get("gpu") or {}).get("id"))
    if typical:
        seconds, count = typical
        extra = f"median of your last {count}" if count > 1 else "your last boot"
        if isinstance(pod.get("cost"), (int, float)):
            extra += f" · ≈ ${pod['cost'] * seconds / 3600:.2f} at this rate"
        ui.step(f"Typical boot: ~{ui.duration(seconds)} {ui.dim('· ' + extra)}")
    download = download_text(pod, profile)
    if download:
        ui.step(download)
    if cloud != cfg.cloud:
        ui.warn(f"No {cfg.cloud.lower()} stock: this pod is on {cloud.lower()} cloud, "
                "which costs more per hour (allowed by --secure)")
    ui.hint("Live detail: `kami logs` in another terminal")
    plan = (budget or {}).get((candidate.gpu, cloud))
    if plan:
        rate = f"at up to ${plan['hourly']:.2f}/h"
        if plan["download"] >= 0.005:
            rate += f" + ${plan['download']:.2f} download"
        ui.step(f"Budget ${args.budget:.2f}: ≈ {ui.duration(plan['hours'] * 3600)} {ui.dim(rate)}")
    guard = guard_text(state)
    if guard:
        ui.step(f"Guard: {guard}")
    else:
        ui.warn("No guard: the pod bills until `kami down` (max_hours / idle_minutes are 0)")
    print()

    url = None                      # known once the pod runs (vast.ai publishes the port then)
    logs = LogStream(rp, pod["id"], tail=5000, reconnect=True)
    recent: deque[str] = deque(maxlen=300)
    starts, fatal_at = 0, None
    running, last_poll = False, 0.0
    snapshot, last_snapshot = None, 0.0
    progress = ui.Progress(f"Waiting for {rp.label} to assign the machine")
    try:
        download_bytes = fit.download_gb(profile) * 1e9
    except LookupError:
        download_bytes = None
    report = BootReport(progress, download_bytes, gpu_bar=False,
                        hourly=pod.get("cost") if isinstance(pod.get("cost"), (int, float)) else None)

    def no_gpu(line: str) -> None:
        if any(m in line for m in NO_GPU_MARKERS):
            progress.stop()
            abort_up(rp, pod["id"], recent, keep=args.keep_on_failure, prefer_reason=True,
                     reason=f"the host's GPU is not usable ({line.strip()[-80:]}). "
                            "A broken host, not your profile: `kami up` again gets another one")

    try:
        # Short timeouts: on Windows Ctrl+C waits for a blocking socket read to finish.
        while not (running and url and healthy(url, timeout=3)):
            now = time.monotonic()
            if (not running or not url) and now - last_poll >= 5:
                last_poll = now
                try:
                    fresh = rp.get_pod(pod["id"], timeout=10)
                except ProviderError as e:
                    if not e.transient:
                        raise
                    fresh = None  # network blip: keep waiting, check again later
                status = fresh["status"] if fresh else None
                if status == "RUNNING":
                    running = True
                    if not url:
                        url = endpoint(fresh, cfg.port)
                        if url:
                            record_url(pod["id"], url)
                elif status in DEAD_STATES:
                    progress.stop()
                    abort_up(rp, pod["id"], recent, keep=args.keep_on_failure,
                             reason=f"the pod ended in state {status}")
                elif progress.elapsed > POD_READY_TIMEOUT[provider]:
                    raise ConfigError("The pod is taking too long to come up\n"
                                      "Check `kami status`; `kami down` to delete it.")
            for line in logs.drain():
                if report.line(line):
                    continue
                recent.append(line)
                if line.startswith(CONTAINER_START):
                    starts += 1
                if fatal_at is None and any(m in line for m in FATAL_MARKERS):
                    fatal_at = time.monotonic()
                no_gpu(line)
                for marker, phase in PHASES:
                    if marker in line:
                        progress.phase(phase)
            # RunPod's long-lived stream can bring the container's lines minutes late; a fresh
            # snapshot every SNAPSHOT_EVERY s catches a dead GPU or a fatal error early.
            # Only those markers (and the guard's reports, which are timestamped) are read
            # from it: it repeats lines the stream already gave.
            # (vast.ai's logs are polled, always fresh: no snapshots.)
            if snapshot and snapshot.done:
                for line in snapshot.lines:
                    if report.line(line):
                        continue
                    if line not in recent:
                        recent.append(line)
                    if fatal_at is None and any(m in line for m in FATAL_MARKERS):
                        fatal_at = time.monotonic()
                    no_gpu(line)
                snapshot = None
            if running and snapshot is None and provider == "runpod" \
                    and now - last_snapshot >= SNAPSHOT_EVERY:
                last_snapshot = now
                snapshot = LogSnapshot(rp, pod["id"])
            # the traceback with the cause arrives after the marker: allow some grace
            if starts > 1 or (fatal_at and time.monotonic() - fatal_at > FATAL_GRACE):
                progress.stop()
                abort_up(rp, pod["id"], recent, keep=args.keep_on_failure)
            if progress.elapsed > MODEL_READY_TIMEOUT:
                raise ConfigError("The server did not get ready in time\n"
                                  "The pod is still up: `kami logs` to investigate, `kami down` to stop it.")
            progress.tick()
            time.sleep(1)
        if url.startswith("http://"):
            # vast.ai: the guard opens an HTTPS tunnel; use it so the key is not sent in clear
            progress.phase("Opening the HTTPS tunnel")
            secure = wait_for_tunnel(logs, recent, progress)
            if secure:
                url = secure
                record_url(pod["id"], url)
    finally:
        progress.stop()

    report.summary()
    ui.ok(f"Server ready in {ui.duration(progress.elapsed)}")
    record_boot(pod["id"], boot_seconds=round(progress.elapsed), storage=cfg.storage,
                download_bytes=report.got or None,
                download_seconds=(report.span[1] - report.span[0]) if report.span else None)
    try:
        served = jobs.info(url, api_key)
        what = [f"mesh {served['mesh_model']}" if served.get("mesh_model") else "",
                f"image {served['image_model']}" if served.get("image_model") else ""]
        ui.ok(f"Serving {' · '.join(w for w in what if w)}")
    except ConfigError as e:
        ui.warn(f"The server answers /health but not /v1/info: {e}")
    if url.startswith("http://"):
        ui.warn("Plain HTTP: the API key, your images and prompts travel unencrypted to the host")
        ui.hint("The HTTPS tunnel did not come up (`kami logs` shows why). "
                "`kami down` if that is not acceptable.")
    print()
    mascot.show("ready", [
        "Ready. The spirit takes form.",
        f"Models: {profile.model}",
        *ready_hints(profile),
    ])


def ready_hints(profile) -> list[str]:
    hints = []
    if profile.mesh_model:
        hints.append("`kami mesh photo.png` to turn an image into a 3D mesh")
    if profile.image_model:
        hints.append('`kami image "a prompt"` to make an image')
    if profile.mesh_model and profile.image_model:
        hints.append('`kami mesh --prompt "a prompt"` for text -> image -> mesh')
    return hints


def record_url(pod_id: str, url: str) -> None:
    """Save the pod's URL, unless `kami down` (and maybe a new `up`) ran meanwhile."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        state["url"] = url
        save_state(state)


def tunnel_url(lines) -> str | None:
    for line in reversed(lines):
        match = TUNNEL_LINE.search(line)
        if match:
            return match.group(1)
    return None


def wait_for_tunnel(logs: LogStream, recent, progress) -> str | None:
    """The guard's HTTPS tunnel URL once it answers, or None after TUNNEL_WAIT."""
    started = time.monotonic()
    while time.monotonic() - started < TUNNEL_WAIT:
        recent.extend(logs.drain())
        url = tunnel_url(recent)
        if url and healthy(url, timeout=5):
            return url
        progress.tick()
        time.sleep(1)
    return None


STOCK_POLL = 30                  # seconds between stock checks while waiting for a GPU
MIN_STOCK_POLL = 5               # floor for --poll, to avoid hammering the provider's API


def is_no_stock(e: ProviderError) -> bool:
    """The provider's own reading: out of stock (worth waiting) vs. a refused request."""
    return bool(getattr(e, "no_stock", False))


def waiting_for_stock(state: dict) -> dict | None:
    """The `kami up` (maybe in another terminal) waiting for stock, if still alive: it
    refreshes a heartbeat every round, so a stale one means that process is gone."""
    w = state.get("waiting")
    if w and time.time() - w.get("heartbeat", 0) <= 2 * STOCK_POLL + 15:
        return w
    return None


def waiting_text(w: dict) -> str:
    return (f"`kami up -p {w['profile']}` is waiting for {'/'.join(c.lower() for c in w['clouds'])} "
            f"stock ({ui.duration(time.time() - w['since'])} so far) · nothing is billed yet")


def create_with_stock(rp, cfg: Config, candidates, clouds: list[str], make_body,
                      max_wait: float = 0, sleep=time.sleep, on_wait=None,
                      poll: int = STOCK_POLL):
    """Create the pod on the first cloud × GPU with stock, in order. When every attempt
    fails for lack of stock, wait and try again -- nothing is billed while waiting --
    until `max_wait` seconds have passed (0 = until Ctrl+C). Returns (pod, candidate, cloud).

    Errors that are not about stock (an invalid request) are shown once and that GPU is
    still retried with the rest, unless no attempt failed for stock: then waiting is pointless."""
    where = rp.label if getattr(rp, "name", None) == "vast" else (cfg.datacenter or "any datacenter")
    started, shown, progress = time.monotonic(), set(), None
    try:
        while True:
            errors = []
            for cloud in clouds:
                for candidate in candidates:
                    try:
                        pod = rp.create_pod(make_body(candidate.gpu, cloud))
                        return pod, candidate, cloud
                    except ProviderError as e:
                        if e.status in (401, 403):
                            raise
                        errors.append(e)
                        # the provider's own reason: "no stock" and "invalid request" look different
                        key = (cloud, candidate.gpu, e.detail)
                        if key not in shown and not is_no_stock(e):
                            shown.add(key)
                            if progress:
                                progress.stop()
                            ui.warn(f"{candidate.gpu} ({cloud.lower()}) in {where}: {e.detail}")
            # "no stock" may come back as a 5xx: it is not the provider being down
            name = errors[0].provider
            if all(e.transient and not is_no_stock(e) for e in errors):
                raise ConfigError(f"Could not create the pod: {name} is not responding\n"
                                  "Try again in a moment.")
            if not any(is_no_stock(e) for e in errors):
                raise ConfigError("Could not create a pod with any GPU in the profile\n"
                                  f"{name} refused the request (see above); waiting would not help.")
            waited = time.monotonic() - started
            if max_wait and waited >= max_wait:
                raise ConfigError(f"No stock for profile's GPUs after {ui.duration(waited)}\n"
                                  "`kami gpus` shows current stock; add fallbacks to the profile's "
                                  "`gpus`, or allow secure cloud with `--secure`.")
            if progress is None:
                gpus = ", ".join(c.gpu for c in candidates)
                ui.warn(f"No {'/'.join(c.lower() for c in clouds)} stock right now for: {gpus}")
                hint = "Ctrl+C to cancel"
                if "SECURE" not in clouds:
                    hint += " · `--secure` also tries secure cloud (pricier)"
                ui.hint(f"Nothing is billed while waiting · {hint}")
                progress = ui.Progress(f"Waiting for stock (checking every {poll} s)")
            if on_wait:
                on_wait()
            for _ in range(poll):
                progress.tick()
                sleep(1)
    finally:
        if progress:
            progress.stop()


def abort_up(rp, pod_id: str, recent, keep: bool, reason: str | None = None,
             prefer_reason: bool = False) -> None:
    """The server will not start: save the log, delete the pod (unless keep) and raise."""
    failure_log = config.project.state_dir / "last-failure.log"
    config.project.state_dir.mkdir(exist_ok=True)
    failure_log.write_text("\n".join(recent) + "\n", encoding="utf-8")
    if prefer_reason and reason:
        cause = reason
    else:
        cause = root_cause(recent) or reason or "cause not identified"
    mascot.show("fail", ["The spirit would not wake.", "The server could not start."])
    print()
    msg = [f"The server could not start: {cause}",
           f"Log saved to {failure_log}"]
    if keep:
        msg.append("The pod is still up (--keep-on-failure): `kami logs` to investigate, `kami down` when done.")
    else:
        terminate_own(rp, pod_id)
        msg.append(f"Pod {pod_id} deleted so the GPU stops billing.")
    raise ConfigError("\n".join(msg))


def hourly_text(pod: dict) -> str:
    """'$1.72/h (GPU $1.69 + disk $0.03)': what the pod bills per hour, all in."""
    cost, gpu = pod.get("cost"), pod.get("gpu_cost")
    if not isinstance(cost, (int, float)):
        return "$?/h"
    text = f"${cost:.2f}/h"
    if isinstance(gpu, (int, float)) and cost - gpu >= 0.005:
        text += ui.dim(f" (GPU ${gpu:.2f} + disk ${cost - gpu:.2f})")
    return text


def download_text(pod: dict, profile) -> str | None:
    """What the host bills once for the weight download (vast.ai), or None if it is free."""
    per_gb = pod.get("download_cost_per_gb") or 0
    if per_gb <= 0:
        return None
    try:
        gb = fit.download_gb(profile)
    except LookupError:
        return f"Model download billed by the host at ${per_gb * 1000:.2f}/TB (size unknown)"
    return (f"Model download ≈ ${gb * per_gb:.2f} once "
            f"{ui.dim(f'· {gb:.0f} GB at ${per_gb * 1000:.2f}/TB, billed by the host')}")


MIN_BUDGET_HOURS = 0.25        # a --budget that buys less than this on a GPU skips the GPU


def budget_plan(rp, cfg: Config, profile, candidates, clouds: list[str],
                budget: float) -> dict[tuple[str, str], dict]:
    """--budget: how long each GPU × cloud can run for `budget` dollars, priced now with
    the disk and (vast.ai) the weight download -- and on vast.ai at the dearest offer
    `up` may fall back to, so the budget holds whichever it rents. {(gpu, cloud):
    {'hours', 'hourly', 'download'}}; GPUs without a price, or that `budget` cannot
    keep up for MIN_BUDGET_HOURS, are left out (ConfigError if none is left)."""
    if budget <= 0:
        raise ConfigError("--budget must be more than $0")
    quote = vast_quotes if rp.name == "vast" else runpod_quotes
    try:
        gb = 0.0 if rp.name == "runpod" else fit.download_gb(profile)
    except LookupError as e:
        ui.warn(f"Weight download size unknown ({e}): the budget leaves the download out")
        gb = 0.0
    plan, skipped = {}, []
    for cloud in clouds:
        for q in quote(rp, cfg, [profile], cloud)[profile.name]:
            if not q.gpu_h:
                skipped.append(f"{q.gpu} ({cloud.lower()}): no price")
                continue
            hourly = max(q.hourly, q.hourly_max)
            download = gb * max(q.per_gb, q.per_gb_max)
            hours = (budget - download) / hourly
            if hours < MIN_BUDGET_HOURS:
                skipped.append(f"{q.gpu} ({cloud.lower()}): ${budget:.2f} lasts "
                               f"{ui.duration(max(0, hours) * 3600)} at ${hourly:.2f}/h")
                continue
            plan[(q.gpu, cloud)] = {"hours": hours, "hourly": hourly, "download": download}
    wanted = {c.gpu for c in candidates}
    if not any(gpu in wanted for gpu, _ in plan):
        raise ConfigError(f"--budget ${budget:.2f} is not enough for any GPU of the profile\n"
                          + "\n".join(skipped))
    for line in skipped:
        ui.warn(f"Skipped by the budget: {line}")
    return plan


def estimated_cost(state: dict) -> float | None:
    """$/h of the pod in the history × time since its creation (None if unknown)."""
    entry = next((h for h in reversed(state.get("history", []))
                  if h.get("pod_id") == state.get("pod_id")), None)
    rate, since = (entry or {}).get("cost_per_hour"), state.get("created_at")
    if not isinstance(rate, (int, float)) or not since:
        return None
    return rate * (time.time() - since) / 3600


def cmd_down(args, cfg: Config) -> None:
    state = load_state()
    if not state.get("pod_id"):
        waiting = waiting_for_stock(state)
        if waiting:
            ui.warn("No pod yet, but " + waiting_text(waiting))
            ui.hint("Ctrl+C it in its terminal to cancel; it creates the pod as soon as there is stock.")
            return
        mascot.show("rest", ["No pod registered.", "The spirit already rests."])
        return
    rp = client(pod_provider(state))
    try:
        session = session_cost(rp.get_pod(state["pod_id"]), since=state.get("created_at"))
    except ProviderError:
        session = None
    before = dict(state)            # terminate() clears the pod keys, balance included
    pod_id = terminate(rp, state)
    detail = (f"The weights remain on the network volume (≈ {monthly(cfg.volume_size_gb)})."
              if cfg.storage == "volume" and rp.name == "runpod"
              else "The downloaded weights go with it.")
    lines = [f"Pod {pod_id} deleted.", detail,
             *down_cost_lines(session, real_session_cost(rp, before), estimated_cost(before), rp.label)]
    lines.append("The spirit returns to the paper.")
    mascot.show("down", lines)


def down_cost_lines(session: str | None, real: float | None, estimate: float | None,
                    label: str) -> list[str]:
    """The cost lines of `kami down`. The provider debits the balance in batches (RunPod:
    every ~15 min, the last one after the pod is gone), so right after a delete the balance
    drop trails the time billed -- $0.00 for a short session. It is shown as the cost only
    once it has caught up with the estimate; until then the estimate is the figure."""
    lines = []
    if session:
        lines.append(ui.dim(f"Session: {session}"))
    elif estimate is not None:
        lines.append(ui.dim(f"Session: ≈ ${estimate:.2f} (estimate)"))
    if real is None:
        return lines
    if real >= 0.005 and (estimate is None or real >= estimate * 0.9):
        lines.append(f"Real cost: ${real:.2f} {ui.dim('(account balance since `kami up`)')}")
    elif estimate is not None:
        debited = f"has debited ${real:.2f} of it so far" if real >= 0.005 else "has not debited it yet"
        lines.append(ui.dim(f"{label} {debited} (it bills in batches); "
                            "`kami cost` has the final figure in about an hour."))
    return lines


STATUS_READY, STATUS_NOT_READY, STATUS_NO_POD = 0, 2, 3   # `kami status` exit codes


def cmd_status(args, cfg: Config) -> int:
    state = load_state()
    info = status_info(cfg, state)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        show_status(cfg, state, info)
    if not info["pod"]:
        # waiting for stock is "coming, not ready", not "nothing there"
        return STATUS_NOT_READY if info.get("waiting") else STATUS_NO_POD
    return STATUS_READY if info["ready"] else STATUS_NOT_READY


def status_info(cfg: Config, state: dict) -> dict:
    """Everything `kami status` knows, as plain data (also its --json output)."""
    provider = pod_provider(state) if state.get("pod_id") else cfg.provider
    info = {"pod": None, "ready": False, "storage": cfg.storage, "provider": provider,
            "volume_id": state.get("volume_id") if cfg.storage == "volume" else None,
            "account": None}
    waiting = waiting_for_stock(state)
    info["waiting"] = ({"profile": waiting["profile"], "clouds": waiting["clouds"],
                        "since": waiting["since"]} if waiting else None)
    try:
        rp = client(provider)
        info["account"] = rp.account()      # nice to have: never a reason for status to fail
    except (ConfigError, ProviderError):
        rp = None
    info["others"] = [{"id": p["id"], "name": p.get("name"), "status": p.get("status"),
                       "cost_per_hour": p.get("cost"), "provider": other.name}
                      for other, p in orphans(state)]
    if not state.get("pod_id"):
        return info
    rp = rp or client(provider)
    try:
        pod = rp.get_pod(state["pod_id"])
    except ProviderError as e:
        if e.status != 404:
            raise
        info["pod"] = {"id": state["pod_id"], "status": "GONE"}
        return info
    uptime = (pod.get("runtime") or {}).get("uptime")
    cost = pod.get("cost")
    # billed time runs from creation (image pull and download included), not container uptime
    billed = time.time() - state["created_at"] if state.get("created_at") else uptime
    url = state.get("url") or endpoint(pod, cfg.port)
    info["ready"] = bool(pod["status"] == "RUNNING" and url and healthy(url, timeout=5))
    usage = server_usage(server_metrics(url, state.get("api_key", ""))) if info["ready"] else None
    info["pod"] = {
        "id": pod["id"], "status": pod["status"], "url": url,
        "profile": state.get("profile"), "model": state.get("model"),
        "gpu": pod.get("gpu", {}).get("id"), "cost_per_hour": cost,
        "uptime_seconds": uptime,
        "created_at": state.get("created_at"),
        "session_cost_estimate": round(cost * billed / 3600, 2)
        if isinstance(cost, (int, float)) and billed is not None else None,
        "session_cost_real": round(real, 2)
        if (real := real_session_cost(rp, state)) is not None else None,
        # kept for scripts that read it; the estimate
        "session_cost": round(cost * billed / 3600, 2)
        if isinstance(cost, (int, float)) and billed is not None else None,
        "guard": {"deadline": state.get("deadline") or None,
                  "idle_minutes": state.get("idle_minutes") or None},
        "usage": usage,
    }
    return info


def usage_lines(usage: dict | None) -> list[str]:
    """`kami status`'s Jobs line from pod.server_usage (nothing without it)."""
    if not usage:
        return []
    if not any(usage.values()):
        return [ui.field("Jobs", ui.dim("none yet"))]
    parts = [f"{usage['finished']} done"]
    if usage["failed"]:
        parts.append(ui.yellow(f"{usage['failed']} failed"))
    if usage["running"]:
        parts.append(f"{usage['running']} running")
    if usage["queued"]:
        parts.append(f"{usage['queued']} queued")
    return [ui.field("Jobs", f" {ui.dim('·')} ".join(parts))]


def show_status(cfg: Config, state: dict, info: dict) -> None:
    storage = (f"volume {state['volume_id']} ({cfg.datacenter}) "
               f"{ui.dim('· ' + str(cfg.volume_size_gb) + ' GB · ≈ ' + monthly(cfg.volume_size_gb))}"
               if cfg.storage == "volume" and state.get("volume_id") else cfg.storage)
    pod = info["pod"]
    account = info.get("account")
    account_line = (ui.field("Account", f"${account['balance']:.2f} left {ui.dim('·')} "
                                        f"spending ${account['spend_per_hour']:.2f}/h now")
                    if account else None)
    warnings = [*runway_warning(account, state if pod else {}), *others_lines(info.get("others"))]
    if not pod:
        waiting = info.get("waiting")
        lines = [ui.field("Pod", ui.yellow("waiting for stock") if waiting else ui.dim("none")),
                 ui.field("Storage", storage)]
        if account_line:
            lines.append(account_line)
        lines += warnings
        if waiting:
            mascot.show("wait", [*lines, "", waiting_text(waiting),
                                 "Ctrl+C it in its terminal to cancel."], top=True)
        else:
            mascot.show("rest", [*lines, "", "The spirit rests. `kami up` to wake it."], top=True)
        return
    label = PROVIDERS[info.get("provider") or "runpod"][0].label
    if pod["status"] == "GONE":
        mascot.show("rest", [
            ui.field("Pod", f"{pod['id']} {ui.yellow('no longer exists on ' + label)}"),
            "",
            "`kami down` to clear the local state.",
        ], top=True)
        return
    ok = info["ready"]
    status_color = ui.green if pod["status"] == "RUNNING" else ui.yellow
    cost = "?" if pod["cost_per_hour"] is None else f"{pod['cost_per_hour']:.2f}"
    lines = [
        ui.field("Pod", f"{pod['id']} {status_color(pod['status'])}"),
        ui.field("Profile", f"{pod['profile']} {ui.dim('· ' + str(pod['model']))}"),
        ui.field("GPU", f"{pod['gpu'] or '?'} {ui.dim('·')} ${cost}/h {ui.dim('· ' + label)}"),
    ]
    uptime = pod["uptime_seconds"]
    session = session_cost({"runtime": {"uptime": uptime}, "cost": pod["cost_per_hour"]},
                           since=pod.get("created_at"))
    if session:
        if pod.get("session_cost_real") is not None:
            # the balance is debited in batches, so this trails the estimate
            session += f" {ui.dim('·')} debited so far ${pod['session_cost_real']:.2f}"
        lines.append(ui.field("Session", session))
    guard = guard_text(state)
    lines.append(ui.field("Guard", guard if guard else ui.yellow("none · bills until `kami down`")))
    lines += [
        ui.field("URL", ui.dim(pod["url"] or "not published yet")
                 + (ui.yellow(" · plain HTTP") if (pod["url"] or "").startswith("http://") else "")),
        ui.field("Server", ui.green("ready") if ok else ui.yellow("not responding yet")),
    ]
    lines += usage_lines(pod.get("usage"))
    lines += [
        ui.field("Storage", storage),
    ]
    if account_line:
        lines.append(account_line)
    lines += warnings
    if uptime and uptime > WARN_AFTER_HOURS * 3600:
        lines += ["", ui.yellow(f"! Running for {ui.duration(uptime)} and still billing — "
                                "`kami down` if you are done.")]
    mascot.show("ready" if ok else "wait", lines, top=True)


RUNWAY_WARN_HOURS = 2         # warn when the balance lasts less than this at the current spend


def runway_hours(balance: float, per_hour: float) -> float | None:
    """Hours the balance lasts at `per_hour` (None when nothing is billing)."""
    return max(0.0, balance) / per_hour if per_hour > 0 else None


def runway_warning(account: dict | None, state: dict) -> list[str]:
    """A warning when the balance runs out soon, or before the guard's deadline: the
    provider then stops the pod mid-session (and its disk may go on billing)."""
    if not account:
        return []
    hours = runway_hours(account["balance"], account["spend_per_hour"])
    if hours is None:
        return []
    deadline = state.get("deadline") or 0
    left = (deadline - time.time()) / 3600 if deadline else None
    if hours >= RUNWAY_WARN_HOURS and (left is None or hours >= left):
        return []
    return ["", ui.yellow(f"! The balance lasts ≈ {ui.duration(hours * 3600)} at this spend"
                          + (f", before the guard's deadline (in {ui.duration(left * 3600)})"
                             if left is not None and hours < left else "")
                          + ": add credit, or the provider stops the pod.")]


def others_lines(others: list[dict] | None) -> list[str]:
    """Pods named kami-* this project has not registered: they bill unseen."""
    if not others:
        return []
    lines = ["", ui.yellow(f"! {len(others)} other kami pod{'s' if len(others) > 1 else ''} "
                           "on the account, not registered here:")]
    for o in others:
        label = PROVIDERS[o["provider"]][0].label
        cost = o.get("cost_per_hour")
        rate = f"${cost:.2f}/h" if isinstance(cost, (int, float)) else "$?/h"
        lines.append(f"  {o['id']}  {o.get('name') or '-'}  {o.get('status') or '?'}  "
                     f"{rate}  {ui.dim(label)}")
    lines.append(ui.dim("Another project's, or left over by a crashed `up`: delete it in the "
                        "provider's console if it is not in use."))
    return lines


def billed_by_pod(rows: list[dict]) -> dict[str, dict]:
    """Billing rows summed per pod: {pod_id: {'amount', 'minutes', 'first'}}."""
    pods: dict[str, dict] = {}
    for row in rows:
        p = pods.setdefault(row["podId"], {"amount": 0.0, "minutes": 0.0, "first": row["time"]})
        p["amount"] += row.get("amount") or 0
        p["minutes"] += (row.get("timeBilledMs") or 0) / 60000
        p["first"] = min(p["first"], row["time"])
    return pods


def cmd_cost(args, cfg: Config) -> None:
    """What the provider actually charged, per pod, from its billing API (RunPod), plus
    estimates for the pods its billing does not show."""
    from datetime import datetime, timedelta, timezone
    provider = args.provider or cfg.provider
    rp = client(provider)
    state = load_state()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days - 1)).strftime("%Y-%m-%dT00:00:00Z")
    rows = rp.pod_billing(since, bucket="day")
    pods = billed_by_pod(rows or [])
    history = {h["pod_id"]: h for h in state.get("history", [])
               if (h.get("provider") or "runpod") == provider}
    if rows is None:
        print(f"  {ui.bold(f'Pods kami created on {rp.label} since {since[:10]}')} "
              f"{ui.dim('(estimates: per-pod billing is not read from ' + rp.label + ')')}\n")
    else:
        print(f"  {ui.bold(f'Billed by {rp.label} since {since[:10]}')} {ui.dim('(UTC · real amounts)')}\n")
    if pods:
        print("  " + ui.dim(f"{'pod':<16}  {'profile':<16}  {'day':<10}  {'billed':>8}  {'amount':>8}"))
        for pod_id, p in sorted(pods.items(), key=lambda kv: kv[1]["first"]):
            # pad before colouring: ANSI codes would count towards the width
            name = history.get(pod_id, {}).get("profile")
            profile = name.ljust(16) if name else ui.dim("-".ljust(16))
            print(f"  {pod_id:<16}  {profile}  {p['first'][:10]:<10}  "
                  f"{ui.duration(p['minutes'] * 60):>8}  {'$' + format(p['amount'], '.2f'):>8}")
        print(f"\n  Total billed: {ui.bold(f'${sum(p['amount'] for p in pods.values()):.2f}')}")
    elif rows is not None:
        ui.step("Nothing billed in this period yet.")
    # pods kami created that billing does not show yet (it is published ~1 h late)
    cutoff = datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp()
    pending = [h for h in history.values()
               if h.get("created_at", 0) >= cutoff and h["pod_id"] not in pods]
    if rows is None and not pending:
        ui.step("No pods in this period.")
    elif pending and rows is None:
        ui.step(f"{len(pending)} pod(s):")
    elif pending:
        print()
        ui.warn(f"{len(pending)} pod(s) not in {rp.label}'s billing yet (it is published about an hour late):")
        for h in pending:
            end = h.get("deleted_at") or time.time()
            secs = end - h["created_at"]
            rate = h.get("cost_per_hour")
            est = f"≈ ${rate * secs / 3600:.2f} (estimate)" if isinstance(rate, (int, float)) else "?"
            running = "" if h.get("deleted_at") else ui.yellow(" · still registered")
            print(f"    {h['pod_id']:<16}  {h['profile']:<16}  {ui.duration(secs):>8}  {est}{running}")
    try:
        account = rp.account()
        print()
        ui.step(f"Account: ${account['balance']:.2f} left · spending ${account['spend_per_hour']:.2f}/h now")
    except ProviderError as e:
        ui.warn(f"Account balance unavailable: {e.detail}")


def cmd_logs(args, cfg: Config) -> None:
    state = load_state()
    if not state.get("pod_id"):
        ui.warn("No pod registered.")
        ui.hint("`kami up` to start one.")
        return
    if args.no_follow:
        ui.step(f"Last {args.tail} log lines of pod {state['pod_id']}")
        print()
        logs = LogStream(client(pod_provider(state)), state["pod_id"], tail=args.tail, reconnect=False)
        # the SSE stream stays open, so stop once the backfill goes quiet
        while True:
            try:
                item = logs.lines.get(timeout=3.0)
            except queue.Empty:
                return
            if item is LogStream.DONE:
                return
            if isinstance(item, Exception):
                raise item
            if args.verbose or not GUARD_PROGRESS.search(item):
                print(item, flush=True)
    ui.step(f"Logs of pod {state['pod_id']} {ui.dim('· Ctrl+C to exit')}")
    print()
    logs = LogStream(client(pod_provider(state)), state["pod_id"], tail=args.tail)
    while True:
        try:
            item = logs.lines.get(timeout=0.5)  # short timeout: let Ctrl+C through
        except queue.Empty:
            continue
        if item is LogStream.DONE:
            return
        if isinstance(item, Exception):
            raise item
        if args.verbose or not GUARD_PROGRESS.search(item):
            print(item, flush=True)


def ensure_pod(args, cfg: Config) -> dict:
    """The state of a pod that can run jobs, bringing one up first with --up. Raises if
    there is no usable pod."""
    state = load_state()
    if state.get("pod_id"):
        problem = pod_problem(cfg, state)
        if problem and args.up and problem.startswith("gone"):
            state = load_state()            # pod_problem cleared it: bring a new one up
        elif problem:
            hint = "`kami up` to start a new one" if problem.startswith("gone") \
                else "`kami status` · `kami logs`"
            raise ConfigError(f"The pod is {problem}\n{hint}")
    if not state.get("pod_id"):
        if not args.up:
            raise ConfigError("No pod is active\n`kami up` first, or pass --up.")
        cmd_up(argparse.Namespace(profile=args.profile, keep_on_failure=False, budget=None,
                                  max_hours=None, idle=None, skip_memory_check=False,
                                  secure=args.secure, max_wait=args.max_wait,
                                  poll=args.poll, provider=args.provider), cfg)
        print()
    elif args.profile and args.profile != state.get("profile"):
        ui.warn(f"--profile is only used with --up when no pod is active; using the active "
                f"pod's ({state.get('profile')})")
    return load_state()


def job_target(cfg: Config, state: dict) -> tuple[str, str, dict]:
    """(url, api key, what the pod serves) of the registered pod."""
    url = pod_url_of(cfg, state)
    served = jobs.info(url, state["api_key"])
    return url, state["api_key"], served


def pod_url_of(cfg: Config, state: dict) -> str:
    from .pod import require_url
    return require_url(cfg, state)


def output_dir(args, cfg: Config) -> Path:
    return Path(args.out_dir or cfg.output_dir)


def run_job(cfg: Config, state: dict, body: dict, label: str, out: Path) -> list[Path]:
    url, key, _ = job_target(cfg, state)
    job = jobs.submit(url, key, body)
    started = time.monotonic()
    job = jobs.wait(url, key, job, label)
    saved = jobs.download(url, key, job, out)
    ui.ok(f"{label} done in {ui.duration(time.monotonic() - started)}")
    for path in saved:
        size = path.stat().st_size
        ui.step(f"{path} {ui.dim(f'· {size / 1e6:.1f} MB')}")
    return saved


def mesh_params(args) -> dict:
    params = {}
    for key in ("texture_size", "decimation_target", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if getattr(args, "no_remesh", False):
        params["remesh"] = False
    if getattr(args, "no_texture", False):
        params["texture"] = False
    return params


def image_params(args) -> dict:
    params = {}
    for key in ("steps", "guidance", "width", "height", "seed", "negative"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    return params


def cmd_mesh(args, cfg: Config) -> None:
    """Image -> textured 3D mesh (GLB), or prompt -> image -> mesh with --prompt."""
    if bool(args.images) == bool(args.prompt):
        raise ConfigError("Give one or more images, or --prompt (not both)\n"
                          'e.g. `kami mesh photo.png` or `kami mesh --prompt "a ceramic owl"`')
    named = [jobs.view_arg(i) for i in args.images]
    views = getattr(args, "views", False) is True or any(view for view, _ in named)
    paths = [Path(p) for _, p in named]
    encoded = [jobs.read_image(path) for path in paths]       # check every file before billing
    if views and len(paths) < 2 and not any(view for view, _ in named):
        raise ConfigError("--views makes one mesh out of several images of one object: give two or more")
    state = ensure_pod(args, cfg)
    _, _, served = job_target(cfg, state)
    if not served.get("mesh_model"):
        raise ConfigError(f"The active pod (profile {state.get('profile')}) serves no mesh model\n"
                          "`kami down`, then `kami up -p <a profile with mesh_model>`.")
    out = output_dir(args, cfg)
    if views:
        most = served.get("max_views")
        if most is None:
            raise ConfigError("This pod's server predates several views per mesh\n"
                              "`kami reload` gives it the current code.")
        if len(paths) > most:
            raise ConfigError(f"This pod's model takes {'one image' if most == 1 else f'at most {most} views'} "
                              f"({served.get('mesh_model')})"
                              + ("\nSeveral views need TRELLIS.2 or tencent/Hunyuan3D-2mv." if most == 1 else ""))
        params = mesh_params(args)
        if getattr(args, "view_mode", None) in ("stochastic", "multidiffusion"):
            params["multiview_mode"] = args.view_mode
        body = {"kind": "mesh", "name": jobs.safe_name(paths[0].stem),
                "images": [{"image": image, "view": view} for (view, _), image in zip(named, encoded)],
                "params": params}
        run_job(cfg, state, body, f"Mesh of {len(paths)} views ({', '.join(p.name for p in paths)})", out)
        ui.hint("`kami ui` shows the results in your browser; or open the .glb in Blender")
        return
    if args.prompt:
        if not served.get("image_model"):
            raise ConfigError(f"--prompt needs an image model too; profile {state.get('profile')} "
                              "has none\nUse a profile with both (e.g. `studio`), or give an image.")
        body = {"kind": "text-to-mesh", "prompt": args.prompt,
                "name": jobs.safe_name(args.prompt),
                "params": {"image": image_params(args), "mesh": mesh_params(args)}}
        run_job(cfg, state, body, "Prompt -> image -> mesh", out)
        return
    for i, (path, image) in enumerate(zip(paths, encoded), 1):
        label = f"Mesh {i}/{len(paths)} ({path.name})" if len(paths) > 1 else f"Mesh of {path.name}"
        body = {"kind": "mesh", "image": image, "name": jobs.safe_name(path.stem),
                "params": mesh_params(args)}
        run_job(cfg, state, body, label, out)
    ui.hint("`kami ui` shows the results in your browser; or open the .glb in Blender")


def cmd_image(args, cfg: Config) -> None:
    """Prompt -> image (PNG)."""
    state = ensure_pod(args, cfg)
    _, _, served = job_target(cfg, state)
    if not served.get("image_model"):
        raise ConfigError(f"The active pod (profile {state.get('profile')}) serves no image model\n"
                          "`kami down`, then `kami up -p <a profile with image_model>`.")
    out = output_dir(args, cfg)
    base_seed = args.seed
    for i in range(args.count):
        params = image_params(args)
        if base_seed is not None:
            params["seed"] = base_seed + i
        label = f"Image {i + 1}/{args.count}" if args.count > 1 else "Image"
        run_job(cfg, state, {"kind": "image", "prompt": args.prompt,
                             "name": jobs.safe_name(args.prompt), "params": params}, label, out)
    ui.hint("`kami ui` shows the results in your browser")


def views_params(args) -> dict:
    params = {}
    for key in ("steps", "guidance", "seed", "lora_scale", "elevation", "distance"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if getattr(args, "angles", None):
        params["views"] = list(args.angles)
    return params


def mesh_command(paths: list[Path]) -> str:
    """`kami mesh front=a.png left=b.png ...` for the views a views job saved."""
    named = []
    for path in paths:
        stem = re.sub(r"-\d+$", "", path.stem)          # a-owl-front-2: kept a file there
        view = next((v for v in sorted(jobs.VIEW_ANGLES, key=len, reverse=True)
                     if stem.endswith(f"-{v}")), None)
        named.append(f"{view}={path}" if view else None)
    return "kami mesh " + " ".join(n for n in named if n) if any(named) else ""


def cmd_views(args, cfg: Config) -> None:
    """Concept (or front picture) -> pictures of the object from several sides, for a
    mesh model that takes views (Hunyuan3D-2mv, TRELLIS.2 --views)."""
    if bool(args.prompt) == bool(args.front):
        raise ConfigError("Give a concept, or --front with a picture of its front (not both)\n"
                          'e.g. `kami views "a ceramic owl"` or `kami views --front owl.png`')
    front = jobs.read_image(Path(args.front)) if args.front else None   # before billing
    state = ensure_pod(args, cfg)
    _, _, served = job_target(cfg, state)
    if not served.get("views_model"):
        raise ConfigError(f"The active pod (profile {state.get('profile')}) serves no views model\n"
                          "`kami down`, then `kami up -p views-qwen` (or views-flux).")
    if not front and not served.get("views_from_prompt"):
        raise ConfigError(f"{served['views_model']} edits pictures and this pod has no image model\n"
                          "Give the front with --front, or use a profile that makes it from text.")
    params = views_params(args)
    angles = params.get("views") or (served.get("views_defaults") or {}).get("views") or []
    body = {"kind": "views", "params": params,
            "name": jobs.safe_name(args.prompt) if args.prompt else jobs.views_name(Path(args.front).stem)}
    if front:
        body["image"] = front
    else:
        body["prompt"] = args.prompt
    label = f"Views ({', '.join(['front', *[a for a in angles if a != 'front']])})"
    saved = run_job(cfg, state, body, label, output_dir(args, cfg))
    command = mesh_command(saved)
    if command:
        ui.hint(f"`{command}` on a Pixal3D pod (`kami up -p pixal3d`), or Hunyuan3D-2mv "
                "(front, left, back, right only)")


RELOAD_TIMEOUT = 15 * 60      # the pipelines load again: a few minutes


def cmd_reload(args, cfg: Config) -> None:
    """Send this kami's server.py to the running pod and restart the server with it: no
    new pod, no download (the weights stay on its disk), only the pipelines load again."""
    state = load_state()
    if not state.get("pod_id"):
        raise ConfigError("No pod is active\n`kami up` starts one with the current code anyway.")
    url, key = pod_url_of(cfg, state), state["api_key"]
    served = jobs.info(url, key)
    if not served.get("reload"):
        raise ConfigError("This pod's server predates `kami reload`\n"
                          "`kami down`, then `kami up`: the new pod gets the current code.")
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    status, raw = jobs._request(url + "/v1/reload", key, {"source": source, "force": args.force})
    if status == 409:
        detail = jobs._json(200, raw, "").get("error", "")
        raise ConfigError(f"The pod is busy: {detail}\n"
                          "Wait for its jobs, or `kami reload --force` (they are lost).")
    jobs._json(status, raw, "Sending the new server code")
    sent = time.time()
    ui.ok(f"New server code sent ({len(source) // 1024} KiB); the server restarts with it")
    progress = ui.Progress("Restarting kami-server")
    started = time.monotonic()
    try:
        while True:
            if time.monotonic() - started > RELOAD_TIMEOUT:
                raise ConfigError(f"The server is not back after {ui.duration(RELOAD_TIMEOUT)}\n"
                                  "`kami logs` shows why; `kami down` + `kami up` starts over.")
            if healthy(url, timeout=5):
                try:
                    info = jobs.info(url, key)
                except ConfigError:
                    info = {}
                if (info.get("started_at") or 0) >= sent - 5:
                    break
                progress.phase("Restarting kami-server")
            else:
                progress.phase("Loading the pipelines again (the weights are on the pod's disk)")
            for _ in range(12):
                progress.tick()
                time.sleep(0.25)
    finally:
        progress.stop()
    ui.ok(f"Server ready with the new code in {ui.duration(time.monotonic() - started)}")


JOBS_WATCH_EVERY = 3            # seconds between refreshes of `kami jobs --watch`


def job_lines(listing: dict, now: float | None = None) -> list[str]:
    """The pod's jobs as console lines: what runs (stage, bar, step), the queue, the rest."""
    now = time.time() if now is None else now
    lines = []
    for j in listing.get("jobs") or []:
        name = j.get("name") or j.get("id")
        head = f"{j.get('id', '?'):<12}  {j.get('kind', '?'):<12}  {name[:32]:<32}"
        status = j.get("status")
        if status == "running":
            fraction = min(1.0, max(0.0, float(j.get("progress") or 0)))
            filled = round(ui.BAR_WIDTH * fraction)
            bar = ui.cyan(ui.BAR[0] * filled) + ui.dim(ui.BAR[1] * (ui.BAR_WIDTH - filled))
            stage = j.get("stage_label") or "running"
            if j.get("steps"):
                stage += f" · step {j.get('step')}/{j['steps']}"
            took = ui.duration(now - (j.get("started") or now))
            lines.append(f"{head}  {bar} {fraction * 100:3.0f}%  {stage} {ui.dim(took)}")
        elif status == "queued":
            waited = ui.duration(now - (j.get("created") or now))
            lines.append(f"{head}  {ui.yellow('queued')} #{j.get('position') or '?'} {ui.dim(waited)}")
        elif status == "done":
            took = ui.duration(j.get("run_seconds") or 0)
            lines.append(f"{head}  {ui.green('done')} in {took}  {ui.dim(', '.join(j.get('files') or []))}")
        else:
            error = (j.get("error") or "failed").splitlines()[0][:80]
            lines.append(f"{head}  {ui.red('failed')}  {ui.dim(error)}")
    return lines


def cmd_jobs(args, cfg: Config) -> None:
    """The pod's jobs: the running one with its stage and progress, the queue, the
    finished ones (whoever sent them: this terminal, another one, `kami ui`)."""
    state = load_state()
    if not state.get("pod_id"):
        raise ConfigError("No pod is active\n`kami up` starts one.")
    url, key = pod_url_of(cfg, state), state["api_key"]
    while True:
        listing = jobs.listing(url, key)
        if listing is None:
            raise ConfigError("This pod's server predates `kami jobs`\n"
                              "`kami reload` gives it the current code (a pod older than "
                              "`kami reload` itself needs `kami down`, `kami up`).")
        if args.json:
            print(json.dumps(listing, indent=2))
            return
        if args.watch and ui.TTY:
            sys.stdout.write("\033[H\033[J")          # redraw in place
        lines = job_lines(listing)
        summary = f"{listing.get('running', 0)} running · {listing.get('queued', 0)} queued"
        ui.step(f"Jobs on pod {state['pod_id']} {ui.dim('· ' + summary)}")
        for line in lines or [ui.dim("no jobs yet")]:
            print(f"    {line}")
        if not args.watch:
            return
        ui.hint(f"refreshing every {JOBS_WATCH_EVERY} s · Ctrl+C to stop")
        try:
            time.sleep(JOBS_WATCH_EVERY)
        except KeyboardInterrupt:
            return


class WebBackend:
    """What `kami ui`'s page needs from kami: the pod's state, where to send jobs, and
    deleting the pod. The pod's API key stays here; the page never sees it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.served: dict[str, dict] = {}       # pod id -> what it serves (/v1/info)

    def status(self) -> dict:
        state = load_state()
        info = status_info(self.cfg, state)
        pod = info["pod"]
        served = None
        if pod and info["ready"]:
            if pod["id"] not in self.served:
                try:
                    self.served[pod["id"]] = jobs.info(pod["url"], state["api_key"])
                except ConfigError:
                    pass
            served = self.served.get(pod["id"])
        if pod:
            pod = {k: v for k, v in pod.items() if k != "url"}
        return {"pod": pod, "ready": info["ready"], "waiting": info["waiting"],
                "served": served and {k: served.get(k) for k in
                                      ("mesh_model", "image_model", "mesh_defaults", "image_defaults",
                                       "mesh_backend", "max_views", "view_names", "multiview_modes",
                                       "views_model", "views_defaults", "view_angles",
                                       "views_from_prompt")}}

    def target(self) -> tuple[str, str]:
        state = load_state()
        if not state.get("pod_id"):
            raise ConfigError("No pod is active: run `kami up` in a terminal")
        return pod_url_of(self.cfg, state), state["api_key"]

    def jobs(self) -> dict | None:
        """Every job on the pod, with stage and progress (None: a server without it)."""
        return jobs.listing(*self.target())

    def down(self) -> dict:
        state = load_state()
        if not state.get("pod_id"):
            raise ConfigError("No pod is registered")
        estimate = estimated_cost(state)
        pod_id = terminate(client(pod_provider(state)), state)
        return {"pod_id": pod_id,
                "session_cost_estimate": round(estimate, 2) if estimate is not None else None}


def cmd_ui(args, cfg: Config) -> None:
    """A local web page: make meshes and images on the active pod, browse the results."""
    folder = Path(args.folder) if args.folder else Path(cfg.output_dir)
    if not folder.is_dir():
        if args.folder:
            raise ConfigError(f"Not a folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)        # empty gallery; fills as results come
    found = webui.cards(folder)
    meshes = sum(c["kind"] == "mesh" for c in found)

    def ready(url):
        ui.ok(f"kami ui · {folder} · {meshes} mesh(es), {len(found) - meshes} image(s)")
        ui.step(f"{ui.bold(url)} {ui.dim('(only this PC can open it)')}")
        ui.hint("It sends jobs to the pod `kami up` started; it never starts one itself.")
        ui.hint("Ctrl+C stops the page (the pod, if any, keeps running: `kami down`).")
        sys.stdout.flush()                   # the URL must show now, even when piped

    try:
        work = webui.serve(folder, WebBackend(cfg), port=args.port,
                           open_browser=not args.no_browser, on_ready=ready)
    except OSError as e:
        raise ConfigError(f"Cannot serve on port {args.port}: {e}\n"
                          "Pick another with --port, or leave it out for a free one.") from None
    print()
    left = work.active() if work else 0
    if left:
        ui.warn(f"{left} job(s) from the page were still in progress: they finish on the pod, "
                "but their files were not downloaded")
    ui.ok("kami ui stopped")
    if _pod_registered():
        ui.hint("The pod is still up and billing: `kami status` · `kami down`")


# Commands whose output is not framed with blank lines (none at the moment).
UNPADDED: tuple[str, ...] = ()


# Commands that can stop half-way with a pod still up: remind that it bills.
BILLING_HINT = ("up", "watchdog", "mesh", "image")


# Commands that load (or create) the config themselves, to report problems with it.
OWN_CONFIG = ("init", "doctor")


# Commands that must work even with a broken kami.toml: a typo must never keep you
# from seeing or deleting a pod that bills. They fall back to the default settings.
LENIENT = ("down", "status", "logs")


def command_config(command: str) -> Config | None:
    if command in OWN_CONFIG:
        return None
    try:
        return load_config()
    except ConfigError as e:
        if command not in LENIENT:
            raise
        ui.warn(f"kami.toml has problems; using default settings for `{command}`")
        for line in str(e).splitlines()[1:4]:
            ui.hint(line)
        ui.hint("`kami doctor` lists every problem")
        print()
        return config.fallback_config()


def save_traceback() -> Path | None:
    try:
        import traceback
        config.project.state_dir.mkdir(exist_ok=True)
        path = config.project.state_dir / "last-error.log"
        path.write_text(traceback.format_exc(), encoding="utf-8")
        return path
    except OSError:
        return None


def _pod_registered() -> bool:
    try:
        return bool(load_state().get("pod_id"))
    except ConfigError:
        return False


def add_provider_option(p, what: str) -> None:
    p.add_argument("--provider", choices=config.PROVIDERS,
                   help=f"{what} (default: provider in kami.toml)")


def add_stock_options(p) -> None:
    p.add_argument("--secure", action="store_true",
                   help="if the configured cloud (community) has no stock, use secure cloud "
                        "(pricier) instead of waiting")
    p.add_argument("--max-wait", type=float, metavar="MINUTES",
                   help="give up after waiting this long for stock (default: wait until Ctrl+C)")
    p.add_argument("--poll", type=int, metavar="SECONDS",
                   help=f"seconds between stock attempts while waiting (default {STOCK_POLL}; "
                        f"minimum {MIN_STOCK_POLL})")


def add_pod_options(p) -> None:
    """--up and what it takes, for the commands that run jobs."""
    p.add_argument("--up", action="store_true", help="bring a pod up first if none is active")
    p.add_argument("-p", "--profile", help="profile to use with --up")
    p.add_argument("-o", "--out-dir", metavar="DIR",
                   help="where to save the results (default: output_dir in kami.toml)")
    add_stock_options(p)
    add_provider_option(p, "where --up runs the pod")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kami", description=__doc__)
    parser.add_argument("--version", action="version", version=f"kami {__version__}")
    parser.add_argument("--config", metavar="PATH",
                        help="kami.toml to use (default: KAMI_CONFIG, else the nearest one "
                             "from the current directory up)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="set up a project here: kami.toml, .env and checks")
    p.set_defaults(func=cmd_init)
    sub.add_parser("doctor", help="check the setup: config, API key, server image, models, stock").set_defaults(func=cmd_doctor)

    p = sub.add_parser("volume", help="list, create or delete RunPod network volumes")
    vsub = p.add_subparsers(dest="action")
    vsub.add_parser("ls", help="list volumes with size and monthly cost")
    vsub.add_parser("create", help="create the volume configured in kami.toml")
    v = vsub.add_parser("rm", help="delete a volume and the models on it")
    v.add_argument("id")
    v.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_volume, action="ls")
    sub.add_parser("profiles", help="list the configured profiles").set_defaults(func=cmd_profiles)

    p = sub.add_parser("gpus", help="list GPUs, price and datacenters with stock")
    p.add_argument("--min-vram", type=int, default=24, help="minimum VRAM in GB (default 24)")
    p.add_argument("--secure", action="store_true",
                   help="list secure cloud instead of the configured cloud")
    add_provider_option(p, "list this provider's GPUs")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("prices", help="what each profile costs now: GPU, disk, download and total")
    p.add_argument("profiles", nargs="*", metavar="PROFILE", help="profiles to price (default: all)")
    p.add_argument("--hours", type=float, default=1,
                   help="session length for the total (default 1)")
    p.add_argument("--secure", action="store_true",
                   help="price secure cloud instead of the configured cloud")
    add_provider_option(p, "price on this provider")
    p.set_defaults(func=cmd_prices)

    p = sub.add_parser("up", help="bring the pod up and wait until the server is ready")
    p.add_argument("-p", "--profile", help="profile from kami.toml (default: the one in [defaults])")
    p.add_argument("--keep-on-failure", action="store_true",
                   help="if the server fails to start, do not delete the pod (to investigate with `kami logs`)")
    p.add_argument("--budget", type=float, metavar="USD",
                   help="dollars this pod may spend: the guard deletes it when they run out "
                        "(GPU + disk, and the weight download on vast.ai, priced now)")
    p.add_argument("--max-hours", type=float,
                   help="the pod deletes itself after this many hours (default: max_hours in kami.toml; 0 = never)")
    p.add_argument("--idle", type=float, metavar="MINUTES",
                   help="the pod deletes itself after this many minutes without jobs "
                        "(default: idle_minutes in kami.toml; 0 = never)")
    p.add_argument("--skip-memory-check", action="store_true",
                   help="try the profile's GPUs even if they have less than min_vram_gb")
    add_stock_options(p)
    add_provider_option(p, "where to run the pod")
    p.set_defaults(func=cmd_up)

    sub.add_parser("down", help="delete the pod (the volume is kept)").set_defaults(func=cmd_down)
    p = sub.add_parser("cost", help="what the provider billed, per pod, and the account balance")
    p.add_argument("--days", type=int, default=7, help="days back, today included (default 7)")
    add_provider_option(p, "whose billing to read")
    p.set_defaults(func=cmd_cost)
    p = sub.add_parser("status", help="pod state, cost and URL",
                       epilog="Exit code: 0 ready, 2 pod not ready or gone, 3 no pod.")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watchdog", help="auto-delete the pod after a maximum of billing hours")
    p.add_argument("--max-hours", type=float,
                   help="delete the pod once its total uptime reaches this many hours "
                        "(default: max_hours in kami.toml)")
    p.add_argument("--poll", type=int, default=WATCHDOG_POLL,
                   help=f"seconds between provider checks (default {WATCHDOG_POLL})")
    p.set_defaults(func=cmd_watchdog)

    p = sub.add_parser("logs", help="stream the pod logs live")
    p.add_argument("-n", "--tail", type=int, default=100)
    p.add_argument("--no-follow", action="store_true", help="print the backlog and exit")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="also show the guard's boot progress samples (every few seconds)")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("mesh", help="image -> textured 3D mesh (GLB); or --prompt for text -> image -> mesh",
                       epilog="e.g. `kami mesh photo.png`, `kami mesh a.png b.jpg -o meshes`, "
                              "`kami mesh --views front.png side.png back.png`, "
                              "`kami mesh front=a.png left=b.png back=c.png` (Hunyuan3D-2mv), "
                              "`kami mesh --prompt \"a ceramic owl, studio light\"`")
    p.add_argument("images", nargs="*", metavar="IMAGE",
                   help="images to turn into meshes (one each); with --views, views of one object; "
                        "VIEW=IMAGE names a view (front, left, back, right) and implies --views")
    p.add_argument("--views", action="store_true",
                   help="one mesh out of all the images: views of the same object (experimental "
                        "on TRELLIS.2; Hunyuan3D-2mv is made for it)")
    p.add_argument("--view-mode", choices=("stochastic", "multidiffusion"),
                   help="TRELLIS.2 with --views: one view per sampling step (stochastic, default) "
                        "or all of them averaged (multidiffusion, as many times slower as views)")
    p.add_argument("--no-texture", action="store_true", help="Hunyuan3D-2: the bare shape, no texture")
    p.add_argument("--prompt", help="make the image first from this prompt (needs an image model)")
    p.add_argument("--seed", type=int, help="random seed, for repeatable results")
    p.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096),
                   help="texture resolution (default 2048)")
    p.add_argument("--decimation-target", type=int, metavar="FACES",
                   help="faces to simplify the mesh to (default 500000)")
    p.add_argument("--no-remesh", action="store_true",
                   help="keep TRELLIS.2's raw surface instead of remeshing it")
    p.add_argument("--steps", type=int, help="with --prompt: image denoising steps")
    p.add_argument("--guidance", type=float, help="with --prompt: image guidance scale")
    p.add_argument("--width", type=int, help=argparse.SUPPRESS)
    p.add_argument("--height", type=int, help=argparse.SUPPRESS)
    p.add_argument("--negative", help=argparse.SUPPRESS)
    add_pod_options(p)
    p.set_defaults(func=cmd_mesh)

    p = sub.add_parser("image", help="prompt -> image (PNG)",
                       epilog='e.g. `kami image "a ceramic owl, studio light" -n 4 --seed 7`')
    p.add_argument("prompt")
    p.add_argument("-n", "--count", type=int, default=1, help="how many images (default 1)")
    p.add_argument("--seed", type=int, help="random seed (with -n, one per image from here up)")
    p.add_argument("--steps", type=int, help="denoising steps (default: the profile's)")
    p.add_argument("--guidance", type=float, help="guidance scale (default: the profile's)")
    p.add_argument("--width", type=int, help="width in pixels (default 1024)")
    p.add_argument("--height", type=int, help="height in pixels (default 1024)")
    p.add_argument("--negative", help="negative prompt (models with guidance only)")
    add_pod_options(p)
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("views", help="concept -> pictures of one object from several sides (for a mesh)",
                       epilog='e.g. `kami views "a ceramic owl"`, `kami views --front owl.png`, '
                              '`kami views "a robot" --angles left back right front-left`')
    p.add_argument("prompt", nargs="?", help="the concept (its front picture is made first)")
    p.add_argument("--front", metavar="IMAGE", help="start from this picture of the front instead")
    p.add_argument("--angles", nargs="+", choices=jobs.VIEW_ANGLES, metavar="VIEW",
                   help=f"views to make (default: left back right); any of {', '.join(jobs.VIEW_ANGLES)}")
    p.add_argument("--seed", type=int, help="random seed (the same for every view)")
    p.add_argument("--steps", type=int, help="denoising steps (default: the profile's)")
    p.add_argument("--guidance", type=float, help="guidance scale (default: the profile's)")
    p.add_argument("--lora-scale", type=float, help="weight of the camera LoRA (default 0.9)")
    p.add_argument("--elevation", help='camera height in the LoRA\'s words (default "eye-level shot")')
    p.add_argument("--distance", help='camera distance in the LoRA\'s words (default "medium shot")')
    add_pod_options(p)
    p.set_defaults(func=cmd_views)

    p = sub.add_parser("reload", help="send this kami's server code to the running pod and restart "
                                      "the server (no new pod, no download)")
    p.add_argument("--force", action="store_true", help="even if jobs are running (they are lost)")
    p.set_defaults(func=cmd_reload)

    p = sub.add_parser("jobs", help="the pod's jobs: stage and progress of the running one, the queue",
                       epilog="e.g. `kami jobs --watch`")
    p.add_argument("-w", "--watch", action="store_true", help=f"refresh every {JOBS_WATCH_EVERY} s")
    p.add_argument("--json", action="store_true", help="the raw listing, as JSON")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("ui", help="web page on this PC: make meshes and images, browse the results",
                       epilog="e.g. `kami ui`, `kami ui meshes --port 8765`")
    p.add_argument("folder", nargs="?", help="folder to show (default: output_dir in kami.toml)")
    p.add_argument("--port", type=int, default=0, help="local port (default: a free one)")
    p.add_argument("--no-browser", action="store_true", help="print the URL, do not open a browser")
    p.set_defaults(func=cmd_ui)

    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    config.use_project(args.config)
    load_env()
    padded = args.command not in UNPADDED and not getattr(args, "json", False)
    if padded:
        print()
    code = 0
    try:
        code = args.func(args, command_config(args.command)) or 0
    except (ConfigError, ProviderError) as e:
        sys.stdout.flush()
        ui.error(str(e))
        code = 1
        # The error may have stopped a command mid-way while the GPU keeps billing.
        if args.command in BILLING_HINT and _pod_registered():
            ui.hint("A pod is still registered and may be billing: `kami status` · `kami down`")
    except (OSError, json.JSONDecodeError) as e:
        sys.stdout.flush()
        ui.error(f"Unexpected error: {e}")
        code = 1
    except KeyboardInterrupt:
        print()
        ui.warn("Interrupted.")
        if args.command in BILLING_HINT and _pod_registered():
            ui.hint("The pod is still up and billing: `kami status` · `kami down`")
        code = 130
    except Exception as e:
        # a bug: say so plainly, keep the traceback for the report, and never let it
        # hide a pod that may be billing
        sys.stdout.flush()
        ui.error(f"Unexpected error (a bug in kami): {type(e).__name__}: {e}")
        log = save_traceback()
        if log:
            ui.hint(f"Traceback saved to {log}")
        if _pod_registered():
            ui.hint("A pod is registered and may be billing: `kami status` · `kami down`")
        code = 1
    if padded:
        sys.stderr.flush()
        print()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
