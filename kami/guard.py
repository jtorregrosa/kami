"""In-pod guard: run kami-server and delete the pod at a deadline or once it sits idle.

This file is not imported by kami: its source is sent to the pod and run there as
`python3 -c <source> python3 -c <server> ...` (standard library only, inside the
kami-server image).
It works even when your PC is asleep or off, unlike `kami watchdog`.

Environment (set by `kami up`):
  KAMI_DEADLINE       unix time after which the pod deletes itself (0 = no deadline)
  KAMI_IDLE           seconds without requests before it deletes itself (0 = never)
  KAMI_PORT           port the server listens on (its /metrics tells requests apart)
  KAMI_BUSY_METRICS   comma-separated gauges of in-flight work (default: kami-server's)
  KAMI_DONE_METRIC    counter that moves on every finished job (default: kami-server's)
  KAMI_PROVIDER       "runpod" (default) or "vast": how the pod deletes itself
  KAMI_TUNNEL         "1": open a Cloudflare quick tunnel to the port and log its HTTPS URL
  KAMI_WATCH          ":"-separated dirs the server downloads into: until it answers, log
                       the bytes they gained and the GPU memory in use (`kami up` draws
                       them as progress bars)
RunPod provides RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY in every pod; vast.ai
provides CONTAINER_ID and an instance-scoped CONTAINER_API_KEY.
"""

import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CHECK_EVERY = 30
# Per provider, tried in order: delete is what kami wants; stop at least ends GPU
# billing (a stopped vast.ai instance still bills its disk).
ACTIONS = {
    "runpod": (
        ("DELETE", "https://api.runpod.io/v2/pods/{id}", None),
        ("DELETE", "https://rest.runpod.io/v1/pods/{id}", None),
        ("POST", "https://api.runpod.io/v2/pods/{id}/action", {"action": "stop"}),
    ),
    "vast": (
        ("DELETE", "https://console.vast.ai/api/v0/instances/{id}/", None),
        ("PUT", "https://console.vast.ai/api/v0/instances/{id}/", {"state": "stopped"}),
    ),
}
PROGRESS_EVERY = 3      # seconds between boot progress samples
HF_PING = "https://huggingface.co/api/whoami-v2"   # tiny; any HTTP answer means reachable
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+[.]trycloudflare[.]com")
# Gauges of in-flight work and a counter that moves on every finished job (kami-server's
# /metrics; see server.py). A failed job counts as activity too.
BUSY = tuple(m for m in os.environ.get(
    "KAMI_BUSY_METRICS", "kami:jobs_running,kami:jobs_queued").split(",") if m)
DONE = tuple(m for m in os.environ.get(
    "KAMI_DONE_METRIC", "kami:jobs_finished_total,kami:jobs_failed_total").split(",") if m)


def log(msg):
    print(f"[kami-guard] {msg}", flush=True)


_refused = False


def activity(port):
    """(busy, finished) from the server's /metrics, or None while the server is not up.

    kami-server guards /metrics with its API key (only /health is public), so the key
    the pod already holds is sent."""
    global _refused
    headers = {}
    key = os.environ.get("KAMI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        # 503 is kami-server still loading the pipelines: not up yet, nothing to report.
        # A refusal (401/403) means the idle limit would silently never fire (only the
        # deadline would still protect the pod): say so, once.
        if e.code in (401, 403) and not _refused:
            _refused = True
            log(f"/metrics answered HTTP {e.code}: the idle limit cannot see activity")
        return None
    except Exception:
        return None
    busy = finished = 0.0
    for line in text.splitlines():
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0]
        try:
            if name in BUSY:
                busy += float(value)
            elif name in DONE:
                finished += float(value)
        except ValueError:
            continue
    return busy, finished


def identity():
    """(provider, pod id, pod-scoped API key) from the env the provider sets."""
    provider = os.environ.get("KAMI_PROVIDER") or "runpod"
    if provider == "vast":
        # VAST_CONTAINERLABEL is "C.<id>": a fallback for the id
        label = os.environ.get("VAST_CONTAINERLABEL", "")
        pod_id = os.environ.get("CONTAINER_ID") or label.partition("C.")[2] or None
        return provider, pod_id, os.environ.get("CONTAINER_API_KEY")
    return provider, os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")


def delete_self():
    provider, pod_id, key = identity()
    if not pod_id or not key:
        log(f"{provider}: the pod id / API key are missing from the env: cannot delete the pod")
        return False
    for method, url, body in ACTIONS.get(provider, ()):
        req = urllib.request.Request(
            url.format(id=pod_id), method=method,
            data=json.dumps(body).encode() if body else None,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": "kami-guard"})
        try:
            urllib.request.urlopen(req, timeout=30).close()
            log(f"{method} {url.format(id=pod_id)}: done")
            return True
        except Exception as e:
            log(f"{method} {url.format(id=pod_id)} failed: {e}")
    return False


def watch(deadline, idle, port):
    idle_since = None       # monotonic time of the last observed activity
    last_finished = None
    while True:
        reason = None
        if deadline and time.time() >= deadline:
            reason = "deadline reached"
        elif idle:
            sample = activity(port)
            if sample is not None:
                busy, finished = sample
                if idle_since is None or busy or finished != last_finished:
                    idle_since = time.monotonic()
                last_finished = finished
                if time.monotonic() - idle_since >= idle:
                    reason = f"idle for {int(idle // 60)} min"
        if reason:
            log(f"{reason}: deleting the pod so the GPU stops billing")
            if delete_self():
                return
        time.sleep(CHECK_EVERY)


def tunnel(port):
    """Download cloudflared and open a quick tunnel to the server: an HTTPS URL for a
    provider that only publishes plain HTTP. Logs the URL (kami reads it from the log)."""
    path = "/tmp/cloudflared"
    try:
        if not os.path.exists(path):
            req = urllib.request.Request(CLOUDFLARED, headers={"User-Agent": "kami-guard"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
                f.write(resp.read())
            os.chmod(path, 0o755)
        proc = subprocess.Popen(
            [path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except Exception as e:
        log(f"tunnel unavailable: {e}")
        return
    announced = False
    for line in proc.stdout:            # keep reading: a full pipe would stall cloudflared
        match = TUNNEL_URL.search(line)
        if match and not announced:
            announced = True
            log(f"tunnel {match.group(0)}")
    log(f"tunnel closed (cloudflared exited with {proc.wait()})")


def check(ok, msg):
    """A boot check `kami up` shows as a line of its own."""
    log(f"check {'ok' if ok else 'warn'} {msg}")


def checks(dirs):
    """What the host brings to the boot: the GPU, free disk, and Hugging Face's latency."""
    gpus = nvidia_smi("name,memory.total,driver_version")
    for name, mib, driver in gpus:
        check(True, f"GPU {name} · {float(mib) / 1024:.0f} GiB · driver {driver}")
    if not gpus:
        check(False, "nvidia-smi gave no GPU (GPU memory will not be reported)")
    disk = next((d for d in dirs if os.path.isdir(d)), None) or (dirs[0] if dirs else "/")
    while not os.path.isdir(disk):
        disk = os.path.dirname(disk) or "/"
    check(True, f"Disk {shutil.disk_usage(disk).free / 1e9:.0f} GB free at {disk}")
    started = time.monotonic()
    req = urllib.request.Request(HF_PING, method="HEAD", headers={"User-Agent": "kami-guard"})
    try:
        urllib.request.urlopen(req, timeout=15).close()
    except urllib.error.HTTPError:
        pass                            # 401 without a token: reachable all the same
    except Exception as e:
        check(False, f"Hugging Face unreachable from the pod: {e}")
        return
    check(True, f"Hugging Face answers in {(time.monotonic() - started) * 1000:.0f} ms")


def nvidia_smi(fields):
    """Rows of `nvidia-smi --query-gpu=<fields>` (one per GPU), or [] without it."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [[v.strip() for v in row.split(",")] for row in out.splitlines() if row.strip()]


def tree_bytes(dirs):
    """Bytes in the regular files under `dirs` (symlinks, e.g. HF snapshots, not followed).
    hf_xet's chunk cache is skipped: it would count the same bytes twice."""
    total = 0
    for top in dirs:
        for root, subdirs, files in os.walk(top):
            if "xet" in subdirs:
                subdirs.remove("xet")
            for name in files:
                try:
                    st = os.lstat(os.path.join(root, name))
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    total += st.st_size
    return total


def answering(port):
    """True once the server's /health answers 200 (kami-server: 503 while loading)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def progress(dirs, port):
    """Until the server answers, log what it has downloaded and the GPU memory in use:
    `progress t=<unix> downloaded=<bytes> rate=<bytes/s> vram=<used>/<total MiB>`."""
    base = tree_bytes(dirs)             # a volume may already hold this or other models
    last, before, rate = None, (time.monotonic(), 0), 0.0
    while not answering(port):
        time.sleep(PROGRESS_EVERY)
        now, got = time.monotonic(), max(0, tree_bytes(dirs) - base)
        # smoothed: one sample is a few seconds of a bursty transfer
        instant = (got - before[1]) / max(now - before[0], 1e-6)
        rate = instant if not rate else 0.6 * rate + 0.4 * instant
        before = (now, got)
        gpus = nvidia_smi("memory.used,memory.total")
        vram = ""
        try:
            if gpus:
                vram = (f" vram={sum(float(u) for u, _ in gpus):.0f}"
                        f"/{sum(float(t) for _, t in gpus):.0f}")
        except ValueError:
            pass
        sample = (got, vram)
        if sample != last:              # nothing new, nothing logged
            last = sample
            log(f"progress t={time.time():.0f} downloaded={got} rate={max(rate, 0):.0f}{vram}")


def main():
    deadline = float(os.environ.get("KAMI_DEADLINE") or 0)
    idle = float(os.environ.get("KAMI_IDLE") or 0)
    port = os.environ.get("KAMI_PORT", "8000")
    parts = []
    if deadline:
        parts.append(time.strftime("deletes itself at %H:%M UTC", time.gmtime(deadline)))
    if idle:
        parts.append(f"or after {int(idle // 60)} min idle")
    log(" ".join(parts) or "no limits set")

    dirs = [d for d in os.environ.get("KAMI_WATCH", "").split(":") if d]
    if dirs:
        threading.Thread(target=checks, args=(dirs,), daemon=True).start()
        threading.Thread(target=progress, args=(dirs, port), daemon=True).start()
    if os.environ.get("KAMI_TUNNEL") == "1":
        threading.Thread(target=tunnel, args=(port,), daemon=True).start()
    child = subprocess.Popen(sys.argv[1:])
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: child.send_signal(s))
    if deadline or idle:
        threading.Thread(target=watch, args=(deadline, idle, port), daemon=True).start()
    sys.exit(child.wait())


if __name__ == "__main__":  # also true under `python3 -c`
    main()
