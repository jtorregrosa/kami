# CLAUDE.md

## What this project is

`kami` is a small Python CLI (standard library only, Python 3.11+) that manages the
lifecycle of a cloud GPU pod on [RunPod](https://runpod.io) or [vast.ai](https://vast.ai)
(chosen by `provider` in `kami.toml`) which runs **kami-server**: TRELLIS.2 for
image → textured 3D mesh (GLB) and a diffusers pipeline for text → image. The CLI sends
jobs to the pod and saves the results locally.

It was forked from `../ronin` (the same tool for LLMs) and keeps its pod lifecycle, cost
guard, providers, stock waiting, budget and cost reporting. A fix in that shared plumbing
(`api`, `runpod`, `vast`, `catalog`, `guard`, `watchdog`, `ui`) probably applies to both.

```
Your PC ── kami mesh photo.png ──HTTPS──► https://<pod>-8000.proxy.runpod.net
                                            └ guard.py → server.py (jobs API)
                                                         ├ TRELLIS.2 (image -> GLB)
                                                         └ diffusers (prompt -> PNG)
```

## Layout

- `kami/cli.py` — argparse CLI and the commands (`init`, `doctor`, `gpus`, `prices`,
  `profiles`, `volume`, `up`, `down`, `status`, `cost`, `logs`, `watchdog`, `mesh`, `image`).
- `kami/server.py` — runs *inside the pod* (its source is sent in `KAMI_SERVER_SRC` and
  run with `python3 -c`): loads the pipelines, serves `/health`, `/metrics`, `/v1/info`
  and the jobs API (`POST /v1/jobs`, `GET /v1/jobs/<id>`, `GET /v1/jobs/<id>/files/<name>`).
  Stdlib HTTP; torch / trellis2 / diffusers are imported lazily, so the tests import it
  without them. kami never imports it at runtime.
- `kami/jobs.py` — client side of the jobs API: submit, poll with a progress line, download
  (never overwriting).
- `kami/pod.py` — pod plumbing: provider client, `server_spec` (env + guard-wrapped
  entrypoint + image), `pod_body` / `vast_body`, endpoint URL, health, log streams,
  termination, `/metrics` reading (`server_usage`).
- `kami/guard.py` — runs *inside the pod*: wraps the server, deletes the pod at a deadline
  or when idle (from kami-server's job metrics), reports boot progress, opens a Cloudflare
  tunnel on vast.ai. Standard library only.
- `kami/fit.py` — `min_vram_gb` check per GPU, download size of a profile's repos, and
  access to every repo (gated ones: DINOv3, RMBG-2.0), from the Hugging Face API.
- `kami/config.py` — project discovery, `kami.toml` parsing and validation (profiles with
  `extends`; `mesh_model` / `image_model`), `.env`, local state in `.kami/`.
- `kami/api.py`, `kami/runpod.py`, `kami/vast.py`, `kami/catalog.py` — provider clients and
  GPU name tables (unchanged from ronin).
- `kami/watchdog.py`, `kami/ui.py`, `kami/mascot.py` — local guard, console output, mascot.
- `docker/Dockerfile` — the kami-server image (torch 2.6 / CUDA 12.4 + TRELLIS.2 + diffusers);
  `.github/workflows/server-image.yml` builds and pushes it to ghcr.io.
- `tools/convert_dinov3.py` — one-off helper (needs torch, not part of the package): Meta's
  DINOv3 `.pth` → transformers format → a private HF repo, for `dino_model`.
- `tests/` — `python -m unittest`.

## Conventions

- Standard library only on the client; no third-party runtime dependencies.
- `server.py` and `guard.py` change without an image rebuild. Only dependencies belong in
  the Dockerfile.
- All console output goes through `kami/ui.py`. Respect `NO_COLOR`, non-TTY output and
  ASCII fallbacks.
- Costs matter: commands that leave a GPU pod running should make the cost and how to
  stop it obvious (`kami down`, `kami status` warnings, `kami watchdog`).
- Check what can be checked before a pod is billed (config, image placeholder, gated
  repos, GPU memory).
- User-facing docs are in English (`README.md`). Keep the command table in sync.

## Not verified yet

The client side is unit-tested with mocks. These parts have not run on a real GPU yet:
the Docker build, TRELLIS.2's `pipeline.run` / `o_voxel.postprocess.to_glb` arguments in
`server.py`, `dino_model` / `rembg_model` swapping TRELLIS.2's sub-models
(`override_model`), and `offload` moving the pipelines between CPU and GPU. Check them on the
first real pod (`kami up --keep-on-failure`, `kami logs`).

## Working notes

- The user often writes in Spanish; reply in the language they use.
- Target OS is Windows (PowerShell is the primary shell), so prefer cross-platform code
  and test PowerShell-facing output paths.
