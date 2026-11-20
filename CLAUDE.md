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
  `profiles`, `volume`, `up`, `down`, `status`, `cost`, `logs`, `watchdog`, `mesh`, `image`,
  `views`, `jobs`, `reload`, `print`, `ui`).
- `kami/server.py` — runs *inside the pod* (its source is sent in `KAMI_SERVER_SRC` and
  run with `python3 -c`): loads the pipelines, serves `/health`, `/metrics`, `/v1/info`
  and the jobs API (`POST /v1/jobs`, `GET /v1/jobs` (all, with stage and progress),
  `GET /v1/jobs/<id>`, `GET /v1/jobs/<id>/files/<name>`). Progress comes from wrapping
  TRELLIS.2's pipeline methods (stages) and its samplers' tqdm (steps), and diffusers'
  `callback_on_step_end`. `POST /v1/reload` takes new server code (`kami reload`): it is
  saved to `/tmp/kami-server.py`, the server exits 75 and the guard starts it again.
  Two mesh backends (`KAMI_MESH_BACKEND`, from `config.mesh_backend(mesh_model)`):
  TRELLIS.2, with several views of one object through `trellis_multiview` (TRELLIS 1's
  tuning-free trick: the samplers' `_inference_model` patched per instance), and
  Hunyuan3D-2 (`HunyuanMesh`: hy3dgen shape + Hunyuan3D-Paint, -2mv with named views),
  and Pixal3D (`PixalMesh`: single and posed-view pipelines, the one in use on the GPU;
  views posed by `pixal_camera` from their names, framed alike by `same_framing`; NAF
  loaded from the image's pinned /opt/NAF instead of torch.hub; `-pixal3d` image).
  Hunyuan profiles need `accept_hunyuan_license` (its licence excludes the EU, UK and
  South Korea) and the `-hunyuan` image (`docker/Dockerfile.hunyuan`, built FROM the base).
  "views" jobs (`views_model`, `KAMI_VIEWS_*`): a front picture (image_model, or FLUX.2
  itself) and a diffusers editing pipeline with a camera LoRA ("<sks> back view ...")
  turning around it; Qwen-Image-Edit gets true CFG, FLUX.2 guidance.
  Stdlib HTTP; torch / trellis2 / diffusers are imported lazily, so the tests import it
  without them. kami never imports it at runtime.
- `kami/jobs.py` — client side of the jobs API: submit, poll with a progress line, download
  (never overwriting). `PRINT_HINT` / `printable`: the hint every prompt gets on the
  client (CLI `print_prompt`, page `webui._sent`) so the picture is printable: flat base,
  one piece, `{colors}` flat colours, matte. On by default; `--as-written`, `print_hint`
  in kami.toml. File names and labels keep the prompt as written.
- `kami/printing.py` — `kami print`, all on the PC: GLB -> 3MF for a multicolour printer.
  `load_glb` (own parser; one colour per triangle from the base colour texture), `solid`
  (weld, drop inner shells and slivers; reports open / non-manifold edges, fills nothing),
  `place` (Z up, base pressed flat, mm), `rebuild` (only when the cleaned mesh is not
  closed or is a hollow skin: voxel grid, flood from outside, signed distance to the
  outward-facing triangles, marching cubes, colours from the nearest old triangle; the
  method of `../jointwright`'s `packages/core/src/import/solidify.ts`, without its
  winding vote and its simplify), `quantize` / `nearest` (k-means in Oklab with lightness
  at half weight; or a fixed palette), `smooth` / `absorb`, `write_3mf` (`paint_color` for
  Bambu Studio / OrcaSlicer plus standard `basematerials`). The one module with
  third-party imports (numpy, Pillow; scipy and scikit-image inside `rebuild`), and only
  `cmd_print` imports it.
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
- `kami/webui.py` + `kami/webui.html` — `kami ui`: a page on 127.0.0.1 in three stages,
  Concept (text or a front picture -> views; image and views jobs), Mesh (views in named
  slots -> GLB) and Print (`printing` on this PC, painted preview), opening on the stage
  the pod's `/v1/info` says it serves. kami relays the jobs (the pod key stays in kami);
  the page never starts a pod, and a stage the pod lacks shows the `kami up -p` for it
  (`WebBackend.status()["profiles"]`). `objects()` is the library: `output_dir` and one
  folder deep, grouped by name (`VIEW_OF`: owl-left-2.png is the newest left view of
  "owl"); `resolve()` is the only way a name becomes a file. Pictures of a job may be
  uploads or `{file}` from the library. `Printer`: POST /api/print (`prepare` cached, so
  colours answer at once; `save` writes the 3MF), /preview/<id>.glb, /thumbs/<mesh> (a
  drawn PNG for meshes with no picture). The page builds its DOM from JSON with
  `textContent`, never HTML from file names. Host header checked; POSTs need the page's
  token, JSON and same Origin. `cli.WebBackend` gives it the pod status, target, `down`.
- `kami/watchdog.py`, `kami/ui.py`, `kami/mascot.py` — local guard, console output, mascot.
- `docker/Dockerfile.pixal3d` — Pixal3D + NAF (pinned) + natten 0.17.5 wheel on top of the base.
- `docker/Dockerfile` — the kami-server image (torch 2.6 / CUDA 12.4 + TRELLIS.2 + diffusers);
  `.github/workflows/server-image.yml` builds and pushes it to ghcr.io.
- `tools/convert_dinov3.py` — one-off helper (needs torch, not part of the package): Meta's
  DINOv3 `.pth` → transformers format → a private HF repo, for `dino_model`.
- `tests/` — `python -m unittest`.

## Conventions

- Standard library only for the pod commands, so they run anywhere. `kami print`
  (`kami/printing.py`) is the exception and may take dependencies (the `print` extra:
  numpy, Pillow, scipy, scikit-image), imported only by that command; its tests are in
  `tests/test_printing.py`, skipped without them.
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

The client side is unit-tested with mocks. Verified on a real 4090 pod: the Docker image,
TRELLIS.2 with `dino_model` (transformers 5 layout aliased by `fit_transformers`), and an
image -> GLB job end to end (PNG textures), `kami reload` (on an L40S). Pixal3D on a 4090
(`pixal3d`): 4 posed views from `kami views` -> a coherent GLB, and a single picture (both
~2.5 min at 1024 in its low-VRAM mode; 1536 with 4 views ran out of 24 GB); its meshes
are not watertight. Boot ~19 min (48 GB of weights). Images go through
`server.open_image` (Pillow; pillow-heif for HEIC; OpenCV if Pillow fails; EXIF turned;
almost opaque alpha dropped so the background is removed). Not run on a real GPU yet: `offload` moving the
pipelines between CPU and GPU, Z-Image-Turbo, text-to-mesh, the job
progress hooks, TRELLIS.2 multiview, a Hunyuan3D-2mv mesh from `kami views` pictures
(left/right: Hunyuan's example left.png faces the picture's left, the LoRAs' "left side
view" faces right, so VIEW_ANGLES swaps them; untested on an asymmetric object) (`kami views` ran: views-qwen works on an A100, jinja2 upgraded on boot by
`ensure_packages`; views-flux loads, LoRA on all 170 layers, but it barely changes the
picture even at lora_scale 3, left/back = the front), a whole Hunyuan3D-2 job (the -hunyuan image builds
and its pipelines load on a 4090, with `dinov2_v4_checkpoints` and
`trust_hunyuan_pipelines`; the first jobs failed in pymeshlab, whose plugins needed
libOpenGL.so.0, now in the image: shape clean-up and Hunyuan3D-Paint never ran), and the guard deleting the pod through GraphQL (the pod-scoped key got 403
from api.runpod.io/v2 and rest.runpod.io did not resolve: the old guard never deleted it).
Check them on the next pod (`kami up --keep-on-failure`, `kami logs`).

`kami print`: the owl's Pixal3D GLB (a double shell: an outer closed surface and the same
one inside, facing inwards, joined by 6 edges) becomes a closed solid of 251k triangles in
~2.5 s; Bambu Studio 2.9.1's command line loads the 3MF with nothing to repair and keeps
the paint (checked by exporting it back). Not checked: the file opened in Bambu Studio's
window (whether it offers to map the standard 3MF colours, and that paint wins if both
are there), a sliced or printed model, OrcaSlicer. TRELLIS.2 meshes (duck.glb) are left
with ~100 non-manifold edges after cleaning, so they are rebuilt: closed, 630k triangles
at 0.3 mm, ~9 s, and Bambu Studio's command line loads that too with nothing to repair.
Not met yet: a skin with an opening wider than a cube (it should come out reported as a
hollow skin; only built by hand in a test). The model's front does not always face the
plate's front (Pixal3D and TRELLIS.2 differ): turn it in the slicer.

The redesigned `kami ui` was looked at in a browser with no pod (Print: the owl painted,
2 colours, "My filaments") and with pretend pods (a backend returning a fixed status):
Concept on a views-qwen-like pod, Mesh on a pixal3d-like pod with a library object's four
views filling the slots, the library with drawn thumbnails, a phone-width layout. Never
with a real pod: no job has been sent from the new page, so "Make views", "Try fronts",
"Redo" of one view, "Make mesh" from library pictures and a finished job landing on the
stage are untested end to end (their request bodies are unit-tested).

The printing hint (`jobs.PRINT_HINT`) has never been sent to a pod: whether Z-Image-Turbo
and Qwen follow it (flat colours, base), and whether `kami print` then gives cleaner
colours than the glossy owl, is to be compared on the next `views-qwen` pod, same concept
and seed, with and without `--as-written`.

Profiles in the shipped `kami.toml` (2026-11-20): two are active, `views` (Z-Image-Turbo +
Qwen-Image-Edit, was `views-qwen`) and `mesh` (Pixal3D, was `pixal3d`, the default). The
rest are commented out at the end of the file: `trellis` (TRELLIS.2, was `mesh`), `image`,
`studio`, `studio-24`, `hunyuan`, `hunyuan-mv`, `views-flux`. Notes above written before
that use the old names. `kami init`'s template in config.py still writes a TRELLIS.2
`mesh` profile.

## Working notes

- The user often writes in Spanish; reply in the language they use.
- Target OS is Windows (PowerShell is the primary shell), so prefer cross-platform code
  and test PowerShell-facing output paths.
