# kami

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](#install)

**Rent a GPU only while you use it, to make 3D meshes and images.** `kami` brings a pod up
on [RunPod](https://runpod.io) or [vast.ai](https://vast.ai) that serves
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (image → textured GLB mesh) and a
[diffusers](https://github.com/huggingface/diffusers) text-to-image model
([Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by default). You send it
jobs from your PC, and kami makes sure the GPU stops billing when you walk away.

```
Your PC ── kami mesh photo.png ──HTTPS──► https://<pod>-8000.proxy.runpod.net    (RunPod)
        ◄── photo.glb ─────────────────   https://<random>.trycloudflare.com      (vast.ai)
                                            └ kami-server (jobs API)
                                              ├ TRELLIS.2   image -> GLB
                                              └ diffusers   prompt -> PNG
```

kami is a sibling of [ronin](../ronin), which does the same for LLMs. It shares ronin's
pod lifecycle, the in-pod cost guard, both providers and the cost reporting.

### Why kami

- **No idle spend.** A guard runs *inside* the pod and deletes it after a time limit, or
  when no job has run for a while, even with your PC asleep or off. `kami down` ends a
  session at once.
- **One command per result.** `kami mesh photo.png --up` brings the pod up, makes the
  mesh and saves it next to your other results.
- **Long jobs, short requests.** A mesh takes from tens of seconds to minutes. kami submits
  a job and polls it, so the RunPod proxy's ~100 s request limit never cuts one off.
- **Checks before billing.** Gated Hugging Face models, GPU memory and the server image
  are checked before a pod is created.
- **Standard library only** on your PC. Everything heavy lives in the pod's image.

### Contents

[Install](#install) · [Build the server image](#build-the-server-image) ·
[Quick start](#quick-start) · [Commands](#commands) · [Profiles](#profiles) ·
[Cost guard](#cost-guard) · [Licenses of the models](#licenses-of-the-models) ·
[How it works](#how-it-works) · [Tests](#tests)

## Install

```powershell
pip install -e .
kami init      # creates kami.toml and .env here if missing
kami doctor    # checks config, API key, server image, model access, GPU stock
```

Requires Python 3.11+ and has no external dependencies.

Put these in `.env`:

- `RUNPOD_API_KEY` (or `VAST_API_KEY` with `provider = "vast"`).
- `HF_TOKEN`: a Hugging Face token. TRELLIS.2 loads two **gated** models, so with that
  token's account accept their terms first:
  [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  and [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0). `kami doctor` tells you
  which ones are still missing.

## Build the server image

The pod runs a container image with PyTorch, TRELLIS.2's CUDA extensions and diffusers.
It is built once from [`docker/Dockerfile`](docker/Dockerfile) and pushed to a registry
the providers can pull from. The build needs no GPU, but compiling the extensions takes
30–60 min and the image is ~15 GB.

**With GitHub Actions** (no local Docker needed): push this repo to GitHub, then run
*Actions → server image → Run workflow*. It pushes `ghcr.io/<you>/kami-server:<tag>`.
Make the package public in its settings, or the pods cannot pull it.

**Locally:**

```powershell
docker build -t ghcr.io/<you>/kami-server:0.1.0 docker
docker push ghcr.io/<you>/kami-server:0.1.0
```

Then set the image in `kami.toml`:

```toml
[defaults]
container = "ghcr.io/<you>/kami-server:0.1.0"
```

kami sends `kami/server.py` and `kami/guard.py` to the pod on every `kami up`, so a
change to them needs no rebuild. Only a change of dependencies does.

The image is built for CUDA 12.4 (TRELLIS.2's tested stack) and Ampere, Ada and Hopper
GPUs. Blackwell cards (RTX 5090, RTX PRO 6000, B200) need a CUDA 12.8 rebuild.

## Quick start

```powershell
kami prices --hours 1             # what each profile costs now, all in
kami up                           # bring up the default profile (mesh)
kami mesh photo.png               # -> kami-out/photo.glb
kami down                         # delete the pod and stop paying
```

Or let `mesh` / `image` bring the pod up themselves:

```powershell
kami mesh chair.jpg lamp.png --up                     # several images, one pod
kami up -p studio                                     # both models on one 48 GB card
kami image "a ceramic owl, studio light" -n 4 --seed 7
kami mesh --prompt "a ceramic owl, studio light"      # text -> image -> mesh
```

Results go to `output_dir` (`kami-out/` by default, or `-o DIR`). kami never overwrites a
file: a second `photo.glb` becomes `photo-2.glb`.

Images can be PNG, JPEG, WebP, AVIF, HEIC (phone photos, turned as their EXIF says),
TIFF, BMP or GIF. The background is removed for you; a picture that is already cut out
(a transparent background) is used as it is.

### Several views of one object

By default every image is its own mesh. `--views` makes **one** mesh out of several
pictures of the same object (front, side, back...), so its back is seen rather than
invented:

```powershell
kami mesh --views front.png side.png back.png                     # TRELLIS.2 (up to 8)
kami mesh --views a.png b.png --view-mode multidiffusion           # slower, often steadier
kami mesh front=a.png left=b.png back=c.png -p hunyuan-mv         # Hunyuan3D-2mv
```

- **TRELLIS.2** was trained on single images. kami adds the tuning-free trick of TRELLIS
  (1)'s `run_multi_image`: the sampler sees one view per step (`stochastic`, default)
  or the average of all of them (`multidiffusion`, as many times slower as views). It is
  **experimental**: views with the same light, size and centring help; mismatched ones
  make it worse.
- **Hunyuan3D-2mv** is trained for it, with named views: `front`, `left`, `back`, `right`
  (`VIEW=IMAGE`; unnamed ones take the free names in that order). Its texture is painted
  from the front view. See the licence note below before using it.

`kami ui` has the same switch ("One object: the images are views of it").

**`kami ui`** does all of this from a web page on your PC: drop images to make meshes,
write prompts to make images (or meshes from text), follow every job on the pod (its stage
and progress, the queue, also the ones sent from a terminal), and browse every
result, with meshes in an interactive 3D viewer (rotate, zoom, full size) next to the
image they came from. It shows the pod's cost and the guard's deadline, and can delete
the pod. It never starts one: run `kami up` first (browsing needs no pod).

The page is served on 127.0.0.1 only, and only that page can act: other sites cannot
send jobs or delete the pod. kami relays the jobs, so the pod's key never reaches the
browser. The 3D viewer ([model-viewer](https://modelviewer.dev)) loads from a CDN.
Blender opens the GLBs too.

If the cloud has no stock, `kami up` **waits** for it (nothing is billed while waiting).
`--secure` also tries secure cloud, `--max-wait MINUTES` gives up.

## Commands

| Command | What it does |
|---|---|
| `kami init` | Create `kami.toml` and `.env` in this directory |
| `kami doctor` | Check config, API key, server image, gated models, GPU memory and stock |
| `kami profiles` | List the profiles and their models |
| `kami gpus` | GPUs with stock, price and datacenters (`--min-vram`, `--secure`) |
| `kami prices [PROFILE...]` | What each profile costs now: GPU, disk, download, total for `--hours` |
| `kami up [-p PROFILE]` | Create the pod and wait until the models are loaded (`--budget`, `--max-hours`, `--idle`) |
| `kami mesh IMAGE...` | Image(s) → textured GLB (`--seed`, `--texture-size`, `--decimation-target`, `--no-remesh`; `--views` for one mesh from several views, `--view-mode`) |
| `kami mesh --prompt TEXT` | Text → image → GLB (needs a profile with both models) |
| `kami image PROMPT` | Text → PNG (`-n`, `--seed`, `--steps`, `--guidance`, `--width`, `--height`) |
| `kami ui [FOLDER]` | Web page on this PC: make meshes and images on the pod, browse the results (`--port`, `--no-browser`) |
| `kami jobs` | The pod's jobs, whoever sent them: stage and progress of the running one, the queue, the last ones (`--watch`, `--json`) |
| `kami reload` | Send this kami's server code to the running pod and restart the server: no new pod, no download (`--force` with jobs running) |
| `kami status` | Pod state, cost so far, guard, jobs done (`--json`; exit 0 ready, 2 not ready, 3 no pod) |
| `kami logs` | Stream the pod logs (`--no-follow`, `-n`) |
| `kami down` | Delete the pod (a network volume is kept) |
| `kami cost` | What the provider billed per pod, and the balance |
| `kami watchdog` | A local guard that deletes the pod after `--max-hours` of uptime |
| `kami volume ls / create / rm` | RunPod network volumes |

`mesh` and `image` take `--up` (bring a pod up if none is active), `-p PROFILE` (with
`--up`), `-o DIR`, and the stock options of `up`.

## Profiles

A profile names the models the pod loads and the GPUs to try, in order:

| Profile | Models | GPUs | For |
|---|---|---|---|
| `mesh` (default) | TRELLIS.2-4B | 24 GB+ (RTX 4090, L40S, A6000...) | image → mesh |
| `image` | Z-Image-Turbo | 24 GB (RTX 4090, A5000...) | text → image |
| `studio` | both | 48 GB (L40S, A6000...) | `mesh --prompt`, mixing both |
| `studio-24` | both, `offload = true` | 24 GB | the same, cheaper; each switch takes seconds |

```toml
[profiles.mesh]
mesh_model = "microsoft/TRELLIS.2-4B"        # a TRELLIS.2 pipeline
gpus = ["RTX 4090", "L40S", "RTX A6000"]
min_vram_gb = 24                             # GPUs below this are skipped
disk_gb = 60                                 # container disk: the weights must fit
mesh_defaults = { texture_size = 2048, decimation_target = 500000, remesh = true }

[profiles.image]
image_model = "Tongyi-MAI/Z-Image-Turbo"     # any diffusers text-to-image pipeline
gpus = ["RTX 4090"]
image_defaults = { steps = 8, guidance = 0.0, width = 1024, height = 1024 }
```

Other keys: `extends` (inherit a profile and override keys), `offload` (keep the model not
in use on the CPU), `min_ram_gb` (only hosts with this much system RAM per GPU: TRELLIS.2
keeps its models in RAM, and a 29 GB host ran out of memory), `dino_model` / `rembg_model` (replace the DINOv3 model or the
background remover TRELLIS.2 loads, see below), `extra_models` (more repos to check access to), `gpu_count`,
`container` (a per-profile image), `min_cuda`. `kami.toml` is checked as a whole: a typo
is reported with a suggestion before anything is created.

**DINOv3 from Meta's download instead of Hugging Face.** If Meta approved your DINOv3
request by email (direct `.pth` links) but Hugging Face's gated repo is not open to you,
convert the ViT-L checkpoint once and keep it in a private Hugging Face repo of your own
with [`tools/convert_dinov3.py`](tools/convert_dinov3.py) (CPU, needs torch; it checks the
result against Meta's reference outputs). Then:

```toml
[profiles.mesh]
dino_model = "<you>/dinov3-vitl16"       # your private repo; HF_TOKEN must read it
```

`rembg_model` works the same way for the background remover, e.g.
`"ZhengPeng7/BiRefNet"` (MIT, not gated) instead of RMBG-2.0.

Any diffusers pipeline works as `image_model` if the image has what it needs. FLUX.1-schnell
(Apache-2.0) wants a 48 GB card or `offload`. Gated models such as FLUX.1-dev need their
terms accepted, and have their own licenses.

## Cost guard

Every pod runs under a guard that deletes it **from inside**, through the provider's API:

- after `max_hours` of uptime (4 by default, `--max-hours`), and
- after `idle_minutes` without a job (30 by default, `--idle`).

It works with your PC off. `--budget USD` sets the deadline from today's price.
`kami status` shows when the guard will act and what the session has cost so far.
`kami down` prints the session's cost. `kami cost` shows what the provider billed (RunPod
publishes it about an hour late). `kami status` and `kami doctor` also warn about pods
named `kami-*` that this project does not know.

**Storage.** `storage = "ephemeral"` (default) downloads the weights on every boot. That
is ~10–30 GB per profile, a few minutes, with no fixed cost. `storage = "volume"` keeps
them on a RunPod network volume (~$0.07/GB/month), so boots are fast, but the pod is
pinned to the volume's datacenter.

## Licenses of the models

kami itself is MIT. The models have their own terms. Check them for your use:

- **TRELLIS.2** (code and weights): MIT.
- **RMBG-2.0** (background removal, loaded by TRELLIS.2): **CC BY-NC 4.0, non-commercial**.
  For commercial use, BRIA sells a license, or remove the background yourself and adapt
  the pipeline.
- **DINOv3** (image features, loaded by TRELLIS.2): Meta's DINOv3 license.
- **Z-Image-Turbo**: Apache-2.0.
- **Hunyuan3D-2 / -2mv** (optional, `hunyuan*` profiles): Tencent Hunyuan Community
  License, which **does not apply in the European Union, the United Kingdom and South
  Korea** ([LICENSE](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE)). kami
  refuses a profile with a Hunyuan3D-2 model until it says `accept_hunyuan_license = true`:
  set it only if the licence covers you. Those profiles run on the `-hunyuan` image
  (`docker/Dockerfile.hunyuan`, the workflow's `variant = hunyuan`), and are commented
  out in this repo's `kami.toml`.

## How it works

- `kami up` creates the pod with the `container` image. Its command runs the guard
  (`kami/guard.py`), which runs `kami/server.py`. Both are sent compressed in the pod
  definition, so they never go stale in the image.
- The server loads the profile's pipelines. `/health` answers 200 once they are loaded,
  and while they load, `kami up` shows the download and phase.
- `kami mesh` / `kami image` POST a job to `/v1/jobs` and poll `/v1/jobs/<id>`, then
  download its files. One worker runs the jobs in order on the GPU. Every request carries
  a per-pod API key that kami generates on `up`.
- The guard reads the server's `/metrics` (jobs running, queued, finished) for the idle
  limit. On vast.ai it also opens a Cloudflare quick tunnel, so traffic goes over HTTPS.

## Tests

```powershell
python -m unittest
```

The tests mock every provider and the GPU. Nothing reaches a live API or needs torch.
