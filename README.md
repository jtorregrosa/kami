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
  (`kami print`, which runs on your PC, is the one command with dependencies of its own.)

### Contents

[Install](#install) · [Build the server image](#build-the-server-image) ·
[Quick start](#quick-start) · [Print it](#print-it) · [Commands](#commands) · [Profiles](#profiles) ·
[Cost guard](#cost-guard) · [Licenses of the models](#licenses-of-the-models) ·
[How it works](#how-it-works) · [Tests](#tests)

## Install

```powershell
pip install -e .
kami init      # creates kami.toml and .env here if missing
kami doctor    # checks config, API key, server image, model access, GPU stock
```

Requires Python 3.11+ and has no external dependencies. Only `kami print` needs some:
`pip install -e ".[print]"` (numpy, Pillow, scipy and scikit-image).

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
kami up -p studio                                     # both models on one 48 GB card (commented out in kami.toml)
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
  (1)'s `run_multi_image`: the sampler sees the average of all the views at every step
  (`multidiffusion`, the default: it keeps small painted details sharp) or one view per
  step (`stochastic`). It assumes no camera for each view, so it takes pictures drawn
  anywhere (`kami up -p trellis`); views with the same light, size and centring still
  help.
- **Hunyuan3D-2mv** is trained for it, with named views: `front`, `left`, `back`, `right`
  (`VIEW=IMAGE`; unnamed ones take the free names in that order). Its texture is painted
  from the front view. See the licence note below before using it.
- **Pixal3D** (`-p mesh`, TencentARC, MIT) is TRELLIS.2 with pixel-aligned
  conditioning and weights trained for posed views: name them by where the camera is
  (`front`, `left`, `back`, `right`, `front-left`, `back-right`...; one must be `front`,
  unnamed ones take the free names in that order). The views are cut out and framed
  alike before it sees them. A single picture works too (`fov` in `mesh_defaults`, 20
  degrees by default, is the camera it assumes). Runs on the `-pixal3d` image. On a
  4090 it runs in its low-VRAM mode at resolution 1024 (~2.5 minutes a mesh); cards of
  40 GB or more get 1536. With a concept's four views from `kami views` it keeps the
  object's proportions; from the front alone it has to invent the back.

### Views from a concept

No photos of the back? `kami views` makes them: a front picture of the concept, then an
image-editing model with a camera LoRA turns the camera around it. The pictures it saves
(`owl-front.png`, `owl-left.png`, `owl-back.png`, `owl-right.png`) go straight into
`kami mesh` on a mesh pod (it prints the command):

```powershell
kami up -p views
kami views "a ceramic owl" --seed 7                    # front + left, back, right
kami views --front owl.png                             # start from your own front picture
kami views "a robot" --angles left back right front-left back-right
kami down
kami up -p mesh
kami mesh front=kami-out/a-ceramic-owl-front.png left=kami-out/a-ceramic-owl-left.png back=kami-out/a-ceramic-owl-back.png right=kami-out/a-ceramic-owl-right.png
```

Two profiles, to compare which turns the camera better:

| Profile | Front picture | Turns the camera | GPU |
|---|---|---|---|
| `views` | Z-Image-Turbo | Qwen-Image-Edit-2511 + fal's Multiple-Angles LoRA (96 poses) | 80 GB |
| `views-flux` | FLUX.2 [dev] | FLUX.2 [dev] (4-bit) + lovis93's Multi-Angles LoRA v2 (72 poses) | 48 GB |

`views` is the one that works: on a test concept its four views kept the same
object (shape, glaze, proportions) and gave it a plausible back, in ~5 minutes on an
A100. `views-flux` made a finer front, but its camera LoRA barely acts through diffusers
(left and back came out as the front), so it is kept only to try other LoRAs.

Views are named as Hunyuan3D-2mv names them, by where the camera goes: `left` is the
picture taken from your left, with the object facing the left of the picture (the
LoRAs would call it its right side; kami asks them for that).

`--elevation` and `--distance` take the LoRA's own words ("eye-level shot", "high-angle
shot"; "medium shot", "wide shot"). The view the model invents (the back above all) may
not match the front: look at them before making the mesh.

**`kami ui`** does all of this from a web page on your PC, in four stages:

- **Concept**: a description becomes pictures of the object's front, a few at a time to
  pick from ("Straight to views" skips the picking).
- **Views**: a picture of the front becomes the object seen from its sides. The picture is
  the one on the stage, or one of your own from anywhere: drop it, paste it or choose it.
  Each view can be made again on its own.
- **Mesh**: the views go into slots named after the side they show (or one mesh per
  picture, or straight from text), with the options of the pod's mesh model, and the
  result opens in an interactive 3D viewer.
- **Print**: `kami print` with a painted preview. Height, flat base, how many filaments or
  the colours of your own; changing colours redraws at once. It runs on your PC, so it
  works with no pod.

The page opens on the first stage the active pod serves (a `views` pod on Concept, a
`mesh` pod on Mesh, no pod on Print), and the header says what the pod is, what it
costs, how much it has cost and when the guard deletes it. A stage the pod does not
serve says which `kami up -p ...` gets you there; the page never starts a pod itself.
Each stage keeps its form short and has a **Settings** button (the gear beside its
title) that opens a dialog with the rest, in sections, every setting with an (i) that
explains it. Concept's has the texts kami adds to your description (the printing hint and
the words that ask for a front picture), both to rewrite, with the prompt exactly as the
pod will get it, and every parameter of each model (steps, guidance, size, negative
prompt; the camera LoRA's weight, height and distance) with the pod's defaults shown.
Print's has the rest of `kami print` (rebuild, detail, border clean-up, smallest patch,
keeping the surfaces inside). The page remembers how you left it, in the browser; "Back
to the defaults" forgets it.

Below, the jobs on the pod (stage and progress, the queue, also those sent from a
terminal) and the **library**: everything in `output_dir` and the folders directly inside
it, grouped by object, so `owl-front.png`, `owl-left.png`, `owl-front.glb` and
`owl-front.3mf` are one card. Click one to put it on the stage.

The page is served on 127.0.0.1 only, and only that page can act: other sites cannot
send jobs or delete the pod. kami relays the jobs, so the pod's key never reaches the
browser. The 3D viewer ([model-viewer](https://modelviewer.dev)) loads from a CDN.
Blender opens the GLBs too.

If the cloud has no stock, `kami up` **waits** for it (nothing is billed while waiting).
`--secure` also tries secure cloud, `--max-wait MINUTES` gives up.

## Print it

**Prompts are written for printing.** `kami image`, `kami views` and `kami mesh --prompt`
add a hint after your prompt, so the picture is of something a printer can make and
`kami print` can colour: a figure standing on a wide flat base, one thick piece with
nothing loose, a few flat colours in clearly separated areas, matte and evenly lit (a
highlight painted into the texture would become a filament). `--colors N` says how many
colours it asks for (4, as `kami print`), `--as-written` sends your prompt alone, and
`print_hint` in `kami.toml` replaces the hint (`{colors}` is the number) or, as `""`,
turns it off.

That hint is a **preset**, one per kind of print; `--preset NAME` picks it for a command,
`print_preset` in `kami.toml` for all of them, and `kami ui` has them in a list:

| Preset | The picture it asks for |
|---|---|
| `figure` (default) | A stylised figurine, simplified but true to its subject, standing on its own wide flat base |
| `flexi` | A stylised animal lying on its belly directly on the ground, body in a straight line and head down: for flexible joints cut into the mesh afterwards. It names no base and no joints, so none are drawn |
| `lowpoly` | A faceted sculpture on a flat base: large flat triangles, one colour each |
| `bust` | Head and shoulders, cut straight across the chest into a flat base |
| `relief` | The subject in raised relief on a thick flat-backed medallion, one colour per level |
| `natural` | The object as it is, in one piece: only matte and evenly lit, with no limit of colours |

None of them counts colours: the picture keeps the subject's own, and `kami print`
reduces them to your filaments afterwards (counted in a prompt, they came out as blocks).
They say what is wanted, never what is not: naming a thing in a prompt ("no base")
tends to draw it. What is not wanted goes in the preset's negative prompt instead (shown
in `kami ui`'s settings; `--negative` on a command), which a model running with guidance
reads. The same goes for your own
description: "a caterpillar without antennae" comes out with antennae, and kami warns
when a prompt has a "no", "not" or "without" in it. Say what you want in its place ("a
caterpillar with a smooth round head"). A picture you bring yourself (`kami mesh photo.png`, `kami views --front`)
is used as it is. The model follows the hint as far as it follows any prompt.

A mesh from `kami mesh` is made to be looked at: a paper-thin double shell (the surface,
and the same surface again inside it), one unit tall, Y up, with a texture. `kami print`
turns the GLB into a 3MF for a multicolour printer. It runs on your PC: no pod, nothing
billed.

```powershell
kami print kami-out\owl.glb                              # owl.3mf next to it: 80 mm tall, 4 colours
kami print owl.glb --height 60 --colors 2                # smaller, two filaments
kami print owl.glb --palette "#1a4fa0,#ffffff,#202020"   # the filaments you have, in slot order
```

```
  › owl.glb -> 3MF
  ✓ owl.3mf
    size     40.6 x 37.6 x 60.0 mm · 251,064 triangles
    solid    closed · 1 inner shell, 4 slivers removed
    base     367 mm² flat on the plate
    slot 1   #455A74  77% of the surface
    slot 2   #909396  23% of the surface
```

What it does:

- **A solid.** Vertices split along texture seams are welded, and the inner shells and
  loose slivers are dropped. If that leaves a closed body, the mesh is kept as it is,
  triangle for triangle.
- **Rebuilt, when it does not.** A mesh that is still open, or still a hollow skin, gets a
  new surface drawn around everything it encloses: its triangles are marked on a grid of
  0.3 mm cubes (`--detail`), the outside is flooded from a corner, and the surface is
  drawn where the old outer one was, with its colours. The result is always closed;
  detail finer than a cube is lost, and an opening wider than a cube still lets the
  flood in (the report then says it is a hollow skin). `--rebuild always` does it to any
  mesh, `--rebuild never` writes the mesh as cleaned and leaves the repair to the slicer.
- **On the plate.** Z up, scaled to `--height` mm, and the lowest `--base` percent of it
  (2 by default) pressed flat so it stands.
- **A few colours.** Every triangle takes the texture's colour under it; those are reduced
  to `--colors` (4, one AMS), the one covering most surface in slot 1. Light and dark
  count less than hue, since a texture has the light baked into it and a shadow is not
  another filament. With `--palette` each triangle takes the closest of your filaments
  instead, in the slot order you wrote. `--smooth` passes clean the borders, and patches
  under `--min-patch` mm² (too small to print, and each one a filament change) join
  their surroundings.
- **The 3MF** says the colours twice: as Bambu Studio / OrcaSlicer paint (one filament
  slot per triangle) and as standard 3MF materials, for other slicers.

Open it in Bambu Studio and give filaments 1 to N the colours `kami print` lists (the
3MF carries the paint, not your filaments). Fewer colours print faster and waste less in
purges: try `--colors 2` or `3` before 4. The mesh keeps every triangle, so the file is
some MB.

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
| `kami image PROMPT` | Text → PNG, with the printing hint (`-n`, `--seed`, `--steps`, `--guidance`, `--width`, `--height`, `--preset`, `--colors`, `--as-written`) |
| `kami views CONCEPT` | Concept → front, left, back, right pictures of it, for `kami mesh` with views (`--front IMAGE`, `--angles`, `--seed`, `--steps`, `--guidance`, `--lora-scale`, `--elevation`, `--distance`, `--preset`, `--colors`, `--as-written`) |
| `kami print MESH...` | GLB → 3MF to print in colours, on this PC: closed solid, flat base, mm, the texture as a few filaments (`--height`, `--base`, `--colors`, `--palette`, `--rebuild`, `--detail`, `--smooth`, `--min-patch`, `--keep-cavities`, `-o`) |
| `kami ui [FOLDER]` | Web page on this PC: Concept → Views → Mesh → Print, opening on what the pod serves; the library by object (`--port`, `--no-browser`) |
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

A profile names the models the pod loads and the GPUs to try, in order. `kami.toml`
has two, one per pod of the way from a concept to a print:

| Profile | Models | GPUs | For |
|---|---|---|---|
| `fronts-flux` | FLUX.2 [dev], 4-bit (non-commercial licence) | 48 GB (L40S, A6000...) | text → pictures of the object's front |
| `views` | Qwen-Image-Edit-2511 (+ Z-Image-Turbo) | 80 GB (A100, H100) | a picture of the front → its sides |
| `mesh` (default) | Pixal3D | 24 GB+ (RTX 4090, L40S, A6000...) | those pictures → mesh |

The front decides everything after it, so it gets the model that draws it best. Three
were compared on the same subjects and seeds (152 pictures): Z-Image-Turbo keeps the
subject but ignores pose and camera; the full Z-Image obeys them but draws the hint over
the subject (a snake as a sausage); FLUX.2 keeps a snake a snake, lying down, facing the
camera, in its own colours. FLUX.2 does not fit on the card next to Qwen-Image-Edit, so
it is a pod of its own: fronts on `fronts-flux` (about a minute each), then `views` to
turn the one you chose. `views` can still do both in one go with Z-Image-Turbo
(`kami views "a ceramic owl"`), quickly and less faithfully to the pose.

The others it can run are in `kami.toml` too, commented out; uncomment a block to use it:

| Profile | Models | GPUs | For |
|---|---|---|---|
| `trellis` | TRELLIS.2-4B | 24 GB+ | image → mesh, the model Pixal3D builds on |
| `image` | Z-Image-Turbo | 24 GB (RTX 4090, A5000...) | text → image alone, cheaply |
| `studio` | TRELLIS.2 + Z-Image-Turbo | 48 GB (L40S, A6000...) | `mesh --prompt`, mixing both |
| `studio-24` | the same, `offload = true` | 24 GB | cheaper; each switch takes seconds |
| `hunyuan`, `hunyuan-mv` | Hunyuan3D-2 | 24 GB+ | meshes; licence excludes the EU, UK and South Korea |
| `views-flux` | FLUX.2 [dev] | 48 GB | views; its camera LoRA barely acts |

The examples below that name one of these (`-p studio`, `-p hunyuan-mv`) need it uncommented.

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
  (`docker/Dockerfile.hunyuan`, the workflow's `variant = hunyuan`).
- **Pixal3D** (`mesh`): MIT; it loads DINOv3 and RMBG-2.0 as TRELLIS.2 does (their
  terms above), NAF (Apache-2.0) and natten (MIT).
- **Qwen-Image-Edit-2511** (`views`): Apache-2.0; fal's Multiple-Angles LoRA:
  Apache-2.0.
- **FLUX.2 [dev]** (`views-flux`): **FLUX [dev] Non-Commercial License**; lovis93's
  Multi-Angles LoRA v2: CreativeML OpenRAIL-M.

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
Those of `kami print` (`tests/test_printing.py`) are skipped without its dependencies.
