# CLAUDE.md

## What this project is

`kami` is a small Python CLI (Python 3.11+) that manages the lifecycle of a cloud GPU pod
on [RunPod](https://runpod.io) or [vast.ai](https://vast.ai) (chosen by `provider` in
`kami.toml`) which runs **kami-server**: Pixal3D or TRELLIS.2 for image → textured 3D
mesh (GLB), and diffusers pipelines for text → image and for the views of an object. The
CLI sends jobs to the pod and saves the results locally; `kami print` then turns a mesh
into a 3MF for a multicolour printer, on the PC. What it is for: figures to 3D print,
among them flexi models whose joints are cut afterwards in `../jointwright`.

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
  turning around it; Qwen-Image-Edit gets true CFG, FLUX.2 guidance. `FRONT_HINT`, what
  such a job adds to its front picture, is the default of the `front_hint` parameter: a
  job or `views_defaults` may replace it, "" drops it.
  Stdlib HTTP; torch / trellis2 / diffusers are imported lazily, so the tests import it
  without them. kami never imports it at runtime.
- `kami/jobs.py` — client side of the jobs API: submit, poll with a progress line, download
  (never overwriting). `PRINT_PRESETS` (figure, flexi, lowpoly, bust, relief, natural:
  label, purpose, text; `print_hint(text, preset)` is the one in force: `--preset`, then
  kami.toml's `print_hint`, then its `print_preset`, then figure; written in the positive,
  and flexi names neither joints nor a base, since jointwright cuts the joints later).
  `PRINT_HINT` / `printable`: the hint every prompt gets on the client (CLI
  `print_prompt`, page `webui._sent`) so the picture is printable; `PRINT_NEGATIVES`, what
  each preset keeps out. On by default; `--as-written`, `print_hint` in kami.toml (a
  custom one may say `{colors}`). File names and labels keep the prompt as written.
  `patiently` keeps asking a job's state and files while the pod cannot be reached
  (`Unreachable`), for `UNREACHABLE_PATIENCE`: the job goes on there meanwhile.
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
- `kami/webui.py` + `kami/webui.html` — `kami ui`: a page on 127.0.0.1 in four stages,
  Concept (text -> front pictures; image jobs, or a views job in one go), Views (a front
  picture, from the stage or dropped / pasted, -> its sides; views jobs), Mesh (views in
  named slots -> GLB) and Print (`printing` on this PC, painted preview), opening on the
  first stage the pod's `/v1/info` says it serves. kami relays the jobs (the pod key stays in kami);
  the page never starts a pod, and a stage the pod lacks shows the `kami up -p` for it
  (`WebBackend.status()["profiles"]`). `objects()` is the library: `output_dir` and one
  folder deep, grouped by name (`VIEW_OF`: owl-left-2.png is the newest left view of
  "owl"), leaving out names that start with `_` or `.` (`kami-out/_sheets/` holds the
  comparison sheets); `resolve()` is the only way a name becomes a file. The stage tabs
  say where a stage can run (a dot: this pod) and tick what the object on the stage
  already has. The negative prompt is sent only where `/v1/info`'s `image_negative` (or,
  from an older pod, the model's defaults) says it is read. The print preview has the
  256 mm plate under the model (`preview_glb(plate=)`), sized by its longest side unless
  told `fit="height"`; the viewer starts three-quarter from above, which is not always
  the model's front (the mesh models do not agree on which way a mesh faces). Pictures of a job may be
  uploads or `{file}` from the library. `Printer`: POST /api/print (`prepare` cached, so
  colours answer at once; `save` writes the 3MF), /preview/<id>.glb, /thumbs/<mesh> (a
  drawn PNG for meshes with no picture). The page builds its DOM from JSON with
  `textContent`, never HTML from file names. Every text added to a prompt is shown and
  editable there (`hint`, `front_hint` in a request; "Sent to the pod" is the result),
  with the models' and `kami print`'s parameters, in each stage's settings dialog
  (`settings()`: a gear beside the form's title, a `<dialog>` inside the form so its
  fields stay the form's; `group()` sections, `field(label, input, help)` with an (i)
  from `HELP`). The page keeps them in localStorage (`kami.concept`, `kami.print`). Host header checked; POSTs need the page's
  token, JSON and same Origin. `cli.WebBackend` gives it the pod status, target, `down`.
- `kami/watchdog.py`, `kami/ui.py`, `kami/mascot.py` — local guard, console output, mascot.
- `docker/Dockerfile.pixal3d` — Pixal3D + NAF (pinned) + natten 0.17.5 wheel on top of the base.
- `docker/Dockerfile` — the kami-server image (torch 2.6 / CUDA 12.4 + TRELLIS.2 + diffusers);
  `.github/workflows/server-image.yml` builds and pushes it to ghcr.io.
- `tools/convert_dinov3.py` — one-off helper (needs torch, not part of the package): Meta's
  DINOv3 `.pth` → transformers format → a private HF repo, for `dino_model`.
- `tools/prompt_grid.py`, `tools/compare_meshes.py` — how wordings and mesh models are
  compared: the same subjects and seeds through each wording on the pod that is up, as
  one sheet; several GLBs side by side from above and four sides.
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

## What has run, and what has not

The client is unit-tested with mocks. Everything below ran on real RunPod pods unless it
says otherwise; dates are when it was last seen.

**The pod and the image.** The Docker image, `kami reload`, `offload` between two image
models (the `views` pod), boot progress and stock waiting. `kami up` takes 4 to 16
minutes depending on the weights and the day (`mesh`, 48 GB of them: 7 to 16). Images
go through `server.open_image` (Pillow; pillow-heif for HEIC; OpenCV if Pillow fails;
EXIF turned; almost opaque alpha dropped so the background is removed).

**Fronts** (2026-11-20; 152 pictures, the same three subjects and two seeds through each
model and wording; sheets in `kami-out/prompt-tests/sheets/`; `tools/prompt_grid.py` is
how). What matters first is that the subject stays itself.
- Z-Image-Turbo keeps the subject and ignores pose and camera, and has no negative prompt.
- The full Z-Image (~32 s a picture on an A100) obeys pose and reads the negative prompt,
  but draws the hint's shape words over the subject (a snake as a sausage) and takes
  colour words at their value (blocks of colour, or unpainted clay).
- FLUX.2 [dev] 4-bit as `image_model` (`fronts-flux`, ~62 s a picture on an L40S) kept
  every subject recognisable, facing the camera, in its own colours, with no base.
- None of them lays a snake in a straight line. Gemini, by hand, did, and followed
  "no antennae": its pictures were the best of the day.
- A prompt draws what it names, also after "no" or "without". A preset is short, leans on
  a style the model knows, says nothing of anatomy and does not count colours. Judging a
  wording by one picture got three presets wrong in a row: compare on a grid.
- Presets: `flexi` is the stylised wording as tried on FLUX.2; `figure` (the same style,
  standing) and lowpoly, bust, relief, natural have never run. Their negative prompts are
  read only by a model with guidance (the full Z-Image).

**Views.** `kami views` with Qwen-Image-Edit and fal's LoRA: coherent when the front is a
front (an owl, a gecko); a front that came out as a profile or from above was turned
wrongly. Diagonal views never tried. `views-flux` (FLUX.2 turning the camera): its LoRA
barely acts, left and back came out as the front.

**Meshes** (2026-11-20; `kami-out/gemini/mesh-comparison.png`, `tools/compare_meshes.py`).
- Pixal3D places each named view at a fixed camera (object height, 90 degrees apart).
  With Qwen's posed views it gave a coherent owl. With pictures drawn elsewhere (Gemini)
  it doubled what did not fit: a gecko whose front and back were drawn from above came
  out with repeated features; retaken at ground level, cleaner. From one picture
  (three-quarter front) it was clean. Its meshes are not closed. At 1536 with four views
  it ran out of memory on 24 GB and on an L40S (45 GiB): `PIXAL_LOW_VRAM_BELOW_GIB = 60`.
- TRELLIS.2 with `--views` assumes no camera, and gave the best gecko and caterpillar from
  the same four pictures. Of its modes, multidiffusion kept painted details (spots, an
  eye) sharp where stochastic blurred them; it is the default, on that one comparison.
- So: views from `kami views` -> Pixal3D (`mesh`); pictures from anywhere -> `trellis`,
  or one picture.
- Textures: the base colour comes out a little less saturated than the pictures (a lime
  (168,198,3) as (150,151,14)), and the models also guess a metallic value (~0.4 on a
  toy), which a viewer shows as a dull, dark surface. It is not an sRGB mix-up. The
  page's viewer sets metallic to 0, since only the base colour is printed.

**`kami print`.** A Pixal3D owl (a double shell) and a TRELLIS.2 duck (non-manifold
edges, so rebuilt) become closed solids; Bambu Studio 2.9.1's command line loads both
3MFs with nothing to repair and keeps the paint. A Gemini caterpillar went from four
pictures to a closed 3MF in three colours.

**Not done yet.**
- A 3MF opened in Bambu Studio's window, sliced, or printed; OrcaSlicer.
- Views or a mesh from a FLUX.2 front; a job sent from `kami ui` to a real pod (its
  request bodies are unit-tested, and the page was looked at with pretend pods).
- Text-to-mesh, Hunyuan3D-2 (its image builds and loads; its first jobs failed in
  pymeshlab for a missing library, since added; never run again), a Hunyuan3D-2mv mesh.
- The guard deleting the pod through GraphQL (a pod-scoped key got 403 once).
- A hollow skin with an opening wider than a cube in `rebuild` (only built in a test).
- A model's front does not always face the plate's front (Pixal3D and TRELLIS.2 differ).
- `kami init`'s template in config.py still writes one TRELLIS.2 profile called `mesh`;
  the shipped `kami.toml` has `views`, `mesh` (Pixal3D, the default), `fronts-flux` and
  `trellis`, with image, studio, studio-24, hunyuan, hunyuan-mv and views-flux commented.

## Working notes

- The user often writes in Spanish; reply in the language they use.
- Target OS is Windows (PowerShell is the primary shell), so prefer cross-platform code
  and test PowerShell-facing output paths.
