"""Compare wordings of a prompt on the pod that is up: the same subjects and seeds through
each wording, one picture each, then one sheet with a row per wording.

Judging a wording by one picture is how three presets were got wrong in a row: a seed
gives a blue snake whatever the words say. This is the way that worked (see CLAUDE.md,
"Front models compared"). Needs a pod with an image model (`kami up -p fronts-flux`),
and Pillow for the sheet. Edit VARIANTS, SUBJECTS and SEEDS, then:

    python tools/prompt_grid.py NAME          # pictures in kami-out/prompt-tests/NAME
    python tools/prompt_grid.py NAME --sheet  # only build the sheet again

Pictures already made are kept, so a run that stopped half-way goes on where it was.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kami import jobs  # noqa: E402

TESTS = ROOT / "kami-out" / "prompt-tests"
# what a views job adds to its front picture (server.FRONT_HINT): `kami image` does not, so it is added here
FRONT = "front view of the whole object, centred, full view, on a plain white background, soft even studio light"

# name -> (what follows the subject, the negative prompt or "")
VARIANTS = {
    "flexi": (jobs.PRINT_PRESETS["flexi"][2], jobs.preset_negative("flexi") or ""),
    "figure": (jobs.PRINT_PRESETS["figure"][2], jobs.preset_negative("figure") or ""),
}
SUBJECTS = ("snake", "caterpillar", "gecko")
SEEDS = (11, 12)
TILE = 300


def run(name: str) -> None:
    for variant, (hint, negative) in VARIANTS.items():
        for subject in SUBJECTS:
            for seed in SEEDS:
                folder = TESTS / name / f"{variant}__{subject.replace(' ', '-')}__{seed}"
                if list(folder.glob("*.png")):
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                args = [sys.executable, "-m", "kami", "image", f"{subject}, {hint}, {FRONT}", "--as-written",
                        "--seed", str(seed), "-o", str(folder)]
                if negative:
                    args += ["--negative", negative]
                began = time.time()
                done = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                print(variant, subject, seed, f"{time.time() - began:.0f} s",
                      "ok" if done.returncode == 0 else "FAILED " + (done.stderr or done.stdout)[-400:], flush=True)


def sheet(name: str) -> Path:
    from PIL import Image, ImageDraw
    cells = {}
    for folder in sorted((TESTS / name).iterdir()):
        pictures = sorted(folder.glob("*.png")) if folder.is_dir() else []
        if pictures:
            variant, subject, seed = folder.name.split("__")
            cells[(variant, f"{subject} {seed}")] = pictures[0]
    rows, cols = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
    label = 150
    out = Image.new("RGB", (label + TILE * len(cols), 24 + TILE * len(rows)), "white")
    draw = ImageDraw.Draw(out)
    for c, col in enumerate(cols):
        draw.text((label + c * TILE + 8, 6), col, fill="black")
    for r, row in enumerate(rows):
        draw.text((8, 24 + r * TILE + TILE // 2), row, fill="black")
        for c, col in enumerate(cols):
            if (row, col) in cells:
                out.paste(Image.open(cells[(row, col)]).convert("RGB").resize((TILE, TILE)), (label + c * TILE, 24 + r * TILE))
    target = TESTS / "sheets" / f"grid-{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.save(target)
    return target


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")     # kami's output has characters a Windows console lacks
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if "--sheet" not in sys.argv:
        run(sys.argv[1])
    print(sheet(sys.argv[1]))
