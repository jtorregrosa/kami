"""Several meshes side by side, a row each: from above, then from four sides, in their
base colour (what `kami print` reads), to compare models or settings on the same object.
Needs numpy and Pillow (the `print` extra).

    python tools/compare_meshes.py out.png "TRELLIS.2=a.glb" "Pixal3D=b.glb" ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kami import printing  # noqa: E402

TILE = 320


def turn(degrees: float, tilt: float = 20.0):
    a, t = np.radians(degrees), np.radians(tilt)
    spin = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    lean = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return lean @ spin


def view(points, normals, colors, matrix):
    """The surface as dots, seen down -Z after `matrix`; near dots over far ones."""
    q, n = points @ matrix.T, normals @ matrix.T
    seen = n[:, 2] > 0
    q, n, c = q[seen], n[seen], colors[seen]
    scale = (TILE / 2 - 8) / (np.linalg.norm(points, axis=1).max() * 1.02)
    order = np.argsort(q[:, 2])
    x = np.clip((q[order, 0] * scale + TILE / 2).astype(int), 0, TILE - 1)
    y = np.clip((TILE / 2 - q[order, 1] * scale).astype(int), 0, TILE - 1)
    image = np.full((TILE, TILE, 3), 255, np.uint8)
    image[y, x] = np.clip(c[order] * (0.62 + 0.38 * n[order, 2:3]), 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def row(label: str, path: Path) -> Image.Image:
    mesh = printing.load_glb(path)
    v = mesh.vertices - (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    tri = v[mesh.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    weights = np.random.default_rng(0).dirichlet((1, 1, 1), (len(mesh.faces), 5))
    points = np.einsum("fsk,fkc->fsc", weights, tri).reshape(-1, 3)
    normals, colors = np.repeat(normal, 5, axis=0), np.repeat(mesh.colors, 5, axis=0)
    tiles = [view(points, normals, colors, turn(0, 89))] + [view(points, normals, colors, turn(d)) for d in (0, 90, 180, 270)]
    out = Image.new("RGB", (TILE * len(tiles), TILE + 22), "white")
    ImageDraw.Draw(out).text((6, 4), f"{label}   ({len(mesh.faces):,} triangles)   from above | four sides", fill="black")
    for i, tile in enumerate(tiles):
        out.paste(tile, (i * TILE, 22))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    printing.require()
    rows = [row(*(arg.split("=", 1) if "=" in arg else (Path(arg).stem, arg))) for arg in sys.argv[2:]]
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(sys.argv[1])
    print(sys.argv[1], len(rows), "rows")
