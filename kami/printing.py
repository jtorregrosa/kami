"""`kami print`: a GLB turned into a 3MF a slicer prints in a few filament colours.

What the mesh models give is made to be looked at: TRELLIS.2 and Pixal3D export a thin
double shell (an outer surface and, inside it, the same surface facing inwards), Y up, one
unit tall, with a texture. A printer wants one closed solid, Z up, in millimetres, sitting
flat on the plate, and an AMS wants a handful of colours, not a texture. So:

  load_glb   the triangles and one colour per triangle (the texture under its centre)
  solid      vertices welded, the inner shells and loose slivers dropped
  place      Z up, the bottom flattened into a base, scaled to a height in mm
  rebuild    only when that is still not a closed body: a new surface around
             everything the old one encloses (the way ../jointwright imports models)
  quantize   the colours reduced to N (or matched to the filaments you have)
  smooth     specks a nozzle cannot print given to the colour around them
  write_3mf  a 3MF with the colours twice: Bambu Studio / OrcaSlicer paint
             (`paint_color`, one filament slot per triangle) and the standard 3MF
             materials other slicers and newer Bambu Studio read

This is the one part of kami that needs more than the standard library: numpy for the
arrays, Pillow for the texture, and scipy and scikit-image to rebuild a mesh. They are
imported here only, so every other command still runs without them.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import warnings as py_warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

from . import __version__
from .config import ConfigError

try:
    import numpy as np
except ImportError:                                  # pragma: no cover - depends on the PC
    np = None

MAX_COLORS = 16                  # four AMS units
WELD = 1e-6                      # vertices closer than this share of the mesh's size are one
FRAGMENT = 1e-5                  # pieces under this share of the surface are slivers
CRUMB = 0.01                     # after a rebuild, bodies under this share of the largest go
COLOR_BITS = 5                   # colours are binned to 32 levels a channel before k-means
LIGHTNESS = 0.5                  # how much light vs dark counts next to hue, when telling
                                 # colours apart: a texture has the light baked into it,
                                 # and a shadow on blue glaze is not another filament
DETAIL = 0.3                     # mm: the cubes a rebuilt surface is drawn on (a nozzle is 0.4)
MAX_CELLS = 24_000_000           # cubes in that grid at most; a bigger model gets bigger cubes
PAD = 3                          # empty cubes around the model, for the flood to go round it
BAND = 2.5                       # cubes from the surface within which true distances are kept
HOLLOW_WALL = 1 / 75             # walls averaging under this share of the longest side: a skin
KMEANS_TRIES = 6
KMEANS_ROUNDS = 40


class PrintError(ConfigError):
    pass


def require() -> None:
    if np is None:
        raise PrintError("`kami print` needs numpy and Pillow\n"
                         "`pip install numpy pillow scipy scikit-image` (no other kami "
                         "command needs them; the last two rebuild meshes that are not closed).")
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise PrintError("`kami print` needs Pillow to read the mesh's texture\n"
                         "`pip install pillow`") from None


@dataclass
class Mesh:
    vertices: "np.ndarray"       # (V, 3) float64
    faces: "np.ndarray"          # (F, 3) int64
    colors: "np.ndarray"         # (F, 3) float64, sRGB 0..255, one per triangle


# -- GLB ---------------------------------------------------------------------------------

COMPONENT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
COMPRESSED = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")


def read_glb(path: Path) -> tuple[dict, bytes]:
    """The JSON and the binary chunk of a .glb file."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise PrintError(f"Cannot read {path}: {e.strerror or e}") from None
    if len(data) < 20 or data[:4] != b"glTF":
        raise PrintError(f"{path} is not a GLB file\n"
                         "`kami print` reads the .glb files `kami mesh` saves.")
    doc, blob, at = None, b"", 12
    while at + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, at)
        chunk = data[at + 8:at + 8 + size]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            blob = chunk
        at += 8 + size + (-size % 4)
    if doc is None:
        raise PrintError(f"{path} has no glTF content")
    return doc, blob


def _accessor(doc: dict, blob: bytes, index: int):
    acc = doc["accessors"][index]
    if "sparse" in acc or "bufferView" not in acc:
        raise PrintError("This GLB uses sparse accessors, which `kami print` does not read")
    view = doc["bufferViews"][acc["bufferView"]]
    if view.get("buffer", 0) != 0 or doc["buffers"][0].get("uri"):
        raise PrintError("This GLB keeps its data in another file; `kami print` needs it embedded")
    dtype = np.dtype(COMPONENT[acc["componentType"]])
    width, count = WIDTH[acc["type"]], acc["count"]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    item = dtype.itemsize * width
    stride = view.get("byteStride") or item
    if stride == item:
        out = np.frombuffer(blob, dtype, count * width, start).reshape(count, width)
    else:                                            # interleaved attributes
        raw = np.frombuffer(blob, np.uint8, (count - 1) * stride + item, start)
        rows = np.arange(count)[:, None] * stride + np.arange(item)
        out = raw[rows].copy().view(dtype).reshape(count, width)
    if acc.get("normalized") and dtype.kind in "iu":
        return out / float(np.iinfo(dtype).max)
    return out


def _node_matrix(node: dict):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    m[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m[:3, :3] *= np.array(node.get("scale", (1, 1, 1)), float)
    m[:3, 3] = node.get("translation", (0, 0, 0))
    return m


def _image(doc: dict, blob: bytes, index: int, folder: Path):
    """A glTF image as an (H, W, 3) sRGB array."""
    from PIL import Image
    image = doc["images"][index]
    if "bufferView" in image:
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = blob[start:start + view["byteLength"]]
    elif image.get("uri", "").startswith("data:"):
        raw = base64.b64decode(image["uri"].split(",", 1)[1])
    else:
        raw = (folder / image.get("uri", "")).read_bytes()
    try:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        raise PrintError(f"Cannot read the mesh's texture: {e}") from None


def to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055) * 255


def to_linear(srgb):
    c = np.clip(np.asarray(srgb, float) / 255, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# Where a triangle's colour is read: its centre and a point towards each corner.
SAMPLES = ((1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))


def sample_texture(texture, uv, faces):
    """One sRGB colour per triangle: the texture under four points of it, averaged."""
    corners = uv[faces]                                          # (F, 3, 2)
    points = np.einsum("sk,fkc->fsc", np.array(SAMPLES), corners)
    h, w = texture.shape[:2]
    x = np.clip((np.mod(points[..., 0], 1.0) * w).astype(np.int64), 0, w - 1)
    y = np.clip((np.mod(points[..., 1], 1.0) * h).astype(np.int64), 0, h - 1)
    return texture[y, x].astype(float).mean(axis=1)


def load_glb(path: Path) -> Mesh:
    """Every triangle of the GLB's scene in one mesh, with a colour each: from the base
    colour texture, else the vertex colours, else the material's colour."""
    require()
    path = Path(path)
    doc, blob = read_glb(path)
    used = set(doc.get("extensionsRequired", ()))
    if used & set(COMPRESSED):
        raise PrintError(f"{path.name} is compressed ({', '.join(sorted(used & set(COMPRESSED)))})\n"
                         "`kami print` reads plain GLBs, as `kami mesh` saves them.")
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    todo = [(i, np.eye(4)) for i in scenes[doc.get("scene", 0)].get("nodes", [])]
    textures: dict[int, object] = {}
    vertices, faces, colors, offset = [], [], [], 0
    while todo:
        index, parent = todo.pop()
        node = nodes[index]
        matrix = parent @ _node_matrix(node)
        todo += [(child, matrix) for child in node.get("children", ())]
        if "mesh" not in node:
            continue
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                continue                                         # points, lines, strips
            attrs = prim["attributes"]
            pos = _accessor(doc, blob, attrs["POSITION"]).astype(float)
            pos = pos @ matrix[:3, :3].T + matrix[:3, 3]
            if "indices" in prim:
                tri = _accessor(doc, blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                tri = np.arange(len(pos) // 3 * 3, dtype=np.int64).reshape(-1, 3)
            if np.linalg.det(matrix[:3, :3]) < 0:                # a mirrored node
                tri = tri[:, ::-1]
            material = doc["materials"][prim["material"]] if "material" in prim else {}
            pbr = material.get("pbrMetallicRoughness", {})
            factor = np.array(pbr.get("baseColorFactor", (1, 1, 1, 1)), float)[:3]
            ref = pbr.get("baseColorTexture")
            if ref and "TEXCOORD_%d" % ref.get("texCoord", 0) in attrs:
                source = doc["textures"][ref["index"]].get("source", 0)
                if source not in textures:
                    textures[source] = _image(doc, blob, source, path.parent)
                uv = _accessor(doc, blob, attrs["TEXCOORD_%d" % ref.get("texCoord", 0)]).astype(float)
                color = sample_texture(textures[source], uv, tri)
                if not np.allclose(factor, 1):
                    color = to_srgb(to_linear(color) * factor)
            elif "COLOR_0" in attrs:
                per_vertex = _accessor(doc, blob, attrs["COLOR_0"]).astype(float)[:, :3]
                color = to_srgb(per_vertex[tri].mean(axis=1) * factor)
            else:
                color = np.tile(to_srgb(factor), (len(tri), 1))
            vertices.append(pos)
            faces.append(tri + offset)
            colors.append(color)
            offset += len(pos)
    if not faces or not sum(len(f) for f in faces):
        raise PrintError(f"{path.name} has no triangles")
    return Mesh(np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors))


# -- topology ----------------------------------------------------------------------------

def _edge_keys(faces, n_vertices: int):
    """One number per side of every triangle, the same for both triangles on an edge."""
    a, b = faces.ravel(), faces[:, (1, 2, 0)].ravel()
    return np.minimum(a, b) * np.int64(n_vertices) + np.maximum(a, b)


def edge_counts(faces, n_vertices: int) -> tuple[int, int]:
    """(open edges, edges shared by more than two triangles). A closed surface has neither."""
    if not len(faces):
        return 0, 0
    counts = np.unique(_edge_keys(faces, n_vertices), return_counts=True)[1]
    return int((counts == 1).sum()), int((counts > 2).sum())


def adjacency(faces, n_vertices: int):
    """Pairs of triangles that share an edge, as two arrays. Only edges with exactly two
    triangles count: where three or more meet (two surfaces touching along a line) they
    are not neighbours, or a shell and the shell inside it would pass for one piece."""
    keys = _edge_keys(faces, n_vertices)
    order = np.argsort(keys, kind="stable")
    keys, owner = keys[order], order // 3
    same = keys[1:] == keys[:-1]
    crowded = np.zeros(len(same), bool)
    crowded[1:] |= same[:-1]
    crowded[:-1] |= same[1:]
    pair = same & ~crowded
    return owner[:-1][pair], owner[1:][pair]


def components(n: int, a, b):
    """Connected components of n nodes joined by the pairs (a, b): each node's label is the
    smallest node of its component. Roots are hooked to smaller roots, then every node
    jumps to its root, so it takes a handful of rounds whatever the shape."""
    label = np.arange(n, dtype=np.int64)
    while len(a):
        la, lb = label[a], label[b]
        low = np.minimum(la, lb)
        new = label.copy()
        np.minimum.at(new, la, low)
        np.minimum.at(new, lb, low)
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            break
        label = new
    return label


def face_areas(mesh: Mesh):
    v = mesh.vertices[mesh.faces]
    return np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) / 2


def weld(mesh: Mesh) -> Mesh:
    """Vertices at the same place become one (a GLB splits them along texture seams, which
    leaves the surface open), and triangles squeezed to nothing by it are dropped."""
    span = float(np.ptp(mesh.vertices, axis=0).max()) or 1.0
    grid = np.round(mesh.vertices / (span * WELD)).astype(np.int64)
    _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2]))
    return Mesh(mesh.vertices[first], faces[keep], mesh.colors[keep])


def compact(mesh: Mesh) -> Mesh:
    """Without the vertices no triangle uses."""
    used = np.unique(mesh.faces)
    remap = np.zeros(len(mesh.vertices), np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.vertices[used], remap[mesh.faces], mesh.colors)


@dataclass
class Solid:
    """What `solid` found and did."""
    cavities: int = 0            # inner shells dropped
    fragments: int = 0           # slivers dropped
    pieces: int = 1              # separate bodies left
    flipped: bool = False        # the surface faced inwards and was turned
    open_edges: int = 0
    nonmanifold_edges: int = 0
    rebuilt: bool = False        # the surface is a new one, drawn around what the old enclosed
    detail: float = 0.0          # mm: detail finer than this was lost in rebuilding
    wall: float = 0.0            # mm: how thick the body is on average (see mean_wall)
    hollow: bool = False         # so thin for its size that it is a skin, not a body

    @property
    def closed(self) -> bool:
        return not self.open_edges and not self.nonmanifold_edges


def solid(mesh: Mesh, keep_cavities: bool = False, crumbs: float = 0.0) -> tuple[Mesh, Solid]:
    """The mesh as a solid body: welded, facing outwards, without the surfaces inside it
    (the inner half of a double shell would print as a hollow, paper-thin figure) and
    without loose slivers. It reports what is left open; it does not fill holes. With
    `crumbs`, separate bodies under that share of the largest one's volume go too."""
    mesh = weld(mesh)
    if not len(mesh.faces):
        raise PrintError("The mesh has no triangles with an area")
    report = Solid()
    n = len(mesh.faces)
    label = components(n, *adjacency(mesh.faces, len(mesh.vertices)))
    _, part = np.unique(label, return_inverse=True)
    tri = mesh.vertices[mesh.faces]
    volume = np.bincount(part, np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6)
    area = np.bincount(part, face_areas(mesh))
    count = len(volume)
    low, high = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    np.minimum.at(low, part, tri.min(axis=1))
    np.maximum.at(high, part, tri.max(axis=1))

    faces = mesh.faces
    if volume[np.argmax(np.abs(volume))] < 0:                    # inside out
        faces, volume, report.flipped = faces[:, ::-1], -volume, True
    keep = area >= area.sum() * FRAGMENT
    if crumbs:
        keep &= (volume < 0) | (volume >= volume.max() * crumbs)
    report.fragments = int((~keep).sum())
    if not keep_cavities:
        margin = (high - low).max() * 1e-3
        outer = [i for i in np.flatnonzero(keep) if volume[i] > 0]
        for i in np.flatnonzero(keep & (volume < 0)):
            inside = any(volume[o] > -volume[i] * 0.5
                         and (low[i] >= low[o] - margin).all() and (high[i] <= high[o] + margin).all()
                         for o in outer)
            if inside:
                keep[i] = False
                report.cavities += 1
    if not keep.any():
        raise PrintError("Nothing is left of the mesh after cleaning it")
    chosen = keep[part]
    out = compact(Mesh(mesh.vertices, faces[chosen], mesh.colors[chosen]))
    report.pieces = int(keep.sum())
    report.open_edges, report.nonmanifold_edges = edge_counts(out.faces, len(out.vertices))
    return out, report


def mean_wall(mesh: Mesh) -> float:
    """How thick a closed body is on average: twice its volume over its surface. A ball
    gives two thirds of its radius; a hollow skin gives the thickness of the skin."""
    tri = mesh.vertices[mesh.faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    return float(2 * abs(volume) / max(face_areas(mesh).sum(), 1e-30))


# -- rebuilding --------------------------------------------------------------------------

def _samples(vertices, faces, step: float):
    """Points on every triangle no further apart than `step`, and the triangle of each."""
    tri = vertices[faces]
    longest = np.maximum.reduce([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1)
                                 for i in range(3)])
    cuts = np.clip(np.ceil(longest / step), 1, 2000).astype(np.int64)
    points, owner = [], []
    for k in np.unique(cuts):
        chosen = np.flatnonzero(cuts == k)
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        inside = i + j <= k
        weights = np.stack([1 - (i + j) / k, i / k, j / k], axis=-1)[inside]
        for part in np.array_split(chosen, max(1, len(chosen) * len(weights) // 4_000_000)):
            points.append(np.einsum("kc,tcx->tkx", weights, tri[part]).reshape(-1, 3))
            owner.append(np.repeat(part, len(weights)))
    return np.concatenate(points), np.concatenate(owner)


def rebuild(mesh: Mesh, detail: float = DETAIL) -> tuple[Mesh, float]:
    """A new closed surface around everything the mesh encloses, with its colours, for a
    mesh that cannot be closed as it is: open, crossing itself, in overlapping pieces, or
    a skin with a hollow inside. The method is jointwright's (solidify.ts):

    - the triangles are marked on a grid of small cubes, the marked skin is thickened by
      one cube (which closes pinholes) and the outside is flooded from a corner: what the
      flood cannot reach is the body;
    - next to the surface each cube holds its distance to the nearest triangle that faces
      the outside (the inner side of a skin is what is being removed), with its sign, and
      the new surface is drawn where that distance is zero: where the old one was.

    Nothing here needs the mesh to be closed. The price is detail finer than a cube, and
    an opening wider than a cube still lets the flood in: a skin open like that comes out
    a skin (`Solid.hollow` says so). Returns the mesh and the cube's edge."""
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        from skimage import measure
    except ImportError:
        raise PrintError("This mesh is not a closed solid, and rebuilding it needs scipy and "
                         "scikit-image\n`pip install scipy scikit-image`, or `--rebuild never` "
                         "to write it as it is and let the slicer repair it.") from None
    v, faces = mesh.vertices, mesh.faces
    size = np.ptp(v, axis=0)
    cell = max(float(detail), float(np.cbrt(np.prod(np.maximum(size, detail)) / MAX_CELLS)))
    origin = v.min(axis=0) - PAD * cell
    shape = tuple(int(n) for n in np.ceil(size / cell).astype(np.int64) + 2 * PAD + 1)

    def cells(points):
        index = np.floor((points - origin) / cell).astype(np.int64)
        return tuple(np.clip(index[:, k], 0, shape[k] - 1) for k in range(3))

    points, owner = _samples(v, faces, cell / 2)
    skin = np.zeros(shape, bool)
    skin[cells(points)] = True
    wall = ndimage.binary_dilation(skin)
    regions, _ = ndimage.label(~wall)
    outside = regions == regions[0, 0, 0]
    # the thickening is given back where it touches the flood, so the flood comes right
    # up to the cubes the triangles pass through, on the outer side only
    outside |= wall & ~skin & ndimage.binary_dilation(outside)

    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= np.maximum(length, 1e-30)
    centre = tri.mean(axis=1)
    outer = outside[cells(centre + normal * 1.5 * cell)] | outside[cells(centre + normal * 2.5 * cell)]
    outer &= length[:, 0] > 1e-12                    # squeezed to a line: it faces nowhere
    if not outer.any():                              # nothing faces the flood: take it all
        outer[:] = True
    kept = outer[owner]
    points, facing = points[kept], normal[owner[kept]]

    band = ndimage.binary_dilation(skin, iterations=2)
    centres = (np.argwhere(band) + 0.5) * cell + origin
    far, near = cKDTree(points).query(centres, workers=-1)
    depth = np.einsum("ij,ij->i", points[near] - centres, facing[near])   # behind it: inside
    # a triangle far to the side says little about the distance, only about the side
    depth = np.where(far > np.abs(depth) + cell, np.copysign(far, depth), depth)
    flooded = outside[band]
    depth = np.where(flooded, -np.abs(depth), depth)
    depth = np.where(~wall[band] & ~flooded, np.abs(depth), depth)        # the flood never got there
    depth[depth == 0] = 1e-6 * cell
    field = np.where(outside, -BAND * cell, BAND * cell).astype(np.float32)
    field[band] = np.clip(depth, -BAND * cell, BAND * cell)

    try:
        with py_warnings.catch_warnings():              # scikit-image's own, under a new numpy
            py_warnings.simplefilter("ignore", DeprecationWarning)
            drawn, triangles = measure.marching_cubes(field, level=0.0)[:2]
    except (ValueError, RuntimeError):
        raise PrintError("Nothing is left of the mesh after rebuilding it: it encloses "
                         f"nothing thicker than {cell:.2f} mm (`--detail`)") from None
    drawn = (drawn + 0.5) * cell + origin
    triangles = triangles.astype(np.int64)
    corners = drawn[triangles]
    if np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() < 0:
        triangles = triangles[:, ::-1]
    source = np.flatnonzero(outer)
    colors = mesh.colors[source[cKDTree(centre[source]).query(drawn[triangles].mean(axis=1), workers=-1)[1]]]
    return Mesh(drawn, triangles, colors), cell


# -- size and base -----------------------------------------------------------------------

def base_area(mesh: Mesh) -> float:
    """The area of the triangles that lie on the plate (Z = 0), in mm²."""
    on_plate = (mesh.vertices[mesh.faces][:, :, 2] <= 1e-9).all(axis=1)
    return float(face_areas(mesh)[on_plate].sum())


def place(mesh: Mesh, height: float, base: float = 2.0, up: str = "y") -> Mesh:
    """The mesh as it goes on the plate: Z up (glTF is Y up), its lowest `base` percent
    pressed flat so it has something to stand on, `height` mm tall, centred over the
    origin with the base at Z = 0."""
    v = mesh.vertices.copy()
    if up == "y":
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)       # +90° about X: still right-handed
    bottom, tall = v[:, 2].min(), float(np.ptp(v[:, 2]))
    if tall <= 0:
        raise PrintError("The mesh is flat: it has no height to print")
    floor = bottom + tall * max(base, 0.0) / 100
    v[:, 2] = np.maximum(v[:, 2], floor)
    v[:, 2] -= floor
    v *= height / float(v[:, 2].max())
    v[:, :2] -= (v[:, :2].min(axis=0) + v[:, :2].max(axis=0)) / 2
    return Mesh(v, mesh.faces, mesh.colors)


# -- colours -----------------------------------------------------------------------------

def oklab(srgb):
    """sRGB 0..255 -> Oklab, where distances are close to how different colours look."""
    lin = to_linear(srgb)
    lms = lin @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                          [0.5363325363, 0.6806995451, 0.2817188376],
                          [0.0514459929, 0.1073969566, 0.6299787005]])
    return np.cbrt(lms) @ np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                                    [0.7936177850, -2.4285922050, 0.7827717662],
                                    [-0.0040720468, 0.4505937099, -0.8086757660]])


def color_space(srgb):
    """Where colours are compared: Oklab with its lightness axis shortened (LIGHTNESS)."""
    return oklab(srgb) * np.array([LIGHTNESS, 1.0, 1.0])


def parse_palette(text: str) -> list[tuple[int, int, int]]:
    """'#1a4fa0, fff, #202020' -> [(26, 79, 160), (255, 255, 255), (32, 32, 32)]."""
    colors = []
    for raw in text.replace(";", ",").split(","):
        word = raw.strip().lstrip("#")
        if not word:
            continue
        if len(word) == 3:
            word = "".join(c * 2 for c in word)
        try:
            if len(word) != 6:
                raise ValueError
            colors.append(tuple(bytes.fromhex(word)))
        except ValueError:
            raise PrintError(f"Not a colour: {raw.strip()!r}\n"
                             "A palette is hex colours with commas: --palette \"#1a4fa0,#ffffff,#202020\"") from None
    if not 1 <= len(colors) <= MAX_COLORS:
        raise PrintError(f"A palette has 1 to {MAX_COLORS} colours, not {len(colors)}")
    return colors


def hex_color(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(float(c))) for c in rgb)


def _bins(colors, weights):
    """The colours grouped into coarse bins: (mean sRGB of each bin, its weight, bin of
    every colour). Half a million triangles become a few thousand points."""
    q = np.clip(colors, 0, 255).astype(np.int64) >> (8 - COLOR_BITS)
    key = (q[:, 0] << (2 * COLOR_BITS)) | (q[:, 1] << COLOR_BITS) | q[:, 2]
    _, index = np.unique(key, return_inverse=True)
    weight = np.bincount(index, weights)
    safe = np.maximum(weight, 1e-30)
    mean = np.stack([np.bincount(index, weights * colors[:, c]) / safe for c in range(3)], axis=1)
    return mean, weight, index


def _kmeans(points, weight, k: int, rng):
    centres = [points[np.argmax(weight)]]
    for _ in range(k - 1):                                       # k-means++: far and heavy
        dist = ((points[:, None] - np.array(centres)[None]) ** 2).sum(-1).min(axis=1) * weight
        total = dist.sum()
        if total <= 0:
            break
        centres.append(points[rng.choice(len(points), p=dist / total)])
    centres = np.array(centres)
    for _ in range(KMEANS_ROUNDS):
        owner = ((points[:, None] - centres[None]) ** 2).sum(-1).argmin(axis=1)
        moved = centres.copy()
        for c in range(len(centres)):
            mine = owner == c
            if weight[mine].sum() > 0:
                moved[c] = (points[mine] * weight[mine, None]).sum(0) / weight[mine].sum()
        if np.allclose(moved, centres):
            break
        centres = moved
    spread = (((points - centres[owner]) ** 2).sum(-1) * weight).sum()
    return owner, spread


def quantize(colors, weights, count: int, seed: int = 0):
    """The triangles' colours reduced to at most `count`: (palette as sRGB rows, the
    palette entry of every triangle), the colour covering most surface first. k-means in
    `color_space`, weighted by area; the same input always gives the same answer."""
    mean, weight, index = _bins(colors, weights)
    points = color_space(mean)
    k = max(1, min(count, len(mean)))
    best = None
    for attempt in range(KMEANS_TRIES):
        owner, spread = _kmeans(points, weight, k, np.random.default_rng(seed + attempt))
        if best is None or spread < best[1]:
            best = (owner, spread)
    owner = best[0]
    share = np.bincount(owner, weight, minlength=k)
    order = [c for c in np.argsort(-share) if share[c] > 0]
    palette = np.array([to_srgb((to_linear(mean[owner == c]) * weight[owner == c, None]).sum(0)
                                / share[c]) for c in order])
    rank = np.zeros(k, np.int64)
    rank[order] = np.arange(len(order))
    return palette, rank[owner][index]


def nearest(colors, weights, palette):
    """Every triangle given the closest colour of a fixed palette (the filaments you have)."""
    mean, _, index = _bins(colors, weights)
    target = color_space(np.array(palette, float))
    return ((color_space(mean)[:, None] - target[None]) ** 2).sum(-1).argmin(axis=1)[index]


def smooth(labels, a, b, count: int, rounds: int):
    """Each triangle takes the colour of two of its three neighbours when they agree
    against it: single-triangle noise and ragged borders go, shapes stay."""
    rows = np.arange(len(labels))
    for _ in range(max(rounds, 0)):
        votes = np.zeros((len(labels), count))
        np.add.at(votes, (a, labels[b]), 1)
        np.add.at(votes, (b, labels[a]), 1)
        votes[rows, labels] += 0.5
        new = votes.argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def absorb(labels, a, b, areas, count: int, min_area: float):
    """Patches of one colour smaller than `min_area` take the colour they touch most: a
    nozzle cannot print them, and each one costs a filament change."""
    if min_area <= 0:
        return labels
    for _ in range(4):
        same = labels[a] == labels[b]
        _, patch = np.unique(components(len(labels), a[same], b[same]), return_inverse=True)
        small = np.bincount(patch, areas) < min_area
        if not small.any():
            break
        votes = np.zeros((len(small), count))
        np.add.at(votes, (patch[a[~same]], labels[b[~same]]), 1)
        np.add.at(votes, (patch[b[~same]], labels[a[~same]]), 1)
        change = small & (votes.sum(axis=1) > 0)
        if not change.any():
            break
        labels = np.where(change[patch], votes.argmax(axis=1)[patch], labels)
    return labels


# -- 3MF ---------------------------------------------------------------------------------

def paint_code(slot: int) -> str:
    """Bambu Studio's `paint_color` for a whole triangle printed with filament `slot`
    (1-based): TriangleSelector's bit stream for an unsplit triangle, as hex with the last
    nibble first. 1 -> "4", 2 -> "8", 3 -> "0C", 4 -> "1C", ... 18 -> "0FC"."""
    if slot < 3:
        return "%X" % (slot << 2)
    rest, nibbles = slot - 3, ["C"]
    while rest >= 15:
        nibbles.append("F")
        rest -= 15
    nibbles.append("%X" % rest)
    return "".join(reversed(nibbles))


CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                 ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                 ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                 '</Types>\n')
RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>\n')


def model_xml(mesh: Mesh, labels, palette, name: str) -> str:
    """The 3MF model. Every triangle of filament slot N > 1 says so twice: `paint_color`
    for Bambu Studio and OrcaSlicer, and `pid`/`p1` into the base materials for the 3MF
    standard. Slot 1 is the object's own material, so its triangles carry nothing."""
    codes = [""] + [f' pid="1" p1="{i}" paint_color="{paint_code(i + 1)}"'
                    for i in range(1, len(palette))]
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n'
           '<model unit="millimeter" xml:lang="en-US" '
           'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
           f' <metadata name="Title">{quoteattr(name)[1:-1]}</metadata>\n'
           f' <metadata name="Application">kami {__version__}</metadata>\n'
           ' <resources>\n  <basematerials id="1">\n']
    out += [f'   <base name="Filament {i + 1}" displaycolor="{hex_color(c)}"/>\n'
            for i, c in enumerate(palette)]
    out.append(f'  </basematerials>\n  <object id="2" type="model" name={quoteattr(name)} '
               'pid="1" pindex="0">\n   <mesh>\n    <vertices>\n')
    out += ['     <vertex x="%.4f" y="%.4f" z="%.4f"/>\n' % row for row in map(tuple, mesh.vertices.tolist())]
    out.append('    </vertices>\n    <triangles>\n')
    out += ['     <triangle v1="%d" v2="%d" v3="%d"%s/>\n' % (a, b, c, codes[slot])
            for (a, b, c), slot in zip(mesh.faces.tolist(), labels.tolist())]
    out.append('    </triangles>\n   </mesh>\n  </object>\n </resources>\n'
               ' <build>\n  <item objectid="2"/>\n </build>\n</model>\n')
    return "".join(out)


def write_3mf(path: Path, mesh: Mesh, labels, palette, name: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model_xml(mesh, labels, palette, name))


# -- the whole thing ---------------------------------------------------------------------

@dataclass
class Printed:
    path: Path | None
    triangles: int
    size: tuple[float, float, float]         # mm
    base_area: float                         # mm²
    solid: Solid
    colors: list[tuple[str, float]] = field(default_factory=list)   # (hex, share of the surface)


def prepare(source: Path, *, height: float = 80.0, base: float = 2.0, keep_cavities: bool = False,
            up: str = "y", rebuilding: str = "auto", detail: float = DETAIL) -> tuple[Mesh, Solid]:
    """The slow half of a conversion, the same whatever the colours: the GLB as a solid on
    the plate (`solid`, `place`, and `rebuild` when it is needed or asked for)."""
    require()
    if height <= 0:
        raise PrintError("--height must be more than 0 mm")
    if not 0 <= base < 50:
        raise PrintError("--base is the share of the height pressed flat, 0 to 50 (percent)")
    if rebuilding not in ("auto", "always", "never"):
        raise PrintError("--rebuild is auto, always or never")
    if detail <= 0:
        raise PrintError("--detail must be more than 0 mm")
    mesh, report = solid(load_glb(source), keep_cavities)
    mesh = place(mesh, height, base, up)

    def measure_wall(report):
        report.wall = mean_wall(mesh)
        report.hollow = report.closed and report.wall < float(np.ptp(mesh.vertices, axis=0).max()) * HOLLOW_WALL

    measure_wall(report)
    if rebuilding == "always" or (rebuilding == "auto" and (not report.closed or report.hollow)):
        first = report
        mesh, cell = rebuild(mesh, detail)
        mesh, report = solid(mesh, crumbs=CRUMB)         # crumbs and bubbles of the grid
        mesh.vertices[:, 2] = np.where(np.abs(mesh.vertices[:, 2]) < cell * 1e-3, 0.0, mesh.vertices[:, 2])
        report.rebuilt, report.detail = True, cell
        report.cavities, report.fragments, report.flipped = first.cavities, first.fragments, first.flipped
        measure_wall(report)
    return mesh, report


def paint(mesh: Mesh, *, colors: int = 4, palette: list | None = None, smoothing: int = 3,
          min_patch: float = 1.0):
    """The quick half: (filament slot of every triangle, the slots' sRGB colours, each
    slot's share of the surface). `palette` fixes the colours and their slots; without it
    the `colors` that describe the texture best are found, the most used first."""
    if palette is None and not 1 <= colors <= MAX_COLORS:
        raise PrintError(f"--colors goes from 1 to {MAX_COLORS}")
    if palette is not None and not 1 <= len(palette) <= MAX_COLORS:
        raise PrintError(f"A palette has 1 to {MAX_COLORS} colours, not {len(palette)}")
    areas = face_areas(mesh)
    weights = np.maximum(areas, 1e-12)
    if palette:
        swatches = np.array(palette, float)
        labels = nearest(mesh.colors, weights, swatches)
    else:
        swatches, labels = quantize(mesh.colors, weights, colors)
    a, b = adjacency(mesh.faces, len(mesh.vertices))
    labels = smooth(labels, a, b, len(swatches), smoothing)
    labels = absorb(labels, a, b, areas, len(swatches), min_patch)
    share = np.bincount(labels, areas, minlength=len(swatches)) / max(areas.sum(), 1e-30)
    return labels, swatches, share


def describe(path: Path | None, mesh: Mesh, report: Solid, swatches, share) -> Printed:
    return Printed(path=path, triangles=len(mesh.faces),
                   size=tuple(float(x) for x in np.ptp(mesh.vertices, axis=0)),
                   base_area=base_area(mesh), solid=report,
                   colors=[(hex_color(c), float(s)) for c, s in zip(swatches, share)])


PLATE_MM = 256                   # the plate of a Bambu P2S / P1 / X1 / A1, the common size


def warnings(done: Printed) -> list[str]:
    """What to know before printing it, in plain sentences (`code` in backticks)."""
    x, y, z = done.size
    solid, out = done.solid, []
    if not solid.closed:
        out.append(f"The surface is not closed ({solid.open_edges} open edges, "
                   f"{solid.nonmanifold_edges} shared by more than two triangles): "
                   "let the slicer repair it when it offers to")
    if solid.hollow:
        out.append(f"It is a hollow skin (walls of {solid.wall:.1f} mm on average), open "
                   "somewhere, so it could not be filled: it will print as a thin shell")
    if solid.pieces > 1:
        out.append(f"{solid.pieces} separate pieces: the loose ones may need supports "
                   "or deleting in the slicer")
    if max(x, y) > PLATE_MM or z > PLATE_MM:
        out.append(f"Bigger than a {PLATE_MM} mm plate: lower the height")
    if done.base_area < 0.02 * x * y:
        out.append("It barely touches the plate: raise the base (percent of the height "
                   "pressed flat) or add a brim")
    return out


def convert(source: Path, target: Path, *, height: float = 80.0, base: float = 2.0,
            colors: int = 4, palette: list | None = None, smoothing: int = 3,
            min_patch: float = 1.0, keep_cavities: bool = False, up: str = "y",
            rebuilding: str = "auto", detail: float = DETAIL) -> Printed:
    """GLB -> 3MF: `prepare`, `paint`, `write_3mf`. `rebuilding`: "auto" draws a new
    surface (see `rebuild`) only when cleaning the mesh did not leave a closed body,
    "always" and "never" say so."""
    require()
    if palette is None and not 1 <= colors <= MAX_COLORS:
        raise PrintError(f"--colors goes from 1 to {MAX_COLORS}")
    mesh, report = prepare(source, height=height, base=base, keep_cavities=keep_cavities, up=up,
                           rebuilding=rebuilding, detail=detail)
    labels, swatches, share = paint(mesh, colors=colors, palette=palette, smoothing=smoothing,
                                    min_patch=min_patch)
    write_3mf(target, mesh, labels, swatches, Path(source).stem)
    return describe(target, mesh, report, swatches, share)


def thumbnail(source: Path, size: int = 256) -> bytes:
    """A small picture of a GLB (PNG, transparent around it), for a library of meshes that
    have no picture of their own: its surface as dots of its texture's colours, seen a
    little from the side and from above, lit from the front."""
    require()
    from PIL import Image
    mesh = load_glb(source)
    v = mesh.vertices - (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    turn, tilt = np.radians(-30), np.radians(15)
    spin = np.array([[np.cos(turn), 0, np.sin(turn)], [0, 1, 0], [-np.sin(turn), 0, np.cos(turn)]])
    lean = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    v = v @ (lean @ spin).T                                      # the camera looks down -Z
    reach = float(np.abs(v[:, :2]).max()) or 1.0
    scale = (size / 2 - 4) / reach
    points, owner = _samples(v, mesh.faces, 0.7 / scale)
    tri = v[mesh.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    seen = normal[owner, 2] > 0
    points, owner = points[seen], owner[seen]
    order = np.argsort(points[:, 2])                             # far first: the near ones win
    points, owner = points[order], owner[order]
    x = np.clip((points[:, 0] * scale + size / 2).astype(np.int64), 0, size - 1)
    y = np.clip((size / 2 - points[:, 1] * scale).astype(np.int64), 0, size - 1)
    light = 0.55 + 0.45 * normal[owner, 2:3]
    image = np.zeros((size, size, 4), np.uint8)
    image[y, x, :3] = np.clip(mesh.colors[owner] * light, 0, 255).astype(np.uint8)
    image[y, x, 3] = 255
    out = io.BytesIO()
    Image.fromarray(image, "RGBA").save(out, "PNG")
    return out.getvalue()


def preview_glb(mesh: Mesh, labels, palette) -> bytes:
    """The painted model as a GLB to look at before saving the 3MF: one primitive per
    filament, each a plain material of its colour, sharing the vertices. Y up and in
    metres, as glTF has it; no normals, so a viewer shades it flat, facet by facet."""
    v = (np.stack([mesh.vertices[:, 0], mesh.vertices[:, 2], -mesh.vertices[:, 1]], axis=1)
         * 0.001).astype(np.float32)
    blob, views, accessors, primitives, materials = bytearray(), [], [], [], []

    def add(data: bytes, target: int) -> int:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data + bytes(-len(data) % 4))
        return len(views) - 1

    accessors.append({"bufferView": add(v.tobytes(), 34962), "componentType": 5126, "count": len(v),
                      "type": "VEC3", "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()})
    for slot, color in enumerate(palette):
        index = mesh.faces[labels == slot].astype(np.uint32)
        if not len(index):
            continue
        accessors.append({"bufferView": add(index.tobytes(), 34963), "componentType": 5125,
                          "count": int(index.size), "type": "SCALAR"})
        materials.append({"name": f"Filament {slot + 1}", "pbrMetallicRoughness": {
            "baseColorFactor": [*to_linear(color).tolist(), 1.0], "metallicFactor": 0.0,
            "roughnessFactor": 0.7}})
        primitives.append({"attributes": {"POSITION": 0}, "indices": len(accessors) - 1,
                           "material": len(materials) - 1})
    doc = {"asset": {"version": "2.0", "generator": f"kami {__version__}"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "meshes": [{"primitives": primitives}],
           "materials": materials, "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(blob)}]}
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = (struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body
