"""Tests for `kami print` (kami/printing.py). They need numpy and Pillow, like the command
itself, and are skipped without them: `python -m unittest`."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import struct
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

from kami import cli, printing

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
try:
    import scipy  # noqa: F401
    import skimage  # noqa: F401
except ImportError:
    skimage = None

RED, BLUE = (200, 30, 30), (30, 60, 200)

# A cube: 8 corners, 12 triangles facing outwards.
CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
TRIANGLES = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]


def cube(scale=1.0, inwards=False):
    """A cube as loose triangles (three vertices each, as a GLB has them along texture
    seams): positions, and texture coordinates that make its top face (y = 1) red (the
    left of the texture) and the other five blue."""
    positions, uv = [], []
    for tri in TRIANGLES:
        corners = [CORNERS[i] for i in (reversed(tri) if inwards else tri)]
        u = 0.25 if sum(c[1] for c in corners) > 2 else 0.75
        positions += [[x * scale for x in c] for c in corners]
        uv += [[u, 0.5]] * 3
    return positions, uv


def glb(parts, texture=True) -> bytes:
    """A GLB with one mesh: `parts` are (positions, uv) lists, joined."""
    positions = np.array([p for part in parts for p in part[0]], np.float32)
    uv = np.array([t for part in parts for t in part[1]], np.float32)
    indices = np.arange(len(positions), dtype=np.uint32)
    png = io.BytesIO()
    Image.fromarray(np.array([[RED, BLUE]], np.uint8)).save(png, "PNG")
    chunks = [indices.tobytes(), positions.tobytes(), uv.tobytes(), png.getvalue()]
    views, blob = [], b""
    for chunk in chunks:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(chunk)})
        blob += chunk + b"\0" * (-len(chunk) % 4)
    material = {"pbrMetallicRoughness": ({"baseColorTexture": {"index": 0}} if texture
                                         else {"baseColorFactor": [0, 1, 0, 1]})}
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0}], "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
           "accessors": [{"bufferView": 0, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
                         {"bufferView": 1, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                         {"bufferView": 2, "componentType": 5126, "count": len(uv), "type": "VEC2"}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2},
                                       "indices": 0, "material": 0}]}],
           "materials": [material], "textures": [{"source": 0}],
           "images": [{"bufferView": 3, "mimeType": "image/png"}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    body = (struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(blob), 0x004E4942) + blob)
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


def model_of(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("3D/3dmodel.model").decode()


class PaintCodeTests(unittest.TestCase):
    def test_codes_are_bambu_studios(self):
        # TriangleSelector::serialize for an unsplit triangle, last nibble first
        self.assertEqual([printing.paint_code(n) for n in (1, 2, 3, 4, 5, 17, 18, 19)],
                         ["4", "8", "0C", "1C", "2C", "EC", "0FC", "1FC"])

    def test_palette(self):
        self.assertEqual(printing.parse_palette("#1a4fa0, FFF;#202020"),
                         [(26, 79, 160), (255, 255, 255), (32, 32, 32)])
        for bad in ("blue", "#12345", "", ",".join(["#000"] * 17)):
            with self.assertRaises(printing.PrintError):
                printing.parse_palette(bad)


@unittest.skipIf(np is None, "numpy and Pillow are not installed")
class PrintingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def save(self, name="cube.glb", **kw) -> Path:
        path = self.root / name
        path.write_bytes(glb(kw.pop("parts", [cube(), cube(0.9, inwards=True)]), **kw))
        return path

    def test_the_texture_gives_each_triangle_its_colour(self):
        mesh = printing.load_glb(self.save(parts=[cube()]))
        self.assertEqual((mesh.vertices.shape, mesh.faces.shape), ((36, 3), (12, 3)))
        upper = mesh.vertices[mesh.faces][:, :, 1].sum(axis=1) > 2
        self.assertEqual(int(upper.sum()), 2)
        self.assertTrue((mesh.colors[upper] == RED).all() and (mesh.colors[~upper] == BLUE).all())

    def test_without_a_texture_the_materials_colour(self):
        mesh = printing.load_glb(self.save(parts=[cube()], texture=False))
        self.assertTrue(np.allclose(mesh.colors, (0, 255, 0)))

    def test_not_a_glb(self):
        path = self.root / "notes.glb"
        path.write_text("hello, this is not a mesh")
        with self.assertRaisesRegex(printing.PrintError, "not a GLB"):
            printing.load_glb(path)

    def test_a_double_shell_becomes_one_closed_solid(self):
        mesh, report = printing.solid(printing.load_glb(self.save()))
        self.assertEqual((len(mesh.vertices), len(mesh.faces)), (8, 12))      # welded, shell gone
        self.assertEqual((report.cavities, report.pieces, report.flipped), (1, 1, False))
        self.assertTrue(report.closed)
        self.assertAlmostEqual(float(np.abs(mesh.vertices).max()), 1.0)       # the outer one stayed
        kept, report = printing.solid(printing.load_glb(self.save()), keep_cavities=True)
        self.assertEqual((len(kept.faces), report.cavities, report.pieces), (24, 0, 2))

    def test_an_inside_out_mesh_is_turned(self):
        mesh, report = printing.solid(printing.load_glb(self.save(parts=[cube(inwards=True)])))
        self.assertTrue(report.flipped and report.closed)
        tri = mesh.vertices[mesh.faces]
        volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
        self.assertAlmostEqual(float(volume), 8.0)

    def test_an_open_mesh_is_reported(self):
        positions, uv = cube()
        _, report = printing.solid(printing.load_glb(self.save(parts=[(positions[:-3], uv[:-3])])))
        self.assertEqual((report.open_edges, report.closed), (3, False))

    def test_crumbs_go_only_when_asked(self):
        both = printing.load_glb(self.save(parts=[cube(), cube(0.1)]))
        both.vertices[36:] += 5                                    # a small cube, apart
        self.assertEqual(printing.solid(both)[1].pieces, 2)
        mesh, report = printing.solid(both, crumbs=0.01)
        self.assertEqual((report.pieces, report.fragments, len(mesh.faces)), (1, 1, 12))

    def test_a_skin_is_told_from_a_body(self):
        body, _ = printing.solid(printing.load_glb(self.save(parts=[cube()])))
        self.assertAlmostEqual(printing.mean_wall(body), 2 * 8 / 24)
        source = self.save(parts=[cube(), cube(0.99, inwards=True)])
        skin = printing.convert(source, self.root / "skin.3mf", keep_cavities=True, rebuilding="never")
        self.assertTrue(skin.solid.hollow and skin.solid.closed)
        self.assertLess(skin.solid.wall, 1.0)
        self.assertIn("hollow skin", " ".join(cli.print_lines(skin)[1]))
        self.assertFalse(printing.convert(source, self.root / "body.3mf").solid.hollow)

    @unittest.skipIf(skimage is None, "scipy and scikit-image are not installed")
    def test_a_mesh_that_does_not_close_is_rebuilt(self):
        # two cubes through each other and a loose triangle inside: open and crossing itself
        positions, uv = cube()
        other = [[x + 0.5 for x in p] for p in positions]
        loose = ([[-0.5, -0.5, -0.5], [-0.5, 0.4, -0.6], [0.4, -0.5, -0.5]], uv[:3])   # facing the centre
        source = self.save(parts=[(positions, uv), (other, uv), loose])
        left = printing.convert(source, self.root / "left.3mf", height=30, base=0, rebuilding="never")
        self.assertFalse(left.solid.closed or left.solid.rebuilt)
        self.assertEqual(left.solid.open_edges, 3)
        done = printing.convert(source, self.root / "done.3mf", height=30, base=0, detail=0.5,
                                smoothing=0, min_patch=0)
        report = done.solid
        self.assertTrue(report.rebuilt and report.closed)
        self.assertEqual((report.pieces, report.detail, report.hollow), (1, 0.5, False))
        self.assertTrue(np.allclose(done.size, (30, 30, 30), atol=0.5))
        # 2.5 units tall: 12 mm a unit. The two cubes, less the 1.5-unit cube they share
        volume = report.wall * self.surface(done.path) / 2
        self.assertAlmostEqual(volume / 12 ** 3, 8 + 8 - 1.5 ** 3, delta=0.4)
        self.assertGreater(done.base_area, 400)                    # the lower cube's face, flat
        colors = dict(done.colors)
        self.assertEqual(set(colors), {"#1E3CC8", "#C81E1E"})       # the colours came along
        self.assertLess(colors["#C81E1E"], 0.3)
        text = " ".join(cli.print_lines(done)[0])
        self.assertIn("rebuilt, detail under 0.50 mm lost", text)

    @unittest.skipIf(skimage is None, "scipy and scikit-image are not installed")
    def test_rebuilding_fills_a_skin_and_keeps_a_closed_mesh_as_it_is(self):
        fine = printing.convert(self.save(), self.root / "fine.3mf", height=20)
        self.assertEqual((fine.solid.rebuilt, fine.triangles), (False, 12))
        mesh = printing.place(printing.weld(printing.load_glb(self.save())), 20, base=0)
        filled, cell = printing.rebuild(mesh, 0.5)                 # both shells went in
        filled, report = printing.solid(filled)
        self.assertTrue(report.closed)
        self.assertEqual((report.cavities, report.pieces, cell), (0, 1, 0.5))
        self.assertAlmostEqual(printing.mean_wall(filled), 2 * 8000 / 2400, delta=0.1)   # a full cube

    def surface(self, path: Path) -> float:
        xml = model_of(path)
        v = np.array(re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', xml), float)
        f = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', xml), int)
        return float(printing.face_areas(printing.Mesh(v, f, None)).sum())

    def test_surfaces_that_touch_along_an_edge_are_not_one_piece(self):
        # three triangles on one edge: none of them is the neighbour of another
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [2, 5, 0]])
        a, b = printing.adjacency(faces, 6)
        self.assertEqual(sorted(zip(a.tolist(), b.tolist())), [(0, 3)])
        self.assertEqual(printing.edge_counts(faces, 6), (7, 1))

    def test_components(self):
        a, b = np.array([0, 1, 5, 6, 7]), np.array([1, 2, 6, 7, 8])
        self.assertEqual(printing.components(10, a, b).tolist(), [0, 0, 0, 3, 4, 5, 5, 5, 5, 9])
        chain = np.arange(999)
        self.assertEqual(set(printing.components(1000, chain[::-1], chain[::-1] + 1).tolist()), {0})

    def test_placed_z_up_in_mm_on_a_flat_base(self):
        mesh, _ = printing.solid(printing.load_glb(self.save(parts=[cube()])))
        mesh.vertices[:, 1] *= 2                                   # 4 tall along Y, 2 wide
        placed = printing.place(mesh, height=60, base=0)
        area = printing.base_area(placed)
        self.assertTrue(np.allclose(np.ptp(placed.vertices, axis=0), (30, 30, 60)))
        self.assertAlmostEqual(float(placed.vertices[:, 2].min()), 0.0)
        self.assertTrue(np.allclose(placed.vertices[:, :2].mean(axis=0), 0))
        self.assertAlmostEqual(area, 900.0)                        # the bottom face
        tri = placed.vertices[placed.faces]                        # turned, not mirrored
        self.assertGreater(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum(), 0)
        # a point at the bottom becomes a base when its lowest part is pressed flat
        spike = printing.Mesh(np.array([[0, 0, 0], [1, 1, .2], [-1, 1, .2], [-1, -1, .2], [1, -1, .2],
                                        [0, 0, 1.0]]),
                              np.array([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
                                        [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]]), np.zeros((8, 3)))
        self.assertEqual(printing.base_area(printing.place(spike, 10, base=0, up="z")), 0.0)
        flat = printing.place(spike, 7, base=30, up="z")            # 0.7 of it is left: 10 mm a unit
        self.assertAlmostEqual(printing.base_area(flat), 400.0)    # the 2 x 2 ring, flat
        self.assertAlmostEqual(float(np.ptp(flat.vertices[:, 2])), 7.0)

    def test_colours_reduced_most_used_first(self):
        colors = np.array([RED] * 3 + [BLUE] * 6 + [(34, 62, 205)], float)   # a blue a bit off
        palette, labels = printing.quantize(colors, np.ones(10), 2)
        self.assertEqual(labels.tolist(), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(printing.hex_color(palette[1]), "#C81E1E")
        again = printing.quantize(colors, np.ones(10), 2)
        self.assertTrue(np.array_equal(palette, again[0]) and np.array_equal(labels, again[1]))
        self.assertEqual(len(printing.quantize(colors, np.ones(10), 8)[0]), 3)   # no empty slots
        self.assertEqual(printing.nearest(colors, np.ones(10), [(0, 0, 255), (255, 0, 0)]).tolist(),
                         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    def test_a_shadow_is_not_another_colour(self):
        # dark blue is closer to blue than to grey, however dark: lightness counts less
        dark_blue, blue, grey = (15, 30, 100), (40, 80, 220), (120, 120, 120)
        labels = printing.nearest(np.array([dark_blue], float), np.ones(1), [grey, blue])
        self.assertEqual(labels.tolist(), [1])

    def test_specks_take_the_colour_around_them(self):
        # a strip of 6 triangles, one of them the odd colour
        faces = np.array([[i, i + 1, i + 2] for i in range(6)])
        a, b = printing.adjacency(faces, 8)
        labels = np.array([0, 0, 1, 0, 0, 0])
        self.assertEqual(printing.smooth(labels, a, b, 2, 3).tolist(), [0] * 6)
        self.assertEqual(printing.smooth(labels, a, b, 2, 0).tolist(), labels.tolist())
        halves = np.array([0, 0, 0, 1, 1, 1])                      # a real border stays
        self.assertEqual(printing.smooth(halves, a, b, 2, 3).tolist(), halves.tolist())
        areas = np.ones(6)
        self.assertEqual(printing.absorb(halves, a, b, areas, 2, 2.0).tolist(), halves.tolist())
        patch = np.array([0, 0, 0, 0, 1, 1])
        self.assertEqual(printing.absorb(patch, a, b, areas, 2, 2.5).tolist(), [0] * 6)

    def test_the_3mf(self):
        target = self.root / "cube.3mf"
        done = printing.convert(self.save(), target, height=40, base=0, colors=4,
                                smoothing=0, min_patch=0)
        self.assertEqual((done.triangles, done.solid.closed, done.solid.cavities), (12, True, 1))
        self.assertTrue(np.allclose(done.size, (40, 40, 40)))
        self.assertEqual([c for c, _ in done.colors], ["#1E3CC8", "#C81E1E"])   # two is all there is
        self.assertAlmostEqual(sum(s for _, s in done.colors), 1.0)
        with zipfile.ZipFile(target) as z:
            self.assertEqual(set(z.namelist()), {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"})
        xml = model_of(target)
        self.assertIn('unit="millimeter"', xml)
        self.assertEqual(re.findall(r'displaycolor="(#\w+)"', xml), ["#1E3CC8", "#C81E1E"])
        self.assertEqual((xml.count("<vertex "), xml.count("<triangle ")), (8, 12))
        # the red face is filament 2, both as Bambu paint and as a 3MF material; the rest
        # is the object's own filament and says nothing
        self.assertEqual(xml.count(' pid="1" p1="1" paint_color="8"/>'), 2)
        self.assertEqual(xml.count("paint_color"), 2)
        heights = [float(z) for z in re.findall(r' z="([-\d.]+)"', xml)]
        self.assertEqual((min(heights), max(heights)), (0.0, 40.0))

    def test_a_palette_keeps_its_slots(self):
        done = printing.convert(self.save(), self.root / "p.3mf", height=40, smoothing=0, min_patch=0,
                                palette=[(0, 0, 255), (255, 255, 255), (255, 0, 0)])
        self.assertEqual([c for c, _ in done.colors], ["#0000FF", "#FFFFFF", "#FF0000"])
        self.assertEqual([round(s, 2) for _, s in done.colors], [0.83, 0.0, 0.17])
        self.assertEqual(model_of(done.path).count('paint_color="0C"'), 2)     # red: slot 3
        # smoothing gives two lone triangles to the colour around them
        smoothed = printing.convert(self.save(), self.root / "s.3mf", palette=[(0, 0, 255), (255, 0, 0)])
        self.assertEqual([round(s, 2) for _, s in smoothed.colors], [1.0, 0.0])

    def test_bad_options(self):
        source = self.save()
        for options in ({"height": 0}, {"base": 80}, {"colors": 0}, {"colors": 17},
                        {"rebuilding": "maybe"}, {"detail": 0}):
            with self.assertRaises(printing.PrintError):
                printing.convert(source, self.root / "x.3mf", **options)
        self.assertFalse((self.root / "x.3mf").exists())

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue() + err.getvalue()

    def test_the_command_needs_no_project_and_never_overwrites(self):
        source = self.save("owl.glb")
        before = os.getcwd()
        self.addCleanup(os.chdir, before)
        os.chdir(self.root)                                        # no kami.toml, no .env here
        code, text = self.run_cli("print", str(source), "--height", "50", "--colors", "2", "--base", "0",
                                  "--smooth", "0", "--min-patch", "0")
        self.assertEqual(code, 0, text)
        self.assertTrue((self.root / "owl.3mf").is_file())
        for expected in ("50.0 x 50.0 x 50.0 mm", "closed", "1 inner shell removed",
                         "slot 1   #1E3CC8", "83% of the surface", "slot 2   #C81E1E", "17% of the surface"):
            self.assertIn(expected, text)
        self.assertEqual(self.run_cli("print", str(source))[0], 0)
        self.assertTrue((self.root / "owl-2.3mf").is_file())
        code, text = self.run_cli("print", str(self.root / "missing.glb"))
        self.assertEqual(code, 1)
        self.assertIn("No such file", text)
        code, text = self.run_cli("print", str(source), "--palette", "blue")
        self.assertEqual(code, 1)
        self.assertIn("Not a colour", text)

    def test_the_page_prepares_and_saves_a_print(self):
        from kami import webui
        source = self.save("owl.glb")
        printer = webui.Printer(self.root)
        with unittest.mock.patch.object(printing, "prepare", wraps=printing.prepare) as prepare:
            shown = printer.run({"file": "owl.glb", "height": 40, "base": 0, "colors": 2})
            self.assertEqual((shown["saved"], shown["closed"], shown["cavities"]), (None, True, 1))
            self.assertEqual(shown["size"], [40.0, 40.0, 40.0])
            self.assertEqual([c["hex"] for c in shown["colors"]], ["#1E3CC8", "#C81E1E"])
            self.assertFalse(list(self.root.glob("*.3mf")))                    # looking saves nothing
            name, painted = printer.preview
            self.assertEqual(shown["preview"], f"preview/{name}.glb")
            back = printing.load_glb(self.write("painted.glb", painted))       # a GLB of plain colours
            self.assertEqual(len(back.faces), 12 + 12)                          # the model, and the plate under it
            self.assertAlmostEqual(float(back.vertices[:, 1].max()), 0.040, places=5)    # Y up, metres
            self.assertAlmostEqual(float(np.ptp(back.vertices[:, 0])), printing.PLATE_MM / 1000, places=5)
            self.assertEqual(shown["plate"], printing.PLATE_MM)
            mine = printer.run({"file": "owl.glb", "height": 40, "base": 0, "palette": ["#0000ff", "#fff"],
                                "save": True})
            self.assertEqual(prepare.call_count, 1)                           # only the colours changed
            self.assertEqual((mine["saved"], [c["hex"] for c in mine["colors"]]), ("owl.3mf", ["#0000FF", "#FFFFFF"]))
            self.assertIn('displaycolor="#0000FF"', model_of(self.root / "owl.3mf"))
            self.assertEqual(printer.run({"file": "owl.glb", "save": True})["saved"], "owl-2.3mf")
            self.assertEqual(prepare.call_count, 2)                           # another height
            # every setting of `kami print` can come from the page
            kept = printer.run({"file": "owl.glb", "height": 40, "base": 0, "keep_cavities": True,
                                "smoothing": 0, "min_patch": 0, "colors": 2})
            self.assertEqual((prepare.call_count, kept["cavities"], kept["triangles"]), (3, 0, 24))
            self.assertEqual([round(c["share"], 2) for c in kept["colors"]], [0.83, 0.17])   # both shells, their red faces too
            with self.assertRaises(ValueError):
                printer.run({"file": "owl.glb", "smoothing": 1.5})
        for bad in ({"file": "missing.glb"}, {"file": "../owl.glb"}, {"file": "owl.glb", "colors": "2"},
                    {"file": "owl.glb", "palette": "#fff"}, {"file": "owl.glb", "height": "tall"}, []):
            with self.assertRaises(ValueError):
                printer.run(bad)
        with self.assertRaises(printing.PrintError):
            printer.run({"file": "owl.glb", "palette": ["blue"]})
        self.assertIsNotNone(source)

    def write(self, name: str, data: bytes) -> Path:
        path = self.root / "scratch" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        return path

    def test_a_mesh_gets_a_thumbnail(self):
        from kami import webui
        self.save("owl.glb")
        printer = webui.Printer(self.root)
        picture = printer.thumbnail("owl.glb")
        image = np.asarray(Image.open(io.BytesIO(picture)))
        self.assertEqual(image.shape, (256, 256, 4))
        self.assertEqual(tuple(image[0, 0]), (0, 0, 0, 0))                    # clear around it
        self.assertGreater(int((image[..., 3] > 0).sum()), 256 * 256 // 10)
        self.assertIs(printer.thumbnail("owl.glb"), picture)                  # made once
        (self.root / "broken.glb").write_text("not a mesh")
        for name in ("broken.glb", "missing.glb", "../owl.glb"):
            self.assertIsNone(printer.thumbnail(name))

    def test_the_size_is_the_longest_side_or_the_height(self):
        mesh, _ = printing.solid(printing.load_glb(self.save(parts=[cube()])))
        mesh.vertices[:, 0] *= 3                                   # 6 long, 2 wide, 2 tall: lying down
        tall = printing.place(mesh, 40, base=0)
        self.assertTrue(np.allclose(np.ptp(tall.vertices, axis=0), (120, 40, 40)))
        long = printing.place(mesh, 40, base=0, fit="longest")
        self.assertTrue(np.allclose(np.ptp(long.vertices, axis=0), (40, 40 / 3, 40 / 3)))
        with self.assertRaises(printing.PrintError):
            printing.place(mesh, 40, fit="width")
        # the page measures the longest side unless told otherwise; the command too
        from kami import webui
        path = self.write("long.glb", glb([cube()]))
        printer = webui.Printer(path.parent)
        self.assertEqual(max(printer.run({"file": "long.glb", "height": 30})["size"]), 30.0)
        self.assertEqual(printer.run({"file": "long.glb", "height": 30, "fit": "height"})["size"][2], 30.0)
        before = os.getcwd()
        self.addCleanup(os.chdir, before)
        os.chdir(self.root)
        source = self.save("lying.glb", parts=[cube()])
        self.assertEqual(self.run_cli("print", str(source), "--size", "30", "--base", "0")[0], 0)
        self.assertEqual(self.run_cli("print", str(source), "--height", "30", "--size", "30")[0], 2)   # one or the other

    def test_turned_so_its_front_faces_the_plates_front(self):
        mesh, _ = printing.solid(printing.load_glb(self.save(parts=[cube()])))
        mesh.vertices[:, 0] *= 3                                   # 6 long along X
        mesh.vertices[:, 2] += (mesh.vertices[:, 0] > 0) * 0.5     # a nose: its +X end leans to the front (+Z)
        before = printing.place(mesh, 40, base=0, fit="longest")
        left = printing.place(mesh, 40, base=0, fit="longest", turn=90)
        self.assertTrue(np.allclose(np.ptp(left.vertices, axis=0)[[1, 0, 2]], np.ptp(before.vertices, axis=0)))
        nose = before.vertices[:, 0].argmax()
        self.assertTrue(np.allclose(left.vertices[nose, :2], [-before.vertices[nose, 1], before.vertices[nose, 0]]))
        tri = left.vertices[left.faces]                            # turned, not mirrored
        self.assertGreater(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum(), 0)
        self.assertTrue(np.allclose(printing.turned(before, 90).vertices, left.vertices))
        self.assertTrue(np.allclose(printing.turned(before, 0).vertices, before.vertices))
        for turn in (45, -90, 360):
            with self.assertRaises(printing.PrintError):
                printing.place(mesh, 40, turn=turn)
        # the page turns what it prepared, without preparing it again
        from kami import webui
        path = self.write("long.glb", glb([cube()]))
        printer = webui.Printer(path.parent)
        with unittest.mock.patch.object(printing, "prepare", wraps=printing.prepare) as prepare:
            flat = printer.run({"file": "long.glb", "height": 30})["size"]
            turned = printer.run({"file": "long.glb", "height": 30, "turn": 90})["size"]
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(turned, [flat[1], flat[0], flat[2]])
        with self.assertRaises(ValueError):
            printer.run({"file": "long.glb", "turn": 45})
        before = os.getcwd()
        self.addCleanup(os.chdir, before)
        os.chdir(self.root)
        source = self.save("lying.glb", parts=[cube()])
        self.assertEqual(self.run_cli("print", str(source), "--turn", "270", "--base", "0")[0], 0)
        self.assertEqual(self.run_cli("print", str(source), "--turn", "45")[0], 2)

    def test_warnings(self):
        report = printing.Solid(pieces=2, open_edges=4)
        done = printing.Printed(Path("x.3mf"), 10, (300.0, 20.0, 80.0), 1.0, report, [("#FFFFFF", 1.0)])
        self.assertEqual(cli.print_lines(done)[1], printing.warnings(done))
        warnings = " ".join(cli.print_lines(done)[1])
        for expected in ("not closed (4 open edges", "2 separate pieces", "256 mm plate", "barely touches"):
            self.assertIn(expected, warnings)
        fine = printing.Printed(Path("x.3mf"), 10, (40.0, 40.0, 80.0), 900.0, printing.Solid(), [])
        self.assertEqual(cli.print_lines(fine)[1], [])


if __name__ == "__main__":
    unittest.main()
