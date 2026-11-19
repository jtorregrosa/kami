"""Build check of the kami-server image: Pillow writes and reads back every format kami
sends (kami/jobs.py IMAGE_SUFFIXES), so a broken Pillow fails the build, not a pod."""
import io

import PIL
from PIL import Image, features
from pillow_heif import register_heif_opener

register_heif_opener()
missing = [f for f in ("webp", "avif", "jpg", "zlib", "libtiff") if not features.check(f)]
assert not missing, f"Pillow without {missing}"

source = Image.new("RGBA", (32, 24), (200, 120, 40, 255))
read = []
for fmt in ("PNG", "JPEG", "WEBP", "AVIF", "HEIF", "TIFF", "BMP", "GIF"):
    image = source if fmt in ("PNG", "WEBP", "TIFF") else source.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, fmt)
    back = Image.open(io.BytesIO(buf.getvalue()))
    back.load()
    assert back.size == (32, 24), (fmt, back.size)
    read.append(fmt)
print("Pillow", PIL.__version__, "reads", ", ".join(read))
