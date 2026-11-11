#!/usr/bin/env python3
"""Writes tests/fixtures/image: PNG and JPEG files and what Pillow decodes
them to, for image_test (src/core/image.hpp) and the model's to_rgb test.

The PNGs are written here byte by byte (every colour type and bit depth,
tRNS, Adam7 interlacing, a gAMA chunk, split IDATs), the JPEGs by Pillow's
encoder (libjpeg-turbo). For each case `expected.txt` lists the file, its
size, and the files holding Pillow's decode:

  <name>.rgb    Image.open(f).convert("RGB").tobytes()
  <name>.alpha  the alpha channel, for images Pillow opens as RGBA
  <name>.torgb  qwen_vl_utils' to_rgb (RGBA composited on white), RGBA only

and `error` for files Pillow cannot decode either.

  python3 tools/reference/image_fixtures.py [OUT_DIR]
"""

import io
import struct
import sys
import zlib
from pathlib import Path

from PIL import Image

WIDTH, HEIGHT = 37, 29  # odd, so JPEG MCUs and PNG bit packing have edges


class Lcg:
    def __init__(self, seed: int):
        self.state = seed

    def next(self) -> int:
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state >> 8


def pattern(channels: int, maxval: int, seed: int):
    """Rows of pixel tuples: gradients plus noise, so every value range and
    the JPEG's high frequencies are exercised."""
    rng = Lcg(seed)
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            px = []
            for c in range(channels):
                base = (x * (c + 1) * 7 + y * (3 - c % 3) * 11) / (WIDTH * 9 + HEIGHT * 33)
                noise = rng.next() % (maxval // 4 + 1)
                px.append(int(base * maxval * 0.75 + noise) % (maxval + 1))
            row.append(tuple(px))
        rows.append(row)
    return rows


def pack_row(row, bit: int) -> bytes:
    out = bytearray()
    if bit == 16:
        for px in row:
            for s in px:
                out += struct.pack(">H", s)
    elif bit == 8:
        for px in row:
            out += bytes(px)
    else:
        acc, nbits = 0, 0
        for px in row:
            for s in px:
                acc = (acc << bit) | s
                nbits += bit
                if nbits == 8:
                    out.append(acc)
                    acc, nbits = 0, 0
        if nbits:
            out.append(acc << (8 - nbits))
    return bytes(out)


ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data +
            struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_png(rows, bit: int, ctype: int, *, plte=None, trns=None,
              interlace=False, extra=(), split_idat=False,
              size=None) -> bytes:
    width, height = size or (len(rows[0]), len(rows))
    raw = bytearray()
    if interlace:
        for x0, y0, dx, dy in ADAM7:
            for y in range(y0, height, dy):
                sub = [rows[y][x] for x in range(x0, width, dx)]
                if sub:
                    raw += b"\0" + pack_row(sub, bit)
    else:
        for row in rows:
            raw += b"\0" + pack_row(row, bit)
    data = zlib.compress(bytes(raw), 9)
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit, ctype,
                                      0, 0, 1 if interlace else 0))
    for kind, payload in extra:
        out += chunk(kind, payload)
    if plte is not None:
        out += chunk(b"PLTE", plte)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    if split_idat:
        half = len(data) // 2
        out += chunk(b"IDAT", data[:half]) + chunk(b"IDAT", data[half:])
    else:
        out += chunk(b"IDAT", data)
    out += chunk(b"IEND", b"")
    return out


def png_cases():
    cases = {}
    for bit in (1, 2, 4, 8, 16):
        cases[f"gray{bit}"] = write_png(pattern(1, (1 << bit) - 1, bit), bit, 0)
    cases["gray8_trns"] = write_png(pattern(1, 255, 21), 8, 0,
                                    trns=struct.pack(">H", 40))
    cases["gray4_adam7"] = write_png(pattern(1, 15, 22), 4, 0, interlace=True)
    cases["graya8"] = write_png(pattern(2, 255, 23), 8, 4)
    cases["graya16"] = write_png(pattern(2, 65535, 24), 16, 4)
    cases["rgb8"] = write_png(pattern(3, 255, 25), 8, 2)
    cases["rgb8_trns"] = write_png(pattern(3, 255, 26), 8, 2,
                                   trns=struct.pack(">HHH", 10, 20, 30))
    cases["rgb8_adam7"] = write_png(pattern(3, 255, 27), 8, 2, interlace=True)
    # gAMA is ignored by Pillow (and so by the decoder); IDAT may be split.
    cases["rgb8_gama"] = write_png(pattern(3, 255, 28), 8, 2,
                                   extra=[(b"gAMA", struct.pack(">I", 45455))],
                                   split_idat=True)
    cases["rgb16"] = write_png(pattern(3, 65535, 29), 16, 2)
    cases["rgba8"] = write_png(pattern(4, 255, 30), 8, 6)
    cases["rgba8_adam7"] = write_png(pattern(4, 255, 31), 8, 6, interlace=True)
    cases["rgba16"] = write_png(pattern(4, 65535, 32), 16, 6)
    for bit in (1, 4, 8):
        colours = 1 << bit
        plte = bytes((i * 37 + c * 91) % 256 for i in range(colours) for c in range(3))
        trns = bytes((i * 53) % 256 for i in range(colours)) if bit == 8 else None
        name = f"pal{bit}" + ("_trns" if trns else "")
        cases[name] = write_png(pattern(1, colours - 1, 40 + bit), bit, 3,
                                plte=plte, trns=trns)
    return cases


def jpeg_cases():
    rgb = Image.frombytes("RGB", (WIDTH, HEIGHT),
                          b"".join(bytes(px) for row in pattern(3, 255, 50) for px in row))
    gray = rgb.convert("L")
    cases = {}

    def save(name, image, **options):
        buf = io.BytesIO()
        image.save(buf, "JPEG", **options)
        cases[name] = buf.getvalue()

    save("jpeg420", rgb, quality=85, subsampling=2)
    save("jpeg422", rgb, quality=90, subsampling=1)
    save("jpeg444", rgb, quality=95, subsampling=0)
    save("jpeg420_progressive", rgb, quality=80, subsampling=2, progressive=True)
    save("jpeg_gray", gray, quality=85)
    save("jpeg_cmyk", rgb.convert("CMYK"), quality=85)  # unsupported here
    return cases


def to_rgb(pil_image):
    """qwen_vl_utils.vision_process.to_rgb (0.0.14)."""
    if pil_image.mode == "RGBA":
        white = Image.new("RGB", pil_image.size, (255, 255, 255))
        white.paste(pil_image, mask=pil_image.split()[3])
        return white
    return pil_image.convert("RGB")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "image")
    out.mkdir(parents=True, exist_ok=True)
    files = {f"{k}.png": v for k, v in png_cases().items()}
    files.update({f"{k}.jpg": v for k, v in jpeg_cases().items()})
    # Files Pillow rejects too.
    files["truncated.png"] = files["rgb8.png"][: len(files["rgb8.png"]) // 2]
    files["truncated.jpg"] = files["jpeg420.jpg"][: len(files["jpeg420.jpg"]) // 2]
    files["not_an_image.png"] = b"GIF89a, or anything else\n"
    # 400 MP claimed in the header, a few bytes of data: above any pixel cap.
    files["huge.png"] = write_png([[(0,)]], 8, 0, size=(20000, 20000))
    lines = ["# file width height outputs (rgb [alpha torgb]) | error"]
    for name, data in sorted(files.items()):
        (out / name).write_bytes(data)
        stem = name.rsplit(".", 1)[0]
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            if image.mode == "CMYK":
                raise ValueError("CMYK is not decoded here")
        except Exception as e:  # noqa: BLE001 - any failure is an error case
            lines.append(f"{name} error")
            print(f"{name}: error ({e})")
            continue
        (out / f"{stem}.rgb").write_bytes(image.convert("RGB").tobytes())
        outputs = "rgb"
        if image.mode == "RGBA":
            (out / f"{stem}.alpha").write_bytes(image.getchannel("A").tobytes())
            (out / f"{stem}.torgb").write_bytes(to_rgb(image).tobytes())
            outputs = "rgb alpha torgb"
        lines.append(f"{name} {image.width} {image.height} {outputs}")
        print(f"{name}: {image.mode} {image.size}")
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    print(f"Pillow {Image.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
