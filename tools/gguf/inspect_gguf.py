#!/usr/bin/env python3
"""Print a GGUF file's metadata and tensor table, from a path or an HTTP URL.

URLs are read with HTTP range requests, so only the header is downloaded.

  inspect_gguf.py model.gguf
  inspect_gguf.py https://huggingface.co/<repo>/resolve/<rev>/<file>.gguf --tensors
"""

import argparse
import struct
import sys
import urllib.request

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Source:
    """Sequential reader over a local file or an HTTP URL (range requests)."""

    def __init__(self, location: str, chunk: int = 4 << 20):
        self.location = location
        self.remote = location.startswith(("http://", "https://"))
        self.chunk = chunk
        self.buffer = b""
        self.base = 0
        self.pos = 0
        self.file = None if self.remote else open(location, "rb")

    def _fill(self, need: int) -> None:
        while self.pos + need > self.base + len(self.buffer):
            start = self.base + len(self.buffer)
            if self.remote:
                req = urllib.request.Request(
                    self.location,
                    headers={"Range": f"bytes={start}-{start + self.chunk - 1}"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
            else:
                self.file.seek(start)
                data = self.file.read(self.chunk)
            if not data:
                raise EOFError("unexpected end of GGUF header")
            self.buffer += data

    def read(self, n: int) -> bytes:
        self._fill(n)
        off = self.pos - self.base
        self.pos += n
        return self.buffer[off:off + n]

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


def read_string(src: Source) -> str:
    return src.read(src.unpack("<Q")).decode("utf-8", errors="replace")


def read_value(src: Source, kind: int):
    if kind in SCALARS:
        return src.unpack(SCALARS[kind])
    if kind == 8:
        return read_string(src)
    if kind == 9:
        item_kind = src.unpack("<I")
        count = src.unpack("<Q")
        return [read_value(src, item_kind) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {kind}")


def parse(location: str) -> dict:
    src = Source(location)
    if src.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = src.unpack("<I")
    n_tensors = src.unpack("<Q")
    n_kv = src.unpack("<Q")
    metadata = {}
    for _ in range(n_kv):
        key = read_string(src)
        metadata[key] = read_value(src, src.unpack("<I"))
    tensors = []
    for _ in range(n_tensors):
        name = read_string(src)
        dims = [src.unpack("<Q") for _ in range(src.unpack("<I"))]
        kind = src.unpack("<I")
        offset = src.unpack("<Q")
        tensors.append({"name": name, "shape": dims,
                        "type": GGML_TYPES.get(kind, str(kind)), "offset": offset})
    return {"version": version, "metadata": metadata, "tensors": tensors,
            "header_bytes": src.pos}


def short(value, limit: int = 8):
    if isinstance(value, list) and len(value) > limit:
        return f"[{', '.join(map(repr, value[:limit]))}, ... ({len(value)} items)]"
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("location")
    parser.add_argument("--tensors", action="store_true", help="list every tensor")
    args = parser.parse_args()
    info = parse(args.location)
    print(f"GGUF v{info['version']}, {len(info['tensors'])} tensors, "
          f"header {info['header_bytes']} bytes")
    for key, value in info["metadata"].items():
        print(f"  {key} = {short(value)}")
    types = {}
    for tensor in info["tensors"]:
        types[tensor["type"]] = types.get(tensor["type"], 0) + 1
    print("tensor types:", dict(sorted(types.items())))
    if args.tensors:
        for tensor in info["tensors"]:
            print(f"  {tensor['name']:<40} {tensor['type']:<6} {tensor['shape']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
