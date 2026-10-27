#!/usr/bin/env python3
"""Write dequantization fixtures from real GGUF blocks, using gguf-py as oracle.

For each ggml type, takes the first `--elements` values of the first tensor of
that type and writes <type>.bin (raw blocks) and <type>.f32 (gguf-py's
dequantized fp32). tests/core/quant_test.cpp checks the C++ dequantizer
against them bit for bit.

  pip install gguf numpy
  make_quant_fixtures.py --out tests/fixtures/quant model-Q4.gguf model-BF16.gguf
"""

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

WANTED = {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=512)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in args.files:
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            name = tensor.tensor_type.name
            if name not in WANTED or name in done:
                continue
            block, block_bytes = GGML_QUANT_SIZES[tensor.tensor_type]
            n = args.elements - args.elements % block
            raw = np.asarray(tensor.data).reshape(-1).view(np.uint8)
            raw = raw[: n // block * block_bytes].copy()
            values = dequantize(raw, GGMLQuantizationType[name]).astype(np.float32)
            assert values.size == n, (name, values.size, n)
            (args.out / f"{name}.bin").write_bytes(raw.tobytes())
            (args.out / f"{name}.f32").write_bytes(values.tobytes())
            print(f"{name}: {n} values from {tensor.name} ({path.name})")
            done.add(name)
    missing = WANTED - done - {"F16"}
    if missing:
        print("missing types:", sorted(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
