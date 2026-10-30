#!/usr/bin/env python3
"""Write tokenizer fixtures (text -> ids) using HF `tokenizers` as the oracle.

  python -I tokenizer_fixtures.py --model /models/Qwen3.5-2B-hf \
      --out tests/fixtures/tokenizer/qwen35.jsonl
  python -I tokenizer_fixtures.py --model /models/Qwen3-VL-Embedding-2B \
      --out tests/fixtures/tokenizer/qwen3vl.jsonl
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and trailing   ",
    "Line one\nLine two\r\n\n\nTabs\there",
    "I'm sure you'll see we've done it, they'd say. DON'T SHOUT",
    "Numbers: 0 7 42 3.14159 1,000,000 and 2026-10-27",
    "def add(a, b):\n    return a + b  # comment\n\n\tif x:\n\t\tpass",
    "Ünïcödé café naïve résumé — “quotes” ‘single’ … ©®™",
    "e\u0301 (decomposed é) and \u212b (angstrom sign)",
    "日本語のテキストと中文文本，还有한국어 텍스트.",
    "Emoji: 😀👍🏽 family 👨‍👩‍👧 flags 🇺🇸",
    "Arabic: مرحبا بالعالم, Hebrew: שלום, Hindi: नमस्ते",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "Mixed<|im_end|>special<|endoftext|>tokens",
    "URL https://example.com/path?q=1&b=two#frag and e-mail a.b@c.de",
    "!!!??? ... --- ___ *** ### @@@ $$$ %%% ^^^",
    "x" * 300,
    "",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for text in TEXTS:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            # Decoding gives the NFC-normalised text, not always the input.
            decoded = tokenizer.decode(ids)
            out.write(json.dumps({"text": text, "ids": ids, "decoded": decoded},
                                 ensure_ascii=False) + "\n")
    print(f"wrote {len(TEXTS)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
