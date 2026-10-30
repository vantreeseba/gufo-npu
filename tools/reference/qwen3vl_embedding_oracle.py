#!/usr/bin/env python3
"""Reference embeddings for Qwen3-VL-Embedding-2B's tests, from HF
transformers in float32 (the model's own weights, widened).

  python -I qwen3vl_embedding_oracle.py --model /models/Qwen3-VL-Embedding-2B \\
      --out qwen3vl_embed_2b_oracle.json

Each case is embedded as the model's reference embedder does it
(scripts/qwen3_vl_embedding.py in the model repo): the instruction as the
system message (trimmed, '.' appended unless it ends in punctuation), the
text as the user message, the assistant prompt, the tokenizer's trailing
<|endoftext|>; the embedding is that last token's final hidden state,
L2-normalized. The output (JSON) holds each case's text, instruction, token
ids and embedding, and a small retrieval set: which document each query
ranks first. Point GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE at it.

With --images it also embeds synthetic test cards (generated here, no
downloads) as the reference embedder embeds an image: qwen_vl_utils'
smart_resize (factor 32, 4096 to 1843200 pixels) and Pillow's resize, then
the processor with do_resize=False, the image before the text in the user
message. Those cases ("image_cases") hold the raw RGB8 pixels, the grid,
token ids and embedding, for qwen3vl_embed_2b_vision_oracle_test.
"""

import argparse
import json
import math
from pathlib import Path
import unicodedata

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

QUERY = "Retrieve passages that answer the question."
DOCUMENTS = [
    "The receipt from Corner Hardware lists two boxes of screws and a hammer, "
    "total 23.40 EUR, paid by card on 12 March 2025.",
    "Our lease renews every September; the landlord must be told in writing "
    "two months before if we intend to leave.",
    "To reset the router, hold the button on its back for ten seconds until "
    "the lights blink, then wait two minutes.",
    "The NPU in the Ryzen AI MAX+ 395 reads weights from DDR at about 56 GB/s, "
    "which bounds how fast a large model decodes.",
    "Grandma's apple cake: four apples, two eggs, 200 g flour, 150 g sugar, "
    "bake 45 minutes at 180 degrees.",
    "Der Mietvertrag verlängert sich jedes Jahr im September automatisch.",
]
QUERIES = [
    "How much did I spend at the hardware store?",
    "When do I have to give notice on the apartment?",
    "What limits the decode speed on the NPU?",
]
# Which document each query should rank first (checked here, so a broken
# oracle run fails early).
EXPECTED = [0, 1, 3]

LONG = " ".join(
    f"Section {i}: the archive keeps receipts, letters and notes, and each is "
    f"searched by meaning and by exact words; entry {i * 7 % 13} links to "
    f"entry {i * 11 % 17}."
    for i in range(60)
)  # ~1,500 tokens: crosses a 1,024-row GEMM chunk

EXTRA = [
    ("default instruction", "Hello, world!", ""),
    ("custom instruction, no period", "Find the invoice", "Represent the query"),
    ("instruction with whitespace", "spaces", "  Classify the topic!  "),
    ("unicode", "Ünïcödé café — 日本語のテキスト, emoji 😀👍🏽", ""),
    ("code", "def add(a, b):\n    return a + b\n", ""),
    ("long", LONG, ""),
]


def normalize_instruction(instruction: str) -> str:
    instruction = instruction.strip()
    if not instruction:
        return "Represent the user's input."
    if not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


# (name, width, height, seed, text, instruction)
IMAGES = [
    ("image", 96, 64, 0, "", ""),  # kept as is (6,144 pixels)
    ("image and text", 128, 96, 1, "A test card with two gradients.", ""),
    ("resized image", 100, 70, 2, "", "Represent the image"),  # to 96 x 64
]
IMAGE_FACTOR = 32
IMAGE_MIN_PIXELS = 4 * 32 * 32
IMAGE_MAX_PIXELS = 1800 * 32 * 32


def test_card(width: int, height: int, seed: int) -> bytes:
    """Deterministic RGB8: two gradients, a checkerboard, scattered dots."""
    px = bytearray()
    for y in range(height):
        for x in range(width):
            r = x * 255 // max(width - 1, 1)
            g = y * 255 // max(height - 1, 1)
            b = 30 + 200 * ((x // 8 + y // 8 + seed) % 2)
            if (x * 7 + y * 3 + seed) % 23 == 0:
                r, g, b = 255 - r, 255 - g, 255 - b
            px += bytes((r, g, b))
    return bytes(px)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """qwen_vl_utils.vision_process.smart_resize with the embedder's bounds."""
    f = IMAGE_FACTOR
    h = max(f, round(height / f) * f)
    w = max(f, round(width / f) * f)
    if h * w > IMAGE_MAX_PIXELS:
        beta = math.sqrt((height * width) / IMAGE_MAX_PIXELS)
        h = math.floor(height / beta / f) * f
        w = math.floor(width / beta / f) * f
    elif h * w < IMAGE_MIN_PIXELS:
        beta = math.sqrt(IMAGE_MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / f) * f
        w = math.ceil(width * beta / f) * f
    return h, w


def image_cases(model_dir: Path, model) -> list[dict]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    end = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    cases = []
    for name, width, height, seed, text, instruction in IMAGES:
        rgb = test_card(width, height, seed)
        image = Image.frombytes("RGB", (width, height), rgb)
        rh, rw = smart_resize(height, width)
        image = image.resize((rw, rh))  # Pillow's default: bicubic
        content = [{"type": "image", "image": image}]
        if text:
            content.append({"type": "text", "text": text})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": normalize_instruction(instruction)}],
            },
            {"role": "user", "content": content},
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        expected = (
            f"<|im_start|>system\n{normalize_instruction(instruction)}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        if prompt != expected:
            raise SystemExit(f"{name}: the chat template renders {prompt!r}")
        inputs = processor(
            text=[prompt], images=[image], do_resize=False, return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()
        if ids[-1] != end:
            raise SystemExit(f"{name}: the tokenizer did not append <|endoftext|>")
        with torch.no_grad():
            hidden = model.model(**inputs).last_hidden_state
        embedding = F.normalize(hidden[0, -1].float(), p=2, dim=-1)
        grid = inputs["image_grid_thw"][0].tolist()
        cases.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "rgb": list(rgb),
                "text": text,
                "instruction": instruction,
                "grid": grid,
                "ids": ids,
                "embedding": embedding.tolist(),
            }
        )
        print(f"{name}: grid {grid}, {len(ids)} tokens")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--images", action="store_true", help="also embed synthetic images"
    )
    args = parser.parse_args()

    # The tokenizer carries the chat template; the processor would also
    # need the vision stack, which text does not use.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32
    ).eval()

    cases = []
    for text in QUERIES:
        cases.append(("query", text, QUERY))
    for text in DOCUMENTS:
        cases.append(("document", text, ""))
    cases.extend(EXTRA)

    out = {"model": str(args.model.name), "dtype": "float32", "cases": []}
    embeddings = []
    for name, text, instruction in cases:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": normalize_instruction(instruction)}],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # What gufo-npu builds by hand (models/qwen3vl_embed_2b/input.cpp).
        expected = (
            f"<|im_start|>system\n{normalize_instruction(instruction)}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        if prompt != expected:
            raise SystemExit(f"{name}: the chat template renders {prompt!r}")
        ids = tokenizer(prompt)["input_ids"]  # appends <|endoftext|>
        if ids[-1] != tokenizer.convert_tokens_to_ids("<|endoftext|>"):
            raise SystemExit(f"{name}: the tokenizer did not append <|endoftext|>")
        with torch.no_grad():
            hidden = model.model(input_ids=torch.tensor([ids])).last_hidden_state
        embedding = F.normalize(hidden[0, -1].float(), p=2, dim=-1)
        embeddings.append(embedding)
        out["cases"].append(
            {
                "name": name,
                "text": text,
                "instruction": instruction,
                "prompt": prompt,
                "ids": ids,
                "embedding": embedding.tolist(),
            }
        )
        print(f"{name}: {len(ids)} tokens")

    queries = list(range(len(QUERIES)))
    documents = list(range(len(QUERIES), len(QUERIES) + len(DOCUMENTS)))
    best = []
    for q in queries:
        scores = [float(embeddings[q] @ embeddings[d]) for d in documents]
        best.append(max(range(len(documents)), key=scores.__getitem__))
        print(f"query {q}: " + " ".join(f"{s:.3f}" for s in scores))
    if best != EXPECTED:
        raise SystemExit(f"retrieval ranks {best}, expected {EXPECTED}")
    out["retrieval"] = {"queries": queries, "documents": documents, "best": best}
    if args.images:
        out["image_cases"] = image_cases(args.model, model)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
