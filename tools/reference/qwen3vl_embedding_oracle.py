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
"""

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
