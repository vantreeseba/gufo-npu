#!/usr/bin/env python3
"""HF transformers oracle for the Qwen3.5 CPU reference.

Runs the HF model in fp32 on fixed prompts (teacher forcing) and writes, per
prompt, to --out:
  <name>.ids.json      token ids
  <name>.logits.f32    [tokens, vocab] float32
  <name>.hidden.f32    [layers + 1, tokens, hidden] float32 (embeddings first)
plus manifest.json with shapes, model revision and library versions.

  python -I qwen35_oracle.py --model /models/Qwen3.5-2B-hf --out /models/oracle-2b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "chat": [{"role": "user", "content": "Write one sentence about the moon."}],
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "prose": (
        "The river ran past old stone bridges and quiet fields of wheat. "
        "In the evening the light turned gold, and the farmers walked home "
        "along the water, talking about the harvest and the coming rain."
    ),
}


def encode(tokenizer, prompt) -> list[int]:
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    manifest = {
        "model": str(args.model),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "model_class": type(model).__name__,
        "prompts": {},
    }
    for name, prompt in PROMPTS.items():
        ids = encode(tokenizer, prompt)
        out = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        logits = out.logits[0].float().numpy()
        hidden = torch.stack([h[0] for h in out.hidden_states]).float().numpy()
        (args.out / f"{name}.ids.json").write_text(json.dumps(ids))
        logits.astype(np.float32).tofile(args.out / f"{name}.logits.f32")
        hidden.astype(np.float32).tofile(args.out / f"{name}.hidden.f32")
        top = np.argsort(-logits[-1])[:5].tolist()
        manifest["prompts"][name] = {
            "tokens": len(ids),
            "vocab": int(logits.shape[1]),
            "hidden_layers": int(hidden.shape[0]),
            "hidden_size": int(hidden.shape[2]),
            "last_top5": top,
            "last_top5_text": [tokenizer.decode([t]) for t in top],
        }
        print(f"{name}: {len(ids)} tokens, last-position top-5 {top}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
