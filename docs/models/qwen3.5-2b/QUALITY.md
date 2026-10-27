# Qwen3.5-2B quality

## CPU reference vs HF transformers

`tests/models/qwen35/reference_oracle_test.cpp` against
`tools/reference/qwen35_oracle.py` (transformers 5.21.0, torch 2.13.0+cpu,
fp32, `Qwen/Qwen3.5-2B@15852e8c`). Same token ids on both sides; three
prompts (chat template, code, prose; 73 tokens). Measured 2026-10-27.

| Weights | Worst layer hidden rel. L2 | Logits rel. L2 | Greedy argmax match | Tolerance |
|---|---|---|---|---|
| BF16 GGUF (same weights as HF) | ≤ 9.9e-07 | ≤ 1.1e-06 | 73/73 | 1e-5 (enforced) |
| UD-Q4_K_XL GGUF | 7.8e-02 – 1.4e-01 | 8.2e-02 – 1.6e-01 | 66/73 (90%) | informational |

- One-token-at-a-time decode reproduces the whole-prompt forward exactly
  (rel. err 0).
- The Q4 row is quantization loss, not a reference error. The NPU path is
  checked against this CPU reference **on the same Q4 weights**.

## Planned

| Check | Tolerance |
|---|---|
| Each NPU op vs CPU reference | relative error < 1e-2 |
| NPU model vs CPU reference, teacher-forced, same weights | greedy argmax match |
| Prefill vs one-token decode on the NPU | identical next token |
| MTP head vs a reference | to be decided (no HF oracle) |
