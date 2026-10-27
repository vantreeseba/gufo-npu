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

## Tokenizer and chat template vs HF

- `tests/core/tokenizer_test.cpp`: 18/18 fixture strings encode to the same
  ids as HF `tokenizers` and decode to the same text (CJK, emoji/ZWJ,
  decomposed Unicode, contractions, code, special tokens inside text).
  Fixtures: `tools/reference/tokenizer_fixtures.py`.
- `tests/models/qwen35/generate_oracle_test.cpp`: the rendered chat prompt
  (thinking off) tokenizes to HF's ids, and greedy decoding on the BF16 GGUF
  reproduces HF `generate` for all 25 tokens up to `<|im_end|>`.

## NPU op tests

`q4nx_gemv_test` (CTest label `npu`): random weights packed to q4nx and a
random bf16 token, NPU output against a double-precision GEMV over the same
packed weights. Pass: relative L2 error < 1e-2. 2026-10-27, on the NPU:

| Design | Relative error |
|---|---|
| q4nx_gemv_k2048_n2048 | 7.8e-3 |
| q4nx_gemv_k2048_n6144 | 4.7e-3 |
| q4nx_gemv_k6144_n2048 | 4.5e-3 |
| q4nx_gemv_k2048_n248832_norm (RMS norm fused) | 6.1e-3 |

The error comes from the kernel's bf16 narrowings: each 32-wide dot product of
codes and token, the per-32 token sums and the output. Real tensors give
0.4–0.6% (BENCHMARKS.md). `q4nx_test` (hosted CI) checks the packing: Q4_K
codes carry over unchanged, and requantized weights stay within half a 4-bit
step.

## Planned

| Check | Tolerance |
|---|---|
| Each NPU op vs CPU reference | relative error < 1e-2 |
| NPU model vs CPU reference, teacher-forced, same weights | greedy argmax match |
| Prefill vs one-token decode on the NPU | identical next token |
| MTP head vs a reference | to be decided (no HF oracle) |
