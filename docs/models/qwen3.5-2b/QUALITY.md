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

| Sequence | Relative error |
|---|---|
| q4nx_gemv_k2048_n2048 | 5.9e-6 |
| q4nx_gemv_k2048_n5120 | 5.4e-6 |
| q4nx_gemv_k2048_n8192 | 6.0e-6 |
| q4nx_gemv_k2048_n12288 | 5.4e-6 |
| q4nx_gemv_k6144_n2048 | 6.3e-6 |
| q4nx_gemv_k2048_n248832 | 4.7e-6 |

The kernel carries its dot products and token sums in two bf16 halves and
writes fp32; with single bf16 (IRON's q4nx_accumulate) the same test gave
4–8e-3. `q4nx_test` (hosted CI) checks the packing: Q4_K codes carry over
unchanged, requantized weights stay within half a 4-bit step, fused stacks
equal their parts, and the CPU q4nx GEMV matches the reference.

## NPU decode, teacher-forced

`qwen35_npu_decode_test` (labels `npu`, `model`; needs
`GUFO_NPU_QWEN35_2B_Q4_GGUF` and `GUFO_NPU_QWEN35_2B_ORACLE`): the oracle
prompts through Qwen3.5-2B UD-Q4_K_XL with every projection on the NPU,
against the same q4nx weights on the CPU. Pass: top-1 agreement >= 95% over
all prompts, logits relative error < 1e-2 per prompt. 2026-10-27:

| Prompt | Tokens | NPU vs q4nx CPU: argmax | Logits rel err | Argmax vs HF: NPU | q4nx CPU | GGUF CPU |
|---|---|---|---|---|---|---|
| chat | 19 | 100% | 5.4e-3 | 84.2% | 84.2% | 84.2% |
| code | 15 | 100% | 4.3e-3 | 100% | 100% | 100% |
| prose | 39 | 100% | 3.0e-3 | 84.6% | 84.6% | 89.7% |

- The NPU adds no top-1 changes over its own weights (73/73). The logits gap
  is mostly the bf16 rounding of each GEMV's input.
- Requantizing the 5–8-bit tensors to q4nx costs 2 of 73 top-1 matches
  against HF here (GGUF CPU vs q4nx CPU), on short prompts. A longer quality
  run decides whether an 8-bit kernel variant is needed.

## Fast decoder vs reference

`qwen35_decoder_test` (label `model`; needs `GUFO_NPU_QWEN35_2B_Q4_GGUF` and
`GUFO_NPU_QWEN35_2B_ORACLE`, no NPU): the fp32 decode step against the
reference, token by token on the chat prompt, both with the same q4nx
projections on the CPU. Pass: logits relative error < 1e-4 and every argmax
equal. 2026-10-27: worst 2.5e-6, argmax 19/19.

## Planned

| Check | Tolerance |
|---|---|
| Each NPU op vs CPU reference | relative error < 1e-2 |
| NPU model vs CPU reference, teacher-forced, same weights | greedy argmax match |
| Prefill vs one-token decode on the NPU | identical next token |
| MTP head vs a reference | to be decided (no HF oracle) |
