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

`q4nx_gemv_test` (CTest label `npu`): every GEMV variant with random weights
and inputs in the input/output stages it is built for, against a
double-precision reference over the same packed weights and the same bf16
operand; then all of them again chained in one runlist. Pass: relative L2
error < 1e-2, and the resnorm residual (H + D) within 1e-6 of its scale.
2026-10-27, on the NPU (same results alone and chained):

| Sequence | Relative error | Residual |
|---|---|---|
| q4nx_gemv_k2048_n2048_plain_store | 5.9e-6 | |
| q4nx_gemv_k2048_n5120_resnorm_store | 5.4e-6 | ✓ |
| q4nx_gemv_k2048_n8192_resnorm_store | 6.4e-6 | ✓ |
| q4nx_gemv_k2048_n12288_resnorm_swiglu | 1.1e-5 | ✓ |
| q4nx_gemv_k6144_n2048_act_store | 5.9e-6 | |
| q4nx_gemv_k2048_n248832_resnorm_store | 5.6e-6 | ✓ |

- The GEMV carries its dot products and token sums in two bf16 halves and
  writes fp32; with single bf16 (IRON's q4nx_accumulate) the same test gave
  4–8e-3.
- SwiGLU computes sigmoid from MLIR-AIE's exp2 polynomial and a Newton
  reciprocal. The `vtanh` instruction (aie::tanh) passes |x| <= 0.5 through
  unchanged and gave 1.1e-2.
- `q4nx_test` (hosted CI) checks the packing: Q4_K codes carry over
  unchanged, requantized weights stay within half a 4-bit step, fused stacks
  equal their parts, row orders (SwiGLU) apply, and the CPU q4nx GEMV matches
  the reference.

## NPU decode, teacher-forced

`qwen35_npu_decode_test` (labels `npu`, `model`; needs
`GUFO_NPU_QWEN35_2B_Q4_GGUF` and `GUFO_NPU_QWEN35_2B_ORACLE`): the oracle
prompts token by token through NpuDecoder (chained GEMVs with norms, residual
and SwiGLU on the NPU), against the CPU decoder over the same q4nx weights.
Pass: top-1 agreement >= 95% over all prompts, logits relative error < 1e-2
per prompt. 2026-10-27:

| Prompt | Tokens | NPU vs q4nx CPU: argmax | Logits rel err | NPU argmax vs HF |
|---|---|---|---|---|
| chat | 19 | 19/19 | 7.2e-3 | 16/19 |
| code | 15 | 15/15 | 5.4e-3 | 15/15 |
| prose | 39 | 39/39 | 4.3e-3 | 33/39 |

- The NPU adds no top-1 changes over its own weights (73/73). The logits gap
  is mostly the bf16 rounding of each GEMV's input (and the bf16 norm
  weights).
- Against HF, q4nx on the CPU and the GGUF reference agree on the same
  positions as the NPU here, except 2 prose positions where the GGUF weights
  do better: requantizing the 5–8-bit tensors costs 2 of 73. A longer quality
  run decides whether an 8-bit kernel variant is needed.

## Speculative decoding

`qwen35_npu_decode_test` also decodes 48 tokens from the chat prompt with
and without MTP speculation and requires identical tokens (2026-10-27:
identical; 28 steps, 19 drafts accepted). A rejected draft rolls back the
DeltaNet state (the draft's update goes to a spare buffer), the conv window
and the attention caches.

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
