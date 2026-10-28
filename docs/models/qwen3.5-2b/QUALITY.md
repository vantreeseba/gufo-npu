# Qwen3.5-2B quality

## CPU reference vs HF transformers

`tests/models/qwen35_2b/reference_oracle_test.cpp` against
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
- `tests/models/qwen35_2b/generate_oracle_test.cpp`: the rendered chat prompt
  (thinking off) tokenizes to HF's ids, and greedy decoding on the BF16 GGUF
  reproduces HF `generate` for all 25 tokens up to `<|im_end|>`.

## NPU op tests

`q4nx_gemv_test` (CTest label `npu`): every GEMV variant with random weights
and inputs in the input/output stages it is built for, against a
double-precision reference over the same packed weights and the same bf16
operand; then all of them again chained in one runlist. Pass: relative L2
error < 1e-2, and the resnorm residual (H + D) within 1e-6 of its scale.
2026-10-27, on the NPU, one token (same results alone and chained), with
5120-byte blocks (bf16 scale and minimum per 32) and with the compact
4736-byte blocks (uint8/int8 per 32, bf16 d and dmin per row):

| Sequence | 5120-byte blocks | 4736-byte blocks | Residual |
|---|---|---|---|
| q4nx_gemv_k2048_n2048_plain_store | 5.9e-6 | 9.6e-6 | |
| q4nx_gemv_k4096_n2048_plain_store | 5.7e-6 | 1.1e-5 | |
| q4nx_gemv_k2048_n5120_resnorm_store | 5.4e-6 | 9.3e-6 | ✓ |
| q4nx_gemv_k2048_n8192_resnorm_store | 5.1e-6 | 7.9e-6 | ✓ |
| q4nx_gemv_k2048_n12288_resnorm_swiglu | 1.7e-3 | 1.6e-3 | ✓ |
| q4nx_gemv_k6144_n2048_act_store | 6.6e-6 | 9.7e-6 | |
| q4nx_gemv_k2048_n248832_resnorm_store | 5.4e-6 | 9.3e-6 | ✓ |

- The compact blocks add one more hi + lo rounding (the per-256 totals
  before d and dmin multiply them): ~1.7x the error, still ~1e-5.
- SwiGLU writes bf16 and the test compares against unrounded values, so its
  error is the bf16 rounding (~2^-9 / sqrt(3)).
- The 2-token n8192 case reads 6.1e-5 in both formats (a few outputs near
  zero in that random draw).

- The GEMV carries its dot products and token sums in two bf16 halves and
  writes fp32; with single bf16 (IRON's q4nx_accumulate) the same test gave
  4–8e-3.
- SwiGLU computes sigmoid from MLIR-AIE's exp2 polynomial and a Newton
  reciprocal. The `vtanh` instruction (aie::tanh) passes |x| <= 0.5 through
  unchanged and gave 1.1e-2.
- `q4nx_test` (hosted CI) checks the packing: Q4_K codes, sub-scales and
  sub-minima carry over unchanged (only d and dmin round to bf16),
  requantized weights stay within half their step with `Fit::kMinMax`, this
  model's fit (the group's range / 15, widened by the int8 minimum's grid
  and rounded up onto d's) and within a whole step with `Fit::kSearch`
  (which may clip a group's extremes), fused stacks equal their parts, row
  orders (SwiGLU) apply, and the CPU q4nx GEMV matches the reference.
  Requantizing N(0, 0.02) weights: relative RMS error 0.0787 with min/max
  (0.0783 with 5120-byte blocks), 0.0702 with the search fit.

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

With the compact 4736-byte blocks (same day, same test):

| Prompt | Tokens | NPU vs q4nx CPU: argmax | Logits rel err | NPU argmax vs HF |
|---|---|---|---|---|
| chat | 19 | 19/19 | 7.1e-3 | 17/19 |
| code | 15 | 15/15 | 5.0e-3 | 15/15 |
| prose | 39 | 38/39 | 4.5e-3 | 31/39 |

- The NPU adds no top-1 changes over its own weights with 5120-byte blocks
  (73/73) and one with 4736-byte blocks (72/73, a near tie in prose). The
  logits gap is mostly the bf16 rounding of each GEMV's input (and the bf16
  norm weights). Against HF the two formats score 64/73 and 63/73: d and
  dmin now round to bf16 instead of each d * sc, which moves a few near
  ties either way.
- Against HF, q4nx on the CPU and the GGUF reference agree on the same
  positions as the NPU here, except 2 prose positions where the GGUF weights
  do better: requantizing the 5–8-bit tensors costs 2 of 73. A longer quality
  run decides whether an 8-bit kernel variant is needed.

### The search fit, measured and not adopted

Each model names its q4nx fit (`q4nx::Fit`; this package's rule is
`qwen35::Q4nxFit`). Qwen3.6-35B-A3B and Qwen3.5-4B pack with the search
fit (llama.cpp's weighted least-squares Q4_K fit,
[measured on the 35B](../qwen3.6-35b-a3b/QUALITY.md#choosing-the-4-bit-values));
this model keeps min/max, because on its decode weights the search fit
lowers KL but not top-1. With the search fit (2026-10-27, compact
blocks):

| Prompt | Tokens | NPU vs q4nx CPU: argmax | Logits rel err | NPU argmax vs HF |
|---|---|---|---|---|
| chat | 19 | 19/19 | 8.2e-3 | 17/19 |
| code | 15 | 15/15 | 5.1e-3 | 14/15 |
| prose | 39 | 39/39 | 4.9e-3 | 30/39 |

- 73/73 against its own weights on the CPU; 61/73 against HF (63/73 with
  the min/max fit), a difference of two near ties in 73 positions. On
  1,515 positions of held-out text against the GGUF as stored (the 35B's
  quality test's harness run on this model, same texts): with every matrix
  in q4nx KL falls from 0.0424 to 0.0332 but top-1 agreement with the GGUF
  goes from 91.29% to 90.89% (6 positions); without the LM head (the tied
  Q6_K embedding) KL falls from 0.0336 to 0.0258 and top-1 rises from
  93.14% to 93.47%.
- The NPU decoder would pack and upload in 2.6 s instead of 1.3 s.
- With min/max kept, the 2B's outputs are main's: the same hashes and
  speed (see EXPERIMENTS.md).

## Speculative decoding

`qwen35_npu_decode_test` also decodes 48 tokens from the chat prompt with
and without MTP speculation and requires identical tokens (2026-10-27:
identical; 29 steps, 18 drafts accepted with the 32k draft vocabulary, 28
and 19 with the full one). A rejected draft rolls back the DeltaNet state
(the draft's update goes to a spare buffer), the conv window and the
attention caches. The draft vocabulary only changes which drafts are
proposed; acceptance per prompt and vocabulary size is in BENCHMARKS.md.

## Determinism

Greedy output on the NPU must be the same on every run.
`q4nx_gemv_chain_test` (label `npu`) runs one decoder layer's chain of
GEMVs 60,000 times with the same inputs, alternating 1 and 2 tokens, and
requires every output to be bit-identical.

2026-10-27: about one sweep run in ten (long prompts: one call in ~30,000)
gave a different reply. The chain test (200,000 runs) found 6 runs whose
gate + up output differed, always in column 1 (the residual column), round 3
or 4. The cause was the wait order in the GEMV's runtime sequence: the
residual drain, queued first on that column's output channel, was awaited
last, so its completion token satisfied the wait for round 0's output, and
the SwiGLU drain window reused round 0's BD while it could still be in
flight. Awaiting the residual first fixed it: 0 differing runs in 200,000.
CPU load, DDR load, memory compaction and a second process on the NPU did
not change the output (175 runs).

## NPU prefill (Phase 4)

`bfp16_gemm_test` (label `npu`): the prefill GEMM at every projection shape
of the 2B and every chunk size (256, 512, 1024 prompt tokens per call), with
random q4nx weights and bf16 activations, against a double-precision product
on the CPU. Pass: relative L2 error < 1e-2 against the q4nx weights (the
values the decode GEMV multiplies). 2026-10-27, every shape and chunk size:

| Against | Relative error |
|---|---|
| The bfp16 weights the NPU holds (kernel error) | 6.9e-3 |
| The q4nx weights (adds their rounding to bfp16) | 9.2e-3 |

- The kernel error is the mmul's conversion of A (the activations) to
  bfp16, 8 values per shared exponent, plus the bf16 output. The weights'
  own bfp16 rounding adds the rest. Both are inside the 1e-2 budget, but
  with little margin; an exact bf16 A would need a second (lo) mmul pass.
- `bfp16_test` (hosted CI) checks the packer bit for bit against IRON's
  `pack_b` order and rounding (ties to even, saturation at +127).
- Determinism: a DMA wait takes its shim channel's next completion token,
  whichever task issued it (the q4nx GEMV's rare nondeterminism). Every
  compiled GEMM sequence has exactly one token-issuing task per channel (a
  column's last C drain), awaited once, so it cannot take another task's
  token. A stress run of every (M, K, N), alone and with two calls in
  flight on separate buffers as prefill's lanes run them, compared 634,544
  outputs bit for bit with the first: none differed (2026-10-27).

`qwen35_npu_prefill_test` (labels `npu`, `model`; needs
`GUFO_NPU_QWEN35_2B_Q4_GGUF` and `GUFO_NPU_QWEN35_2B_ORACLE`): the oracle
prompts and prompts of 1, 2, 256, 257, 600, 1,100 and 2,049 tokens (one and
two 256-row chunks, two lanes of 256, of 512 and of 1,024), all but the last
token through NpuPrefill; the reference is the same tokens through NpuDecoder
two per pass (same q4nx weights). It compares the final residual of every
prompt position; the state the decoder continues from, layer by layer (each
DeltaNet layer's conv window and recurrent state, each attention layer's
fp16 K/V cache); the MTP layer's K/V cache from NpuPrefill::FillMtp against
NpuDecoder::FillMtp on the same hidden rows; then the decoder runs the last
token and 16 greedy steps from both states, teacher-forced on the decode
path's tokens. Pass: hidden < 6e-2, every state and the MTP cache < 8e-2,
logits < 5e-2 (relative L2), the first token equal, argmax agreement >=
95%. 2026-10-27:

| Prompt | Tokens | Hidden | Conv | DeltaNet state | K/V | MTP K/V | Logits | Argmax |
|---|---|---|---|---|---|---|---|---|
| chat | 19 | 3.6e-2 | 3.4e-2 | 3.6e-2 | 2.5e-2 | 1.1e-2 | 1.0e-2 | 17/17 |
| code | 15 | 2.4e-2 | 2.0e-2 | 2.3e-2 | 2.0e-2 | 1.1e-2 | 1.3e-2 | 17/17 |
| prose | 39 | 2.0e-2 | 2.0e-2 | 2.1e-2 | 1.9e-2 | 1.2e-2 | 1.3e-2 | 17/17 |
| len1 | 1 | – | – | – | – | – | 0 | 17/17 |
| len2 | 2 | 3.6e-2 | 2.5e-2 | 4.0e-2 | 2.0e-2 | 3.3e-3 | 6.9e-3 | 17/17 |
| len256 | 256 | 3.4e-2 | 3.2e-2 | 3.7e-2 | 3.1e-2 | 1.2e-2 | 1.4e-2 | 17/17 |
| len257 | 257 | 3.4e-2 | 3.0e-2 | 3.7e-2 | 3.1e-2 | 1.2e-2 | 1.4e-2 | 17/17 |
| len600 | 600 | 3.8e-2 | 4.4e-2 | 5.5e-2 | 3.8e-2 | 1.2e-2 | 2.6e-2 | 17/17 |
| len1100 | 1,100 | 3.9e-2 | 3.4e-2 | 5.6e-2 | 3.9e-2 | 1.2e-2 | 3.3e-2 | 17/17 |
| len2049 | 2,049 | 4.0e-2 | 3.8e-2 | 5.8e-2 | 3.9e-2 | 1.2e-2 | 3.2e-2 | 17/17 |

- With the fp16 K/V cache and compact q4nx scales (decode branch e3605ce);
  prefill writes the cache in fp16 as the decoder does.
- 170/170 argmax; the first token after the prompt is the same on every
  prompt. The DeltaNet states carry the largest error (they integrate
  every token's); a wrong offset or a stale window would be of order 1.
- MTP speculative decoding after prefill gives the plain greedy output
  after prefill token for token (48 tokens). Greedy decoding after prefill
  and after the token-by-token prompt agree on the first 26 of 48 tokens
  of the chat prompt (48/48 with the earlier fp32 cache and 5-bit scales),
  the same in four runs: a near-tie at token 27, where 1–3% logit
  differences pick the other token, not nondeterminism.
- The gap to the decode path is mostly the bfp16 activations: each of the
  96 GEMMs of a pass (4 per layer) adds ~7e-3 where the decode GEMV only
  rounds its input to bf16, and the GEMMs write bf16 where the GEMV writes
  fp32; the residual stream accumulates it. The DeltaNet and attention
  mixers run in fp32 on the CPU with the decoder's arithmetic.

### First token from the prefill (2026-10-28)

PrefillPrompt now runs the whole prompt through NpuPrefill, and the first
token comes from its last row through the LM head alone
(`NpuDecoder::Logits`), with no decode step for the last prompt token. The
same test now prefills every prompt token (states compared after the whole
prompt), takes the first token's logits from `Logits` on prefill's last
row, and checks it against the decode path's step of the last token: the
same token, or a near tie in the decode path's own top two. On the ten
prompts above:

| Check | Measured |
|---|---|
| First token the same as the decode path's | 10/10 |
| Largest change of any logit, prefill row vs decode step | 0.89 (len600) |
| Smallest decode-path top-2 gap among the prompts | 0.26 (prose; still the same token) |
| States after the whole prompt (worst) | hidden 4.0e-2, conv 3.9e-2, DeltaNet 5.8e-2, K/V 3.9e-2 (under the same limits) |
| Argmax over the first token and 16 teacher-forced steps | 170/170 |
| Speculative decoding after prefill vs plain greedy (chat, 48 tokens) | identical |

The near-tie limit is 2 logits. A token changes only if its logit and the
runner-up's move across each other, which takes a gap smaller than about
twice the largest logit change (0.89 here, 0.60 on the 4B), so 1.8. A
first token may differ only below 2 logits; none did.

The outputs on the decode benchmarks change only where a near tie falls
the other way. The sky prompt without drafting and the 960-token prompt
with MTP keep their hashes (`a49f1c4c`, `8edbce90`). The Fibonacci prompt
with MTP changes (`69e1c62d` → `fdb595fb`, 239 → 256 generated tokens):
its MTP output still equals plain greedy from the same prefilled state
([BENCHMARKS.md](BENCHMARKS.md), "TTFT quick wins").

## Fast decoder vs reference

`qwen35_decoder_test` (label `model`; needs `GUFO_NPU_QWEN35_2B_Q4_GGUF` and
`GUFO_NPU_QWEN35_2B_ORACLE`, no NPU): the decode step against the
reference, token by token on the chat prompt, both with the same q4nx
projections on the CPU. The decoder keeps the attention K/V cache in fp16,
the reference in fp32. Pass: logits relative error < 2e-3 and every argmax
equal. 2026-10-27: worst 6.2e-4 with the fp16 cache (9.3e-7 with an fp32
one), argmax 19/19.

## Planned

| Check | Tolerance |
|---|---|
| Each NPU op vs CPU reference | relative error < 1e-2 |
| NPU model vs CPU reference, teacher-forced, same weights | greedy argmax match |
| Prefill vs one-token decode on the NPU | identical next token: met (`qwen35_npu_prefill_test`, first token equal on all 10 prompts of 1–2,049 tokens, 170/170 argmax over 17 steps each; greedy output equal for the first 26 of 48 tokens on the chat prompt) |
| MTP head vs a reference | to be decided (no HF oracle) |

## Prompt cache (2026-10-28)

`qwen35_prompt_cache_test` (labels `qwen35_2b;npu;model`; how the cache works is in
[BENCHMARKS.md](BENCHMARKS.md), "Prompt cache across requests"). A chat
over ~1k tokens of notes, turns 2 and 3, each state against the whole
prompt through the model's own `PrefillPrompt`: the decode state (every
layer's conv window, DeltaNet state and K/V cache), the MTP layer's K/V
cache and the last prompt row's final residual.

| Check | Turn 2 / turn 3 |
|---|---|
| The previous turn's checkpoint restored, the rest (~40 tokens) through the prefill | **bit-identical** (every error 0) |
| An exact repeat: the checkpoint restored, the generation prompt through the prefill | **bit-identical** |
| A 5-token remainder through the decoder (below the prefill threshold) | hidden 9.6e-2 / 1.0e-1, states 4.7e-2 / 4.9e-2 |
| The same prompt entirely through the decode path, for scale | hidden 1.1e-1 / 1.1e-1, states 5.4e-2 / 5.6e-2 |
| The checkpoint the decoder took mid-remainder, continued through the prefill | hidden 3.2e-2 / 3.1e-2, states 3.9e-2 / 3.8e-2 |
| Greedy MTP output, 3 turns at ~1k and ~4k tokens of history, two rounds: with the cache (hits) and on a miss, against `PrefillPrompt` | **identical** every turn |

- **Bit-identical restore:** the prefill's arithmetic for a token does not
  depend on where a pass starts. Its GEMM rows are independent, the
  DeltaNet runs token by token, and attention reads the same cache, so a
  checkpoint plus the rest equals one prefill exactly.
- **The decoder's remainders** differ from the prefill as the decode path
  always does (bfp16 GEMMs against q4nx GEMVs), and no more: within the
  limits of the model's prefill test (hidden 6e-2 (or 1.5× the decode path's), states 8e-2).
- **Untouched without a cache:** the capture is off unless a checkpoint
  is asked for, and outputs keep their hashes: `a49f1c4c` (sky, no draft), `fdb595fb` (Fibonacci, MTP) and `8edbce90` (960 tokens, MTP), unchanged.
- **Cache logic** (`qwen35_prompt_cache_logic_test`, CPU, label
  `qwen35_family`): on a toy model with an order-dependent recurrent
  layer, a restored run leaves exactly the uncached state over four
  turns, with and without the MTP cache and with the last token left to
  the decoder; short prompts go to the decoder untouched; a repeat runs
  only the generation prompt; least-recently-used and byte-budget
  eviction; and a draft request skips checkpoints without the MTP cache.
