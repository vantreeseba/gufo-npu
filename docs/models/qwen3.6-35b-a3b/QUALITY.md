# Qwen3.6-35B-A3B quality

Planned contracts (see [the plan](../../PLAN.md)):

| Check | Tolerance | State |
|---|---|---|
| CPU reference vs llama.cpp (CPU), teacher-forced logits | see below | enforced |
| Each NPU op vs CPU reference | relative error < 1e-2 | enforced: MoE gathers 4.8e-4 (1 token), 4.8e-4 to 7.9e-4 (2 tokens, unions of 9 to 17) |
| Prefill ops vs CPU over the same q4nx weights | GEMMs relative error < 1e-2; dequant weights within 5% of the host's bfp16 packing | enforced: GEMMs 9.1–9.4e-3, dequant 6.170e-3 against 6.163e-3 (below) |
| NPU decode vs CPU decoder over the same q4nx weights, teacher-forced, routing followed | logits rel. L2 < 1e-2 per prompt, argmax ≥ 98%, routing differences at near ties only | enforced (below) |
| NPU MTP speculation vs plain greedy decode | identical output | enforced: 400 tokens after 4 prompts, 73% of drafts accepted |
| NPU prefill vs NPU decode, routing followed | hidden < 0.15, states < 0.2, logits < 0.3, argmax ≥ 95%, first token identical unless the decode path's top two are within 0.1 | enforced: hidden ≤ 8.2e-2, states ≤ 0.111, logits ≤ 0.21, argmax 170/170 (below) |
| CPU decoder vs reference | see below | enforced |
| MTP head (reference) vs main model's top-1 | ≥ 50% of drafts | enforced (69%) |
| MTP head vs reference | cosine > 0.999 | |

There is no HF transformers oracle for this model here (the BF16 weights are
70 GB); llama.cpp on the same GGUF is the independent implementation.

## CPU reference vs llama.cpp

`tests/models/qwen36_35b_a3b/reference_llama_test.cpp` (label `model`;
`GUFO_NPU_QWEN36_MODEL` is the UD-Q4_K_XL GGUF, `GUFO_NPU_QWEN36_ORACLE` the
output of `tools/reference/qwen36_llama_oracle.py`). llama.cpp
0c1e57098bba43ac29e6e3b677cdceebdd22334f via llama-cpp-python 0.3.36, built
CPU-only (AVX2/FMA/F16C, [TOOLCHAIN.md](../../TOOLCHAIN.md)); whole prompt in
one batch, fp32 KV cache, no flash attention, no weight repacking. Same GGUF,
same token ids, four prompts (chat template, code, prose, and a 230-token
prose passage; 301 tokens). Measured 2026-10-27.

| Prompt | Tokens | End to end: logits rel. L2 | Argmax | llama.cpp vs itself¹: rel. L2 | Argmax | Layer by layer²: contributions < 1e-5 | Worst | LM head³ rel. L2 |
|---|---|---|---|---|---|---|---|---|
| chat | 17 | 7.9e-02 | 16/17 | 7.3e-02 | 16/17 | 572/680 (84.1%) | 1.4e-02 | 1.5e-07 |
| code | 15 | 3.5e-02 | 15/15 | 3.2e-02 | 15/15 | 499/600 (83.2%) | 1.2e-02 | 1.6e-07 |
| prose | 39 | 2.4e-02 | 39/39 | 2.6e-02 | 38/39 | 1269/1560 (81.3%) | 1.3e-02 | 1.4e-07 |
| long | 230 | 4.4e-02 | 224/230 | 4.3e-02 | 224/230 | 6735/9200 (73.2%) | 3.5e-02 | 7.5e-06 |

1. llama.cpp one token per `llama_decode` against llama.cpp on the whole
   prompt (recorded in the oracle's manifest).
2. Each decoder layer run alone on llama.cpp's own input to it (its
   `l_out-<il - 1>`), with llama.cpp's activation rounding applied (below),
   per token. The error is relative to the layer's own contribution
   (output − input), not to the whole residual. Median 3.6e-7.
3. The final norm and LM head on llama.cpp's last hidden state, same
   rounding; argmax 301/301.

Also: our tokenizer gives llama.cpp's ids for all four prompts, and
`RenderChat` gives the GGUF chat template's rendering (thinking on). One
token at a time through the reference reproduces the whole-prompt logits
exactly (rel. err 0).

**Why end to end is only ~3–8%.** llama.cpp's CPU matmul first rounds its
input to the weight's dot-product type: 8 bits per 32 values with an fp16
scale (Q8_0) for the Q8_0 tensors, 8 bits per 256 values (Q8_K) for the
K-quant experts. Hidden states carry outliers of 10–30× their RMS, so that
rounding is coarse (~1% per matmul), and it compounds over 40 layers. The
reference keeps fp32 activations. llama.cpp disagrees with itself by as much
as with the reference when only its kernels and accumulation order change
(one token per call), so the end-to-end number measures llama.cpp's noise,
not the reference's error.

**Layer by layer is exact.** The test can apply llama.cpp's rounding to the
reference's matrix inputs (`LlamaCpuRounding`: Q8_0 with an fp16 scale and
round-to-nearest-even like `quantize_row_q8_0` on x86, Q8_K like
`quantize_row_q8_K_ref`, none for the F32 router and DeltaNet alpha/beta).
Fed llama.cpp's own layer inputs, most layer contributions then agree to
float precision. The rest follow a single 8-bit rounding that came out
differently because the input differed in its last bits (accumulation
order): one rounding step on one element of a block with an outlier moves
the output by ~1e-3 of the residual, and the DeltaNet state and the KV cache
carry it to the later tokens of that layer, so one layer of one prompt can
have most of its tokens off (chat layer 0: 1 of 17 tokens exact). A
layer-0 trace (prose) showed exactly this: DeltaNet outputs equal to
1e-7 until a Q8_0 flip in `attn_qkv`'s input at token 31, and expert outputs
equal to 1e-7 except on tokens whose FFN input had a flipped `ssm_out`
rounding upstream. Same experts chosen for every token, router weights equal
to 1e-4.

Thresholds; the layer checks hold for every layer on its own:

| Check | Pass | Measured | Why |
|---|---|---|---|
| End to end logits rel. L2, per prompt | < 0.15 | ≤ 7.9e-2 | 2× llama.cpp's spread against itself (7.3e-2) |
| End to end argmax, all prompts | ≥ 95% | 294/301 (97.7%) | llama.cpp against itself: 293/301 |
| A layer's tokens within 1e-5, all prompts | ≥ 10% | ≥ 26.6% (layer 0) | a math error in a layer moves all its tokens |
| A layer's median, per prompt | < 3e-2 | ≤ 9.2e-3 | |
| A layer's worst token, per prompt | < 0.1 | ≤ 3.5e-2 | a flip moves ~1e-3 of the residual |
| LM head logits rel. L2, per prompt | < 1e-4 | ≤ 7.5e-6 | same rounding, only accumulation order left |
| LM head argmax, all prompts | 301/301 | 301/301 | |
| Token by token vs whole prompt | < 1e-6 | 0 | |

Fault injection: scaling layer 34's routed-expert outputs by 1.001 (a 0.1%
error confined to one of the Q6_K layers) leaves that layer 0% of tokens
within 1e-5 and fails the test. Pooled over all 40 layers, as this test
first did, the same fault still left 71–82% of each prompt's layer outputs
exact, above the 70% that check required, and passed.

The first end-to-end top-1 difference (chat, position 12, after
`<|im_start|>`) is a near tie in llama.cpp (gap 0.70 logits) that the
reference resolves to the token that actually follows (`assistant`, gap
1.51). llama.cpp run one token at a time picks `assistant` there too (its
other change on the short prompts is prose position 11, gap 0.10). The
long prompt's six differences are near ties on one side or the other (gaps
0.02–0.62 logits).

### Reference speed and memory

Release build, Ryzen AI MAX+ 395 (matrix products on 16 threads,
`GgufProjections` in double), the GGUF in page cache, same run: whole-prompt
forward 5.5–6.2 tok/s (17–230 tokens: each expert's rows are dequantized
once per prompt), one token at a time 5.0 tok/s. Peak RSS 16.7 GB, all but
0.33 GB of it file-backed pages of the memory-mapped GGUF (only the experts
the prompts touch are read); nothing is dequantized ahead of use. The full
test takes ~110 s.

## Decoder vs reference

`tests/models/qwen36_35b_a3b/decoder_test.cpp` (label `model`, same variables): the
CPU decode path, `Decoder` (fp32 with an fp16 K/V cache, the MoE routing of
`moe.hpp`) over `GgufProjections` in fp32, against the reference (the same
GGUF in double), token by token over the chat prompt. 2026-10-27:

| Check | Pass | Measured |
|---|---|---|
| Logits rel. L2 up to the first token routed differently | < 2e-3 | 3.3e-4 (2B decoder on the same projections: 4.3e-4) |
| Reference's 8th/9th expert gap where routing first differs | < 1e-3 | 1.3e-4 (position 13, layer 36) |
| Argmax | 17/17 | 17/17 |

The same decoder over the reference's own double projections gave 3.6e-4
and the same routing differences, so the fp32 dot products add little.

Top-k routing is discontinuous. In 17 tokens × 40 layers the reference saw
7 routings whose 8th and 9th expert probabilities were within 0.1% of each
other; at one of them (position 13, layer 36) the decoder's fp32 and fp16
rounding picked the other expert. That changes the layer output by a few
percent, which the later layers (layer 39 then routes differently too, gap
2.2e-2) and the DeltaNet state carry on: logits rel. err 4.9e-2 at position
13 and 1e-3–3e-2 after, top-1 unchanged. Any two implementations will
disagree like this somewhere (llama.cpp included), so MoE decode checks
compare closely only up to the first routing difference and require it to
be at a near tie.

## MTP head

`tests/models/qwen36_35b_a3b/mtp_test.cpp` (label `model`, same variables).
Neither oracle runs this model's MTP layer (llama.cpp's MTP context API is
experimental), so the reference's head (`Reference::Mtp`, the `blk.40`
nextn layer with its MoE FFN) is checked against the main model: teacher
forced, the draft for position p + 1 (from the main model's last hidden
state at p - 1 and the token at p) against the main model's top-1 for
p + 1. 2026-10-27:

| Prompt | Drafts matching the main model's top-1 |
|---|---|
| chat | 10/15 |
| code | 12/13 |
| prose | 23/37 |
| all (pass ≥ 50%) | 45/65 (69%); 31/65 match the text itself |

A miswired head (swapped or unnormalized inputs, the wrong norm or head)
drafts next to nothing right. Drafting one token at a time gives the same
logits as all at once (rel. err 0). RoPE starting at the token's position
or at the hidden state's gives identical drafts (only position differences
matter inside the MTP block's own cache).

## q4nx requantization

What the NPU's q4nx format (`src/core/q4nx.hpp`: 4.625 bits per weight,
4-bit codes, 8-bit sub-scale and sub-minimum per 32, bf16 d and dmin per row
and 256 inputs) costs when every matrix is stored in it.
`tests/models/qwen36_35b_a3b/q4nx_requant_test.cpp` (label `model`, same variables)
runs the fp32 CPU decoder teacher forced over the four oracle prompts (chat,
code, prose and a 230-token prose passage; 301 positions) with four weight
sets:

- **(a)** as stored in the GGUF;
- **(d)** every non-expert matrix through q4nx (DeltaNet and attention
  projections, shared experts; all Q8_0 in the GGUF), LM head as stored;
- **(b)** (d) plus the LM head (Q8_0) through q4nx;
- **(c)** (b) plus the routed experts: Q4_K gate/up keep their codes (d and
  dmin round to bf16), Q5_K/Q6_K down projections (and layer 39's Q5_K
  gate/up) requantize.

Matrices through q4nx are packed with `q4nx::PackRows` and multiplied with
`q4nx::Gemv`, i.e. the weights `q4nx::Unpack` returns. Q4_K expert rows are
unpacked on the fly by the same rule, checked bit for bit against
`q4nx::Unpack(q4nx::PackRows(...))` first. q4nx works per row and per 256
inputs, so how the NPU stacks or orders rows does not change any weight. F32
matrices (router, DeltaNet alpha/beta) and the embedding lookup stay as
stored. KL is KL(softmax(a) ‖ softmax(set)) per position, in nats.
2026-10-27, with the min/max fit the packer used until the change in the
next section, deterministic across two runs:

| Set | Packed | Top-1 vs llama.cpp | Top-1 vs reference | Top-1 vs (a) | First top-1 change vs (a): chat / code / prose / long | KL vs (a): mean | max | Logits rel. L2 vs reference (per prompt) |
|---|---|---|---|---|---|---|---|---|
| (a) | — | 296/301 (98.3%) | 299/301 (99.3%) | 301/301 | — | 0 | 0 | 1.3e-4 – 1.5e-2 |
| (d) | 0.82 GB | 272/301 (90.4%) | 275/301 (91.4%) | 277/301 (92.0%) | none / 0 / 4 / 35 | 0.041 | 0.42 | 0.09 – 0.17 |
| (b) | 1.11 GB | 269/301 (89.4%) | 272/301 (90.4%) | 274/301 (91.0%) | none / 0 / 6 / 35 | 0.047 | 0.47 | 0.11 – 0.18 |
| (c) | 7.62 GB¹ | 260/301 (86.4%) | 266/301 (88.4%) | 265/301 (88.0%) | none / 0 / 4 / 32 | 0.048 | 0.40 | 0.12 – 0.18 |

1. Only the requantized experts are packed by the test (Q4_K experts are
   read from the GGUF); all of (c) in q4nx is ~20.5 GB.

For scale: llama.cpp against (a), which differs only by its 8-bit
activation rounding, has KL mean 0.006 (max 0.50) and 98.3% top-1. The
reference's 2B UD-Q4_K_XL GGUF against BF16 HF agrees on 90% of top-1
(66/73).

Greedy decoding of the chat prompt, 64 tokens: (c) matches (a) for the first
38 tokens, then words the same plan differently (38/64 equal).

- (a): `Thinking Process:\n\n1.  **Deconstruct the request:**\n    *   Topic:
  The moon.\n    *   Constraint: One sentence.\n\n2.  **Brainstorming
  angles/facts about the moon:**\n    *   It's Earth's only natural
  satellite.\n    *   It reflects`
- (c): `Thinking Process:\n\n1.  **Deconstruct the request:**\n    *   Topic:
  The moon.\n    *   Constraint: One sentence.\n\n2.  **Brainstorm
  ideas/facts about the moon:**\n    *   It's Earth's only natural
  satellite.\n    *   It affects t`

Reading: almost all of the loss comes from requantizing the 8-bit
non-expert matrices (d): 8% of top-1 choices change and KL rises ~7× over
llama.cpp's own noise. The LM head adds 1 point, the Q5_K/Q6_K experts 3
more. Keeping the non-expert matrices at 8 bits on the NPU (an 8-bit GEMV
variant; 1.9 B weights, ~2 GB at Q8_0 against 1.1 GB in q4nx) would recover
most of it; the experts, which dominate the bytes, can stay q4nx. Peak RSS
of the test 26.6 GB, of which 8.2 GB anonymous (the packed matrices) and the
rest page cache of the memory-mapped GGUF; ~7 minutes. Choosing the 4-bit
values better, in the same format, recovers part of it without the bytes
(next section).

## Choosing the 4-bit values

The q4nx format fixes the bytes and the kernel, but for every weight that
is not Q4_K the packer still chooses each row's d and dmin and each group's
sub-scale, sub-minimum and codes. `tests/models/qwen36_35b_a3b/q4nx_quality_test.cpp`
(label `model`; `GUFO_NPU_QWEN36_TEXT` is the output of
`tools/reference/fetch_text.py`) compares ways of choosing them, against the
GGUF as stored, with the weight sets above:

- **Evaluation text:** six pinned rows of public datasets (English, German
  and Chinese Wikipedia, an UltraChat reply, a CodeFeedback answer, a
  NuminaMath solution), the first 256 tokens of each: 1,515 positions,
  teacher forced through the fp32 CPU decoder. KL(a ‖ set) mean is the main
  number, top-1 agreement with (a) the second. The 301 oracle positions give
  top-1 against (a) and against llama.cpp as a cross-check.
- **Calibration text**, for the methods that use activations: 16 other rows
  (Wikipedia in five languages, UltraChat, CodeFeedback, NuminaMath,
  alpaca-zh), 3,739 tokens.
- **Methods**, all in the same format:
  - *min/max*: per 32 weights the minimum and range, rounded onto the
    grids (minimum down, step up); the packer until 2026-10-27.
  - *search*: llama.cpp's Q4_K fit (`make_qkx3_quants`): per group the
    weighted least-squares step and minimum over the code assignments of 37
    candidate steps, weighting each weight by av + |w|; then each group takes
    the sub-scale and sub-minimum within one grid step of its fit with the
    lowest error. *Equal weights* drops the |w| term; *refit* then solves d
    and dmin by least squares with the integers fixed.
  - *library*: `q4nx::PackRows` with `Fit::kSearch` (the search fit,
    vectorized), which `qwen35::Q4nxFit` names for this model: the NPU
    decoder, its MTP block and draft head load these weights, and the NPU
    prefill converts the decoder's buffers.
  - *imatrix*: search with each weight's error weighted by its input's mean
    x² from the calibration pass (llama.cpp's `qw * sqrt(sigma2 + w²)`);
    *mean x² alone* drops the sqrt term; *Unsloth's imatrix* takes the mean
    x² from the `imatrix_unsloth.gguf` published with this GGUF
    (847k calibration tokens; it has no LM head entry, which comes from our
    calibration pass).
  - *AWQ*: per-input scales s = mean|x|^α, rounded to powers of two and α
    chosen per matrix on 64 calibration inputs, multiplied into the weights
    and divided out of the input. Only where the NPU path can absorb 1/s
    exactly: in the norm weight of the mixers' input projections and of the
    LM head, in the CPU-written inputs of the mixers' output projections,
    and in the shared expert's up rows for its down projection. Not the
    shared expert's gate/up, which shares its NPU call and input with the
    routed Q4_K experts.

2026-10-27, deterministic:

| Method | (d) KL | (d) top-1 | (b) KL | (b) top-1 | (c) KL | (c) top-1 | (b) oracle top-1 vs (a) / llama.cpp | Packing, 32 threads |
|---|---|---|---|---|---|---|---|---|
| min/max (before) | 0.0316 | 93.33% | 0.0359 | 92.21% | 0.0394 | 91.95% | 274 / 269 | 0.4 ns per weight |
| search, equal weights | | | 0.0338 | 93.07% | | | 277 / 274 | |
| search | | | 0.0261 | 93.66% | | | 280 / 279 | |
| search + refit | | | 0.0267 | 93.80% | | | 278 / 277 | |
| **library (this model's fit)** | **0.0230** | **94.65%** | **0.0268** | **93.80%** | **0.0301** | **93.07%** | 278 / 277 | 1.4 ns per weight |
| imatrix | 0.0111 | 95.84% | 0.0149 | 94.65% | | | 281 / 281 | |
| imatrix, mean x² alone | | | 0.0157 | 95.51% | | | 282 / 280 | |
| Unsloth's imatrix | 0.0107 | 96.70% | 0.0147 | 95.64% | 0.0175 | 94.85% | 280 / 279 | |
| AWQ + search | | | 0.0194 | 94.79% | | | 280 / 276 | |
| AWQ + imatrix | | | 0.0153 | 94.92% | | | 283 / 284 | |

For scale: llama.cpp against (a) has KL 0.006 on the oracle prompts.

Reading:

- The search fit cuts KL by about a quarter on every set at no cost in
  format, kernel or decode speed, so it is this model's fit
  (`qwen35::Q4nxFit`). Each model names its own: Qwen3.5-4B takes it too,
  Qwen3.5-2B keeps min/max (its EXPERIMENTS.md). Its weighting of larger weights does most of that:
  with equal weights it gains only 6%. Refitting d and dmin adds nothing.
- Weighting by activation statistics halves KL again (0.0149 against 0.0261
  at (b)) and moves the oracle's top-1 against llama.cpp from 269 (min/max)
  and 279 (search) to 281. Unsloth's published imatrix does as well as our
  own calibration pass. Next step, not built: the NPU path needs the
  statistics at load time (an importance argument to `q4nx::PackRows`, the
  imatrix GGUF read by tensor name, per expert for the expert tensors, and
  a calibration-pass fallback for the LM head, which llama.cpp's imatrix
  leaves out).
- AWQ helps the data-free fit (0.0194) but adds nothing on top of the
  importance weighting (0.0153 against 0.0149), and it needs calibration
  data too.
- Packing the 35B's 1.9 B non-expert weights takes ~3–4 s on 32 threads
  (an AVX2 loop; the scalar version took 6.6 ns per weight). Per-text
  numbers are in the test's output; German Wikipedia and Chinese are the
  hardest texts for every method.

## NPU decode

`tests/models/qwen36_35b_a3b/npu_decode_test.cpp` (labels `npu`, `model`; same
variables plus the kernel directory): `NpuDecoder` (every matrix in q4nx on
the NPU, experts through `npu::Q4nxGather`, router and mixers on the CPU)
against the fp32 CPU `Decoder` over `Q4nxProjections`, the same q4nx weights
(Q4_K experts unpacked by q4nx's rule, everything else packed with
`q4nx::PackRows`, router and DeltaNet alpha/beta in fp32 as on the NPU
path). Teacher forced over the four oracle prompts (301 positions),
2026-10-27, after the experts' order in the gathers became ascending ids
(for MTP, below); the first measurements, with experts in order of weight,
were within the same bounds (logits ≤ 5.9e-3, argmax 301/301, top-1 against
llama.cpp 263/301).

The NPU rounds every call's input to bf16, so where a layer's 8th and 9th
experts are nearly tied it can route differently from the CPU, and both
decodes then carry that on. In a first version of the test, which let both
route on their own, that happened at position 0 or 1 in three of the four
prompts (gaps 3.2e-4 to 2.7e-2), so "agree until the first routing
difference" checked almost nothing. The test now makes the CPU follow the
NPU's routing (the other experts' router logits drop 64 below the lowest
chosen one, which leaves the chosen experts' renormalized weights exactly as
the CPU computes them), compares the logits at every position, and checks
separately that each routing the CPU would have made differently was a near
tie in the CPU's own logits: the gap is 1 − exp(l_x − l_y) for the
lowest-scored expert only the NPU chose and the highest-scored one only the
CPU's top 8 had.

| Prompt | Tokens | Logits rel. L2 (worst position) | Argmax vs CPU | Routings unlike the CPU's | Largest gap | NPU top-1 vs llama.cpp |
|---|---|---|---|---|---|---|
| chat | 17 | 4.3e-03 (6.2e-03) | 17/17 | 12 of 680 | 1.6e-02 | 16/17 |
| code | 15 | 8.1e-03 (2.1e-02) | 15/15 | 12 of 600 | 2.7e-02 | 14/15 |
| prose | 39 | 3.2e-03 (4.3e-03) | 39/39 | 24 of 1,560 | 5.2e-03 | 33/39 |
| long | 230 | 3.7e-03 (6.4e-03) | 228/230 | 134 of 9,200 | 1.0e-02 | 198/230 |
| all | 301 | | 299/301 | 182 of 12,040 (1.5%) | 2.7e-02 | 261/301 (86.7%) |

Thresholds:

| Check | Pass | Measured | Why |
|---|---|---|---|
| Logits rel. L2, per prompt | < 1e-2 | ≤ 8.1e-3 | the 2B's NPU bound; bf16 rounding of each call's input |
| Argmax vs CPU, all prompts | ≥ 98% | 299/301 | |
| Largest routing gap | < 5e-2 | 2.7e-2 | a misrouted expert lands far from the top 8 (gaps of order 1); bf16 noise moves router logits by ~1e-2 |

Letting both route on their own (the first version), the NPU and the CPU
still agreed on top-1 at 294/301 positions; the worst position's logits
were 4–12% apart, depending on the prompt. Against llama.cpp, top-1 agrees
at 261–263/301 positions on the NPU and at 260/301 on the CPU over the same
weights. That matches set (c) above: requantizing to q4nx sets the quality;
the NPU adds nothing measurable on top.

### MTP speculation

The same test then decodes greedily after each oracle prompt twice:
plainly (one token per pass after the prompt) and with MTP speculation (a
draft from the MTP block on the NPU, verified in a two-token pass that reads
the pair's expert union), both with the prompt in two-token passes. The
outputs must be identical token for token:

| After | Tokens | Identical | Verify steps | Drafts accepted |
|---|---|---|---|---|
| chat | 256 | yes | 145 | 110 (76%) |
| code | 48 | yes | 24 | 23 (96%) |
| prose | 48 | yes | 28 | 19 (68%) |
| long | 48 | yes | 32 | 16 (50%) |
| all | 400 | yes | 229 | 168 (73%) |

Identical, not merely close: a two-token pass gives each token exactly the
logits of a one-token pass (checked bit for bit over 3 positions × 248,320
logits and the final hidden states with a scratch probe), because each
token weighs the experts only the other token chose 0, those add exact
zeros, and the experts sum in ascending id order in both. Before the
experts were sorted by id they summed in order of weight, which differs
between a token's own call and the union. A bug the probe found on the way:
the mixer output's buffer view covered only the first 1.5 slots, enough for
Qwen3.5-2B's 2,048-wide mixers but not for this model's 4,096; the second
token's upper half never reached the device.

Determinism: `q4nx_moe_chain_test` (label `npu`) runs one layer's calls as
the decoder wires them (input and output projections, then the two gathers
and the next input projection, re-pointed between submissions) over five
selections: one token with 9 experts (twice), and two tokens with unions of
9, 13 and 17 experts, each token weighing about a third of them 0. 20,000
runs; every byte each selection writes is identical to its first run.
`q4nx_moe_gather_test` checks the same gathers against the CPU (worst
relative error 4.8e-4 to 7.9e-4).

## NPU prefill

`qwen36_35b_a3b::NpuPrefill` (src/models/qwen36_35b_a3b): every prompt
token but the last through the model layer by layer: the dense matrices as
bfp16 GEMMs on the NPU (bfp16 weights converted from the decoder's own
q4nx), the mixers and the routing on the CPU, the experts on the CPU
(`ExpertBackend::kCpu`, the default) or on the NPU (`kNpu`); then the
decoder runs the last token. Measured 2026-10-27 (NPU experts) and
2026-10-28 (CPU experts) on the box.

### CPU expert step (`qwen36_cpu_experts_test`, labels `qwen36_35b_a3b;kernels`)

Eight random q4nx experts (σ 0.03) and a shared one in the decoder's arena
layout, 37 tokens routed to 2–4 experts each plus the shared one (one
expert takes none), bf16 activations, against a double-precision
computation over the same q4nx weights (SwiGLU unrounded): the experts'
contribution to each token's residual.

| Path | Worst token rel. L2 | Repeat | Limit |
|---|---|---|---|
| Portable (any x86-64-v3 CPU; hosted CI) | 3.67e-3 | bit-identical | < 1e-2 |
| AVX-512 BF16 (the box) | 3.67e-3 | bit-identical | < 1e-2 |

(The experts are packed with the search fit, as the 35B packs its weights;
3.80e-3 with the min/max fit.)

The step rounds each q4nx weight once to bf16 (dmin·mn + d·sc·code built
exactly in fp32) and the SwiGLU output to bf16 before the down projection;
the sums are fp32. The test needs no NPU and runs on hosted CI.

### Ops (`qwen36_prefill_ops_test`, labels `qwen36_35b_a3b;npu;kernels`)

Random q4nx weights (σ 0.02) of every converted shape, against the CPU over
the same q4nx weights:

| Op | Shapes | Result | Limit |
|---|---|---|---|
| Dequant (`qwen36_35b_a3b_dequant`): q4nx in the GEMV's block order to bfp16 in the GEMM's B order | DeltaNet in, attention in, mixer out, one expert's gate + up and down | weights vs q4nx 6.170–6.174e-3, host packing 6.163–6.167e-3 | ≤ 1.05× the host's |
| GEMM (`qwen36_35b_a3b_prefill`) on the converted weights, M = 256, 512, 1024 | the same five | output rel. L2 9.13e-3 – 9.38e-3 | < 1e-2 |

The dequant now rounds each weight once (the spike's kernel gave 8.4e-3 and a
GEMM output of 1.08e-2). About 0.3% of its bytes still differ from the
host's: q4nx weights have few significant bits, so exact ties to an 8-bit
block mantissa are common and the core breaks some of them the other way;
the error is the host's to 0.1%.

### Against the decode path (`qwen36_npu_prefill_test`, labels `qwen36_35b_a3b;npu;model`)

Each prompt runs once through `NpuDecoder` one token per pass (recording each
layer's experts) and once through the prefill, which follows those experts
(`FollowRouting`: the decode path's choice, weighed by the prefill's own
router probabilities renormalized over it). Compared: every position's final
residual, every layer's conv window, DeltaNet state and K/V cache, the MTP
layer's K/V cache (`FillMtp` from the same hidden rows), then the logits of
the last token and 16 greedy steps from both states. Then the prefill routes
on its own: its first token against the decode path's, and a second run must
be bit-identical. The prompts are the oracle's four and repetitions of them
cut to 1, 2, 1,024, 1,025, 2,049 and 4,000 tokens.

| Prompt | Tokens | Hidden | Conv | DeltaNet state | K/V | MTP K/V | Logits | Argmax | Routings unlike the prefill's own | Own routing: first token (decode margin) | Hidden per token: median / max / over 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 17 | 8.6e-3 | 1.4e-2 | 1.8e-2 | 1.8e-2 | 1.2e-2 | 5.1e-2 | 17/17 | 44 of 640 | same (1.44) | 1.7e-2 / 3.6e-2 / 0 |
| code | 15 | 3.4e-2 | 1.4e-2 | 7.5e-2 | 3.7e-2 | 1.3e-2 | 4.3e-2 | 17/17 | 53 of 560 | same (3.87) | 1.7e-2 / 0.10 / 1 |
| prose | 39 | 1.6e-2 | 1.2e-2 | 1.6e-2 | 1.8e-2 | 1.4e-2 | 2.8e-2 | 17/17 | 110 of 1,520 | same (0.60) | 1.6e-2 / 2.9e-2 / 0 |
| long | 230 | 1.9e-2 | 1.5e-2 | 1.6e-2 | 1.9e-2 | 1.4e-2 | 2.6e-2 | 17/17 | 682 of 9,160 | differs, a tie (0.01) | 1.8e-2 / 3.3e-2 / 0 |
| 2 | 2 | 6.2e-3 | 1.2e-2 | 2.3e-2 | 2.1e-2 | 1.8e-2 | 2.1e-2 | 17/17 | 3 of 40 | same (7.84) | 6.2e-3 / 6.2e-3 / 0 |
| 1,024 | 1,024 | 4.5e-2 | 5.8e-2 | 6.1e-2 | 6.5e-2 | 1.5e-2 | 9.2e-2 | 17/17 | 7,759 of 40,920 | same (7.74) | 3.0e-2 / 0.52 / 45 |
| 1,025 | 1,025 | 4.5e-2 | 5.9e-2 | 5.2e-2 | 6.5e-2 | 1.5e-2 | 8.9e-2 | 17/17 | 7,776 of 40,960 | same (9.89) | 3.0e-2 / 0.52 / 45 |
| 2,049 | 2,049 | 7.4e-2 | 6.2e-2 | 6.0e-2 | 0.10 | 1.5e-2 | 0.21 | 17/17 | 21,282 of 81,920 | same (9.49) | 3.7e-2 / 0.88 / 205 |
| 4,000 | 4,000 | 8.2e-2 | 2.0e-2 | 5.6e-2 | 0.11 | 1.6e-2 | 0.12 | 17/17 | 47,085 of 159,960 | same (8.62) | 3.9e-2 / 0.88 / 426 |

(A one-token prompt prefills nothing; its row is all zeros.) Argmax agrees
170/170; every prompt's repeat is bit-identical; MTP speculation after the
prefill gives exactly the plain greedy output after it (48 tokens, 68% of
drafts kept). That table is the experts on the NPU (`kNpu`, the test's `npu`
argument, `qwen36_npu_prefill_npu_experts_test`); 2026-10-28 it gives the
same results.

With the experts on the CPU (the default, `qwen36_npu_prefill_test`),
2026-10-28, on the min/max-fit weights before #15: the same checks, the
errors equal or a little smaller (the experts' weights are rounded to bf16
per value rather than to bfp16 per 8, and their activations stay bf16):

| Prompt | Hidden | Conv | DeltaNet state | K/V | MTP K/V | Logits | Argmax | Routings unlike the prefill's own (largest gap) | Hidden per token: max / over 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| chat (17) | 6.0e-3 | 1.2e-2 | 1.6e-2 | 1.8e-2 | 1.2e-2 | 5.4e-2 | 17/17 | 43 of 640 (3.8e-2) | 3.8e-2 / 0 |
| code (15) | 2.4e-2 | 1.2e-2 | 4.8e-2 | 2.9e-2 | 1.3e-2 | 4.2e-2 | 17/17 | 45 of 560 (4.1e-2) | 7.4e-2 / 0 |
| prose (39) | 1.4e-2 | 1.1e-2 | 1.6e-2 | 1.7e-2 | 1.4e-2 | 2.8e-2 | 17/17 | 80 of 1,520 (3.4e-2) | 3.2e-2 / 0 |
| long (230) | 1.6e-2 | 1.3e-2 | 1.4e-2 | 1.9e-2 | 1.4e-2 | 2.8e-2 | 17/17 | 651 of 9,160 (6.4e-2) | 3.2e-2 / 0 |
| 2 | 2.2e-3 | 1.2e-2 | 2.2e-2 | 2.2e-2 | 1.8e-2 | 4.1e-2 | 16/17 | 3 of 40 (5.3e-3) | 2.2e-3 / 0 |
| 1,024 | 4.3e-2 | 5.6e-2 | 5.5e-2 | 7.1e-2 | 1.5e-2 | 9.8e-2 | 17/17 | 7,286 of 40,920 (1) | 0.59 / 40 |
| 1,025 | 4.3e-2 | 5.7e-2 | 5.1e-2 | 7.1e-2 | 1.5e-2 | 8.4e-2 | 17/17 | 7,299 of 40,960 (1) | 0.59 / 40 |
| 2,049 | 7.3e-2 | 3.0e-2 | 5.0e-2 | 0.11 | 1.5e-2 | 0.16 | 17/17 | 20,483 of 81,920 (1) | 0.91 / 191 |
| 4,000 | 8.0e-2 | 2.9e-2 | 5.1e-2 | 0.11 | 1.6e-2 | 0.12 | 17/17 | 45,142 of 159,960 (1) | 0.91 / 370 |

Argmax 169/170 (the 2-token prompt's one teacher-forced step that flips is
within the 95% limit); the first token is the decode path's on 9 of 10
prompts (the 230-token passage again a 0.01-logit tie); every repeat is
bit-identical; speculation after the prefill equals plain greedy (48 tokens,
57% of drafts kept). The same limits hold.

With the search fit (#15), 2026-10-28, both backends pass the same limits:

| Experts | Argmax | First token (own routing) | 1,024: hidden / logits | 2,049: hidden / K/V / logits | 4,000: hidden / K/V / logits | Speculation after prefill |
|---|---|---|---|---|---|---|
| CPU (default) | 169/170 | same on 10 of 10 | 4.5e-2 / 0.17 | 7.2e-2 / 9.9e-2 / 0.11 | 8.2e-2 / 0.11 / 0.17 | equals greedy (48 tokens, 66% kept) |
| NPU | 167/170 | same on 10 of 10 | 4.1e-2 / 0.16 | 7.5e-2 / 0.10 / 0.12 | 8.5e-2 / 0.11 / 0.16 | equals greedy (48 tokens, 66% kept) |

The NPU backend's 167/170 is #15's own measurement again (near ties in the
decode path's top two after routing diverges on the long prompts); every
repeat is bit-identical on both.

**With the search fit** (this model's fit since 2026-10-28, `qwen35::Q4nxFit`)
argmax agrees 167/170 (pass ≥ 95%); the states after prefill are as close to
the decode path's as with min/max or closer (len1024/1025: hidden 4.1e-2
against 4.5e-2, conv 5.3e-2 against 5.9e-2, ssm 3.9e-2 against 5.2e-2, kv
5.5e-2 against 6.5e-2). The test prints each top-1 change with both sides'
top-2 gaps, and for every prompt each step's logits error and the number of
layers routed unlike the decode path's same step:

| Prompt, step | Decode path's top-2 gap | After prefill: top-2 gap |
|---|---|---|
| prose, step 5 | 0.042 | 0.065 |
| len2, step 11 | 0.008 | 0.019 |
| len1025, step 6 | 1.114 | 0.420 |

Two are near ties. The third is not, but it sits in a divergence the
min/max weights show as well: from 1,024 prompt tokens on, the steps after
prefill route unlike the decode path in 15–34 of the 40 layers at nearly
every step with either fit, and single steps reach logits errors of 0.3–0.65
on main's weights too (len2049 steps 0, 2, 8 and 12; len4000 step 6) without
changing top-1 there. At len1025 step 6 the search fit's run has a step error
of 0.097 with 19 layers routed differently, and top-1 changes. The prefill
itself is not less accurate; how often a long prompt's routing divergence
flips a top-1 is luck of the positions, and the GEMM's bfp16 rounding is what
a tighter bound needs (as above).

**Where the error comes from.** It is the GEMMs' bfp16 rounding (activations
rounded to bfp16 inside the mmul, 8 values sharing an exponent; outputs to
bf16), compounding over 40 layers, not the batched CPU side: the family's
mixers (src/models/qwen35/prefill_mixer.cpp, also the 4B's arithmetic) match
token-by-token decode to ~1e-4 when the GEMMs are replaced by exact fp32
products ([the 4B's measurement](../qwen3.5-4b/QUALITY.md)). On natural text
every token stays under 0.1 (the 230-token passage: largest 3.3e-2). On the
long repetitive prompts a minority of tokens drift further (45 of 1,023
tokens over 0.1 at 1k, ~10% at 2k and 4k, up to 0.88), and routing
amplifies it: once a token's residual is off, its router logits can rank
experts very differently, so the share of routings unlike the decode path's
grows from 7–10% on the short prompts to 19–29% on the long ones, and the largest gap reaches 1
(an expert the prefill's own router all but rules out). The decode test's
near-tie check (`kNearTie`) therefore does not carry over; the prefill test
reports the gap and checks the states, the logits and the argmax instead.
The one first-token difference (the 230-token passage) is a tie: the decode
path's top two are 0.01 apart.

**Limits.** Hidden < 0.15, conv / DeltaNet state / K/V and MTP K/V < 0.2,
logits < 0.3, argmax ≥ 95%, first token identical unless the decode path's
top two are within 0.1 logits: about 1.5–2× the largest measured. A wrong
offset, a stale window or a misrouted token gives errors of order 1. A more
precise GEMM (a second, lo-part pass, or bf16 activations in the expert
GEMMs) would tighten them; that is open (EXPERIMENTS.md).

## GGUF load

`tests/models/qwen36_35b_a3b/gguf_load_test.cpp` (label `model`): ReadConfig gives
the shape in [README.md](README.md) (40 + 1 layers, 10 full-attention, 16/32
DeltaNet heads, 256/8 experts of width 512, shared expert 512, untied head),
and every tensor dequantizes to finite values (753 tensors, 35.5 B values,
~2.5 s). `tests/models/qwen35/config_test.cpp` (hosted CI) covers the same
fields on synthetic `qwen35` / `qwen35moe` headers, plus inconsistent ones.

## Prompt cache (2026-10-28)

`qwen36_prompt_cache_test` (labels `qwen36_35b_a3b;npu;model`; how the cache works is in
[BENCHMARKS.md](BENCHMARKS.md), "Prompt cache across requests"). A chat
over ~1k tokens of notes, turns 2 and 3, each state against the whole
prompt through the model's own `PrefillPrompt`: the decode state (every
layer's conv window, DeltaNet state and K/V cache), the MTP layer's K/V
cache and the last prompt row's final residual.

| Check | Turn 2 / turn 3 |
|---|---|
| The previous turn's checkpoint restored, the rest (~40 tokens) through the prefill | **bit-identical** (every error 0) |
| An exact repeat: the checkpoint restored, the generation prompt through the prefill | **bit-identical** |
| A 5-token remainder through the decoder (below the prefill threshold) | hidden 4.4e-2 / 6.3e-2, states 6.2e-2 / 7.2e-2 |
| The same prompt entirely through the decode path, for scale | hidden 6.9e-2 / 6.9e-2, states 0.23 / 0.22 (its routing drifts from the prefill's over 1k tokens, as "NPU prefill" above shows) |
| The checkpoint the decoder took mid-remainder, continued through the prefill | hidden 5.4e-2 / 9.7e-2, states 6.1e-2 / 8.6e-2 |
| Greedy MTP output, 3 turns at ~1k and ~4k tokens of history, two rounds: with the cache (hits) and on a miss, against `PrefillPrompt` | **identical** every turn |

- **Bit-identical restore:** the prefill's arithmetic for a token does not
  depend on where a pass starts. Its GEMM rows are independent, the
  DeltaNet runs token by token, and attention reads the same cache, so a
  checkpoint plus the rest equals one prefill exactly.
- **The decoder's remainders** differ from the prefill as the decode path
  always does (bfp16 GEMMs against q4nx GEMVs), and no more: within the
  limits of the model's prefill test (hidden 0.15, states 0.2).
- **Untouched without a cache:** the capture is off unless a checkpoint
  is asked for, and outputs keep their hashes: the 960-token prompt with MTP `1de5ad9f` (as #15 recorded); the sky prompt with MTP `2c9f87d3`.
- **Cache logic** (`qwen35_prompt_cache_logic_test`, CPU, label
  `qwen35_family`): on a toy model with an order-dependent recurrent
  layer, a restored run leaves exactly the uncached state over four
  turns, with and without the MTP cache and with the last token left to
  the decoder; short prompts go to the decoder untouched; a repeat runs
  only the generation prompt; least-recently-used and byte-budget
  eviction; and a draft request skips checkpoints without the MTP cache.
