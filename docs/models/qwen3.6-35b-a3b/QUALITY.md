# Qwen3.6-35B-A3B quality

Planned contracts (see [the plan](../../PLAN.md)):

| Check | Tolerance | State |
|---|---|---|
| CPU reference vs llama.cpp (CPU), teacher-forced logits | see below | enforced |
| Each NPU op vs CPU reference | relative error < 1e-2 | enforced: MoE gathers 6.1e-4 (1 token), 8.6e-4 (2) |
| NPU decode vs CPU decoder over the same q4nx weights, teacher-forced, routing followed | logits rel. L2 < 1e-2 per prompt, argmax ≥ 98%, routing differences at near ties only | enforced (below) |
| Prefill vs one-token decode | identical next token | reference: identical logits |
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
2026-10-27, deterministic across two runs:

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
rest page cache of the memory-mapped GGUF; ~7 minutes.

## NPU decode

`tests/models/qwen36_35b_a3b/npu_decode_test.cpp` (labels `npu`, `model`; same
variables plus the kernel directory): `NpuDecoder` (every matrix in q4nx on
the NPU, experts through `npu::Q4nxGather`, router and mixers on the CPU)
against the fp32 CPU `Decoder` over `Q4nxProjections`, the same q4nx weights
(Q4_K experts unpacked by q4nx's rule, everything else packed with
`q4nx::PackRows`, router and DeltaNet alpha/beta in fp32 as on the NPU
path). Teacher forced over the four oracle prompts (301 positions),
2026-10-27.

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
| chat | 17 | 4.4e-03 (7.4e-03) | 17/17 | 9 of 680 | 5.0e-03 | 16/17 |
| code | 15 | 5.9e-03 (1.4e-02) | 15/15 | 8 of 600 | 2.7e-02 | 14/15 |
| prose | 39 | 3.4e-03 (5.8e-03) | 39/39 | 24 of 1,560 | 3.6e-03 | 34/39 |
| long | 230 | 3.8e-03 (6.7e-03) | 230/230 | 129 of 9,200 | 6.3e-03 | 199/230 |
| all | 301 | | 301/301 | 170 of 12,040 (1.4%) | 2.7e-02 | 263/301 (87.4%) |

Thresholds:

| Check | Pass | Measured | Why |
|---|---|---|---|
| Logits rel. L2, per prompt | < 1e-2 | ≤ 5.9e-3 | the 2B's NPU bound; bf16 rounding of each call's input |
| Argmax vs CPU, all prompts | ≥ 98% | 301/301 | |
| Largest routing gap | < 5e-2 | 2.7e-2 | a misrouted expert lands far from the top 8 (gaps of order 1); bf16 noise moves router logits by ~1e-2 |

Letting both route on their own (the first version), the NPU and the CPU
still agreed on top-1 at 294/301 positions; the worst position's logits
were 4–12% apart, depending on the prompt. Against llama.cpp, top-1 agrees at 263/301 positions on
the NPU and at 260/301 on the CPU over the same weights. That matches set
(c) above: requantizing to q4nx sets the quality; the NPU adds nothing
measurable on top.

Determinism: `q4nx_moe_chain_test` (label `npu`) runs one layer's calls as
the decoder wires them (input and output projections, then the two gathers
and the next input projection, re-pointed between submissions at four
expert selections) 20,000 times; every output is bit-identical to the first
run with the same selection.

## GGUF load

`tests/models/qwen36_35b_a3b/gguf_load_test.cpp` (label `model`): ReadConfig gives
the shape in [README.md](README.md) (40 + 1 layers, 10 full-attention, 16/32
DeltaNet heads, 256/8 experts of width 512, shared expert 512, untied head),
and every tensor dequantizes to finite values (753 tensors, 35.5 B values,
~2.5 s). `tests/models/qwen35/config_test.cpp` (hosted CI) covers the same
fields on synthetic `qwen35` / `qwen35moe` headers, plus inconsistent ones.
