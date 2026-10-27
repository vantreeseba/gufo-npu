# Qwen3.6-35B-A3B quality

Planned contracts (see [the plan](../../PLAN.md)):

| Check | Tolerance | State |
|---|---|---|
| CPU reference vs llama.cpp (CPU), teacher-forced logits | see below | enforced |
| Each NPU op vs CPU reference | relative error < 1e-2 | |
| NPU model vs CPU reference, teacher-forced | greedy argmax match | |
| Prefill vs one-token decode | identical next token | reference: identical logits |
| MTP head vs reference | cosine > 0.999 | |

There is no HF transformers oracle for this model here (the BF16 weights are
70 GB); llama.cpp on the same GGUF is the independent implementation.

## CPU reference vs llama.cpp

`tests/models/qwen36/reference_llama_test.cpp` (label `model`;
`GUFO_NPU_QWEN36_MODEL` is the UD-Q4_K_XL GGUF, `GUFO_NPU_QWEN36_ORACLE` the
output of `tools/reference/qwen36_llama_oracle.py`). llama.cpp
0c1e57098bba43ac29e6e3b677cdceebdd22334f via llama-cpp-python 0.3.36, built
CPU-only (AVX2/FMA/F16C, [TOOLCHAIN.md](../../TOOLCHAIN.md)); whole prompt in
one batch, fp32 KV cache, no flash attention, no weight repacking. Same GGUF,
same token ids, three prompts (chat template, code, prose; 71 tokens).
Measured 2026-10-27.

| Prompt | Tokens | End to end: logits rel. L2 | Argmax | llama.cpp vs itself¹: rel. L2 | Argmax | Layer by layer²: outputs < 1e-5 | Worst | LM head³ rel. L2 |
|---|---|---|---|---|---|---|---|---|
| chat | 17 | 7.9e-02 | 16/17 | 7.3e-02 | 16/17 | 578/680 (85.0%) | 4.5e-03 | 1.5e-07 |
| code | 15 | 3.5e-02 | 15/15 | 3.2e-02 | 15/15 | 506/600 (84.3%) | 5.2e-03 | 1.6e-07 |
| prose | 39 | 2.4e-02 | 39/39 | 2.6e-02 | 38/39 | 1279/1560 (82.0%) | 4.6e-03 | 1.4e-07 |

1. llama.cpp one token per `llama_decode` against llama.cpp on the whole
   prompt (recorded in the oracle's manifest).
2. Each decoder layer run alone on llama.cpp's own input to it (its
   `l_out-<il - 1>`), against llama.cpp's output, per token, with
   llama.cpp's activation rounding applied (below). Median 1e-7.
3. The final norm and LM head on llama.cpp's last hidden state, same
   rounding; argmax 71/71.

Also: our tokenizer gives llama.cpp's ids for all three prompts, and
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
Fed llama.cpp's own layer inputs, most outputs then agree to float
precision. The rest (15–18%) follow a single 8-bit rounding that came out
differently because the input differed in its last bits (accumulation
order): one rounding step on one element of a block with an outlier moves
the output by ~1e-3, and the DeltaNet state and the KV cache carry it to the
later tokens of that layer. A layer-0 trace (prose) showed exactly this:
DeltaNet outputs equal to 1e-7 until a Q8_0 flip in `attn_qkv`'s input at
token 31, and expert outputs equal to 1e-7 except on tokens whose FFN input
had a flipped `ssm_out` rounding upstream. Same experts chosen for every
token, router weights equal to 1e-4.

Thresholds, enforced per prompt:

| Check | Pass | Measured | Why |
|---|---|---|---|
| End to end logits rel. L2 | < 0.15 | ≤ 7.9e-2 | 2× llama.cpp's spread against itself (7.3e-2) |
| End to end argmax, all prompts | ≥ 95% | 70/71 | llama.cpp against itself: 69/71 |
| Layer outputs within 1e-5 | ≥ 70% | ≥ 82.0% | a math error moves every output, a rounding flip only some |
| Worst layer output | < 2e-2 | ≤ 5.2e-3 | a flip moves ~1e-3, carried flips a few 1e-3 |
| LM head logits rel. L2 | < 1e-5 | ≤ 1.6e-7 | same rounding, only accumulation order left |
| LM head argmax, all prompts | 71/71 | 71/71 | |
| Token by token vs whole prompt | < 1e-6 | 0 | |

The one end-to-end top-1 difference (chat, position 12, after
`<|im_start|>`) is a near tie in llama.cpp (gap 0.70 logits) that the
reference resolves to the token that actually follows (`assistant`, gap
1.51). llama.cpp run one token at a time picks `assistant` there too (its
other change is prose position 11, gap 0.10).

### Reference speed and memory

Release build, 32 threads (Ryzen AI MAX+ 395), the GGUF in page cache, same
run: whole-prompt forward 2.7–3.9 tok/s (17–39 tokens: each expert's rows
are dequantized once per prompt), one token at a time 1.2 tok/s. Peak RSS
13.5 GB, all but 0.25 GB of it file-backed pages of the memory-mapped GGUF
(only the experts the prompts touch are read); nothing is dequantized ahead
of use. The full test takes ~60 s.

## Decoder vs reference

`tests/models/qwen36/decoder_test.cpp` (label `model`, same variables): the
one-token decode step (`Decoder`, fp32 with an fp16 K/V cache, the MoE
routing of `moe.hpp`) against the reference, token by token over the chat
prompt, both on the reference's GGUF projections, so only the non-matrix
math differs. 2026-10-27:

| Check | Pass | Measured |
|---|---|---|
| Logits rel. L2 up to the first token routed differently | < 2e-3 | 3.6e-4 (2B decoder: 4.3e-4) |
| Reference's 8th/9th expert gap where routing first differs | < 1e-3 | 1.3e-4 (position 13, layer 36) |
| Argmax | 17/17 | 17/17 |

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

## GGUF load

`tests/models/qwen36/gguf_load_test.cpp` (label `model`): ReadConfig gives
the shape in [README.md](README.md) (40 + 1 layers, 10 full-attention, 16/32
DeltaNet heads, 256/8 experts of width 512, shared expert 512, untied head),
and every tensor dequantizes to finite values (753 tensors, 35.5 B values,
~2.5 s). `tests/models/qwen35/config_test.cpp` (hosted CI) covers the same
fields on synthetic `qwen35` / `qwen35moe` headers, plus inconsistent ones.
