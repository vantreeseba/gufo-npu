# Qwen3.5-2B benchmarks

Hardware: Strix Halo (Ryzen AI MAX+ 395), 64 GB, Ubuntu 26.04, Linux 7.0,
xdna-driver 2.26.

## FastFlowLM baseline

`qwen3.5:2b` (FLM Q4_K pack, footprint 3.1 GB) served by Lemonade 2026.43.3.
Measured 2026-10-27 with `tools/bench/openai_bench.py`: streaming chat,
temperature 0, unique prompt per request, model loaded, 2 warm-up requests,
3 repeats, 256 generated tokens. Two independent sweeps agree within ~1%; the
table is the mean of both (6 requests per row).

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 35 | 0.60 | 58.6 | 28.5 |
| 125 | 0.61 | 204.7 | 28.4 |
| 1,055 | 1.80 | 586.6 | 28.0 |
| 4,246 | 3.34 | 1,272 | 26.4 |
| 8,499 | 6.12 | 1,388 | 24.6 |

- Decode 28.5 tok/s at ~1.1 GB per token is ~31 GB/s effective, the same
  rate FLM reaches on the 35B MoE (~30 GB/s), against our measured ~80 GB/s
  combined copy bandwidth.
- Fixed TTFT is ~0.6 s here against ~3.3 s on the 35B, and prefill at 8.5k is
  5.6× faster than the 35B with similar active parameters. FLM's MoE expert
  handling, not the shared layers, is what makes the 35B's prefill slow.

## Targets

| | FLM | Target |
|---|---|---|
| Decode, 1k context | 28.0 tok/s | ≥ 38 tok/s (ceiling ~45–50) |
| TTFT, short prompt | 0.60 s | < 0.3 s |

## NPU DDR bandwidth

`gufo-npu bench bandwidth`, 2026-10-27, two runs:

| Probe | Mean | Best |
|---|---|---|
| Copy 64 MiB through all columns (read + write counted) | 79.0 GB/s | 79.9 GB/s |
| Read 256 MiB only, 2 shim channels per column | 55.9 GB/s | 56.4 GB/s |
| Read only, 1 channel per column (one-off probe) | 55.5 GB/s | 55.9 GB/s |

Read-only by column count (one channel per column, one-off probe; the 16 MiB
column has the ~62 µs dispatch taken out):

| Columns | 256 MiB | 16 MiB (fits the 32 MB Infinity Cache) |
|---|---|---|
| 1 | 13.8 GB/s | 13.6 GB/s |
| 2 | 27.1 GB/s | 27.0 GB/s |
| 4 | 49.0 GB/s | 50.7 GB/s |
| 8 | 55.9 GB/s | 58.4 GB/s |

The copy's 79 GB/s is ~40 GB/s each way at once. Reads scale ~14 GB/s per
column up to 4 columns, then stop at ~56 GB/s however many channels issue
them: a shared limit on the NPU's side of the fabric, not per column and not
DRAM. A hot 16 MiB buffer reads no faster, so NPU reads are not served from
the Infinity Cache (which only GPU traffic fills). **56 GB/s is the decode
ceiling**: decode only reads weights.

For comparison, the CPU (AVX2 loads over a 1 GiB buffer, best of five,
2026-10-27) reads 39.8 GB/s on one thread, 62.1 on two and 113–116 GB/s
on 4 to 32 threads: twice the NPU. Work that is all reads of state rather
than weights (the attention K/V cache) is better done on the CPU.

## NPU q4nx GEMV

`gufo-npu bench gemv --model Qwen3.5-2B-UD-Q4_K_XL.gguf`: one stacked q4nx
matrix per decode stage, resident on the device; random bf16 token. 20
warm-up dispatches, then the mean over 200 (start + wait, no host copies).
Runs agree within 2%. All shapes share one xclbin and hardware context.
Measured 2026-10-27.

| Stage | GGUF types | dim → rows | MB | Mean µs | GB/s | Err NPU | Err total |
|---|---|---|---|---|---|---|---|
| DeltaNet in (qkv + z) | Q6_K, Q5_K | 2048 → 8192 | 10.5 | 253 | 41.4 | 4.1e-6 | 7.1% |
| Attention in (q + k + v) | Q5_K, Q5_K, Q6_K | 2048 → 5120 | 6.6 | 187 | 35.1 | 6.2e-6 | 8.1% |
| Mixer out (ssm_out) | Q8_0 | 2048 → 2048 | 2.6 | 117 | 22.3 | 5.9e-6 | 8.4% |
| FFN in (gate + up) | Q4_K, Q4_K | 2048 → 12288 | 15.7 | 349 | 45.0 | 7.2e-6 | 0.50% |
| FFN down | Q6_K | 6144 → 2048 | 7.9 | 211 | 37.3 | 5.9e-6 | 8.0% |
| LM head (token_embd) | Q6_K | 2048 → 248320 | 318.5 | 5,702 | 55.9 | 6.2e-6 | 7.3% |

- "Err NPU": relative L2 error against a double-precision GEMV over the same
  packed weights and bf16 token. "Err total" compares against the GGUF
  weights, so it adds requantization (and the bf16 rounding of Q4_K scales).
- The LM head streams at the read ceiling. Small matrices pay a fixed
  ~66 µs per dispatch (`bench dispatch`).
- The kernel rounds to nearest even (a core starts in floor mode: 1.3%
  error), carries each block's dot product and each per-32 token sum as two
  bf16 halves, and writes fp32. That took its error from 4.5e-3 to 6e-6 at
  the same speed.

## NPU decode, end to end (Phase 3)

`gufo-npu generate --device npu --model Qwen3.5-2B-UD-Q4_K_XL.gguf --prompt
"Name three primary colors, briefly." --max-tokens 128` (19 prompt tokens,
stops after 64). Three runs in a row per row, 2026-10-27:

| Path | Decode tok/s | ms per token | TTFT (19 tokens) |
|---|---|---|---|
| CPU reference (GGUF weights) | 5.55 | 180 | 1.83 s |
| NPU GEMVs + CPU reference math (97 submissions/token) | 22.4 | 44.8 | 0.85 s |
| NPU GEMVs + fp32 CPU decoder (97 submissions/token) | 32.5 | 30.7 | 0.59 s |
| **NpuDecoder: chained, norms/residual/SwiGLU on the NPU (25 submissions/token)** | **35.5** (35.56, 35.41, 35.40) | **28.2** | 0.54 s |
| FLM (Lemonade), 35-token prompt | 28.5 | 35.1 | 0.60 s |

- FLM's number comes from its server (`tools/bench/openai_bench.py`, 256
  generated tokens); ours from the CLI with a 64-token reply. Both are warm
  and short-context; a like-for-like run waits for `gufo-npu serve`
  (Phase 5).
- NpuDecoder: every RMS norm before a projection, every residual add and
  SwiGLU run inside the GEMV (input and output stages), and each GEMV writes
  into the next one's input buffer, so out → gate+up → down → next layer's
  in is one runlist. The CPU runs only the DeltaNet/attention mixers and the
  16+16 alpha/beta rows.
- Of the 28.2 ms: ~26.1 ms waiting on the NPU, ~2.1 ms CPU. Streaming 1.2 GB
  at 56 GB/s is ~21.4 ms; the rest is ~62 µs per submission (25), ~20 µs per
  chained call (72) and ~20 µs of ramp-up and drain inside each call (token
  broadcast, input stage, first and last blocks).
- Remaining levers: one instruction sequence per chain (no per-call runlist
  cost), the mixers on the NPU (one submission per token, no CPU), and
  speculative decoding with the MTP head (a few tokens per weight read; a
  4-token GEMV costs about the same as one).
- Prompts run token by token through the decode path (prefill is Phase 4).

### Two tokens per weight read (speculative decoding, step 1)

Every GEMV variant also takes two tokens per call (the weights are read once
and each block is widened once for both). `gufo-npu bench gemv`, two runs,
2026-10-27:

| Stage | 1 token µs | 2 tokens µs | 2 vs 1 |
|---|---|---|---|
| DeltaNet in (qkv + z) | 273–279 | 288–295 | 1.06× |
| Attention in (q + k + v) | 206–208 | 223–224 | 1.07× |
| Mixer out (ssm_out) | 125–134 | 126–147 | 1.01–1.10× |
| FFN in (gate + up, SwiGLU) | 376 | 392–393 | 1.04× |
| FFN down | 224–225 | 232 | 1.03× |
| LM head | 5,775–5,782 | 5,792–5,803 | 1.00× |

A second token costs 0–10% per call: verifying one draft token per step
reads the weights once. Single-token decode after the change: 35.27, 35.41,
35.05 tok/s (35.5 before; the token buffer grew from 24 to 36 KB).

### MTP speculative decoding (step 2)

`gufo-npu generate --device npu` now drafts with the model's MTP head
(`--draft mtp`, the default) and verifies each draft in a 2-token pass: the
output is the main model's greedy output, byte for byte (checked against
`--draft none` here and in `qwen35_npu_decode_test`). The prompt also runs
two tokens per pass. Warm runs, 2026-10-27:

| Prompt | `--draft none` | `--draft mtp` | Drafts accepted | Steps / tokens |
|---|---|---|---|---|
| "Explain in a few sentences why the sky is blue." | 34.8 tok/s | **43.8** (43.78, 43.97, 43.61) | 72% | 50 / 87 |
| "Write a Python function that returns the n-th Fibonacci number…" | 34.1 tok/s | **47.6** (47.41, 47.61, 47.75) | 93% | 122 / 237 |

- A step is one 2-token verify pass (~29.5 ms, 25 submissions) plus an MTP
  draft (~7.5 ms: eh_proj and the MTP block in 2 submissions, then the
  shared LM head at 5.8 ms). The draft's LM head is the largest fixed cost
  left; a draft-only vocabulary subset would cut it.
- Acceptance measured offline (`gufo-npu bench mtp`, draft vs the greedy
  token two positions ahead): 72.4% on the first prompt with the CPU and NPU
  heads alike. RoPE position and whether h goes through output_norm first
  make no difference (a constant RoPE shift cancels in attention).

### Step 3a: CPU mixer

Profiling the CPU side of a step (DeltaNet, attention, alpha/beta for two
tokens over 25 blocks): buffer syncs and copies are ~3% of it; the math is
the rest. DeltaNet heads and the conv now run on a persistent spin pool, RoPE
cos/sin are computed once per position, and the decoder's host code is
built with AVX2 + FMA. Same prompt as above (sky), three runs:

| | tok/s | CPU mixer time (whole run) |
|---|---|---|
| Before | 43.6–44.0 | 0.32 s |
| After | **45.6–45.8** | 0.21 s |

A step is now ~34 ms of NPU waits (~27.5 ms streaming, the rest per-call and
per-submission cost) and ~3.4 ms of CPU. A draft vocabulary restricted to
the lowest token ids was tried and dropped: acceptance fell a few points on
English and code and from 62% to 14% on Chinese (CJK tokens have high ids).

### Step 3c: long context (CPU attention)

At 2k tokens of context the CPU attention dominated: one thread, scalar
dot products. Query heads now run on the spin pool and the score dot
products use AVX2 (`vec::Dot`). Prompt of 2,120 tokens ("Item i: the quick
brown fox…", then the sky question), 93 generated tokens, old and new
binaries interleaved, warm, 2026-10-27:

| | Before | After |
|---|---|---|
| Decode, `--draft none` | 24.28, 24.24 tok/s | **33.22, 33.13, 33.09** |
| Decode, `--draft mtp` | 26.38, 26.38 | **41.32, 41.40, 41.13** |
| Prompt (token by token), `--draft none` | 44.0 s | 33.7 s |
| CPU mixer time, `--draft none` run | 16.2 s | 4.9 s |
| Short prompt (sky), `--draft mtp` | 45.84, 45.80, 45.54 | 46.76, 46.83, 46.86 |

A third pair of runs was disturbed by another process on the NPU and is
left out. At 2k context, decode now runs ~6% below short context (no draft:
30.2 vs 28.2 ms per token). The extra ~2 ms is mostly attention reading
the fp32 K/V cache (6 layers x 2 KV heads x 256 x 2 x 4 B = 24.6 KB per
position, 52 MB per token at 2,120), about 26 GB/s effective.

The DeltaNet alpha/beta rows (32 x 2048 per token and layer, on the CPU)
were a scalar loop with one running sum, which does not vectorize; they and
the q/k L2 norms now use `vec::Dot`. Same binaries otherwise, warm,
2026-10-27:

| | Before | After |
|---|---|---|
| Short prompt (sky), `--draft none` | ~35.2 tok/s | **37.20, 37.18** (35.87 disturbed) |
| Short prompt (sky), `--draft mtp` | 46.76, 46.83, 46.86 | **47.70, 47.91, 47.64** |
| 2,120-token prompt, `--draft mtp` | 41.32, 41.40, 41.13 | 41.67, 41.37, 41.65 |

The CPU side of a single-token step is now ~0.9 ms of ~26.9 ms (NPU waits
~26 ms).

Attention then went split-K (flash decoding): a job takes one KV head and a
slice of at least 128 positions, reads each key and value once for the 4
query heads that share it, and leaves a partial softmax; the slices are
merged per head. Same 2,120-token prompt, `--draft none`, warm:

| | Per query head | Split-K |
|---|---|---|
| Decode | 33.22, 33.13, 33.09 tok/s | **34.23, 34.59, 34.59** |
| Prompt (token by token) | 33.7 s | 32.3–32.5 s |
| CPU mixer time | 4.9 s | 3.5–3.6 s |

Decode at 2k context is now ~8% below short context (no draft: 29.1 vs
26.9 ms per token). The extra ~2.2 ms is mostly attention over the 52 MB
of K/V, still well below the CPU's 115 GB/s (a bf16 cache would halve the
bytes).

### fp16 K/V cache

The attention cache now holds fp16 (llama.cpp's default cache type: half
the bytes of fp32, 3 more mantissa bits than bf16); F16C converts on load.
Prompts of 2,120 and 9,050 tokens ("Item i: …", then the sky question),
the committed build and this one interleaved, warm, 2026-10-27:

| | fp32 K/V | fp16 K/V |
|---|---|---|
| 2k context, `--draft none` | 34.56, 34.62, 34.57 tok/s | **35.49, 35.67, 35.45** |
| 2k context, `--draft mtp` | 47.56, 47.28, 46.17 | **48.20, 48.40, 48.70** |
| 9k context, `--draft none` | 28.67, 28.68, 28.63 | **31.62, 31.84, 31.85** |
| 9k prompt (token by token) | 160.2–160.8 s | 146.9–148.0 s |
| CPU mixer time, 9k run | 36.4–37.1 s | 23.1–24.2 s |

At 9k context the cache is 222 MB per token in fp32 and 111 MB in fp16;
decode there is now 18% below short context (31.8 vs 37.2 tok/s, no
draft).

### Draft vocabulary for the MTP head

A draft spent 5.8 of its ~7.5 ms in the LM head (248,832 rows, 290 MB). It
now scores a draft vocabulary of 32,768 ids instead (`DraftVocab()`, 0.8
ms): every special token, then the top of a ranking built by
`tools/draft_vocab/rank_tokens.py` from a multilingual sample, weighted
toward chat, code and the most used languages. The main model still checks
each draft over the whole vocabulary, so the output is unchanged
(`qwen35_npu_decode_test`). The MTP cache over the prompt (`FillMtp`) skips
the LM head entirely. Warm, three runs each, 2026-10-27:

| Prompt | Full vocabulary | Draft vocabulary (32k) | Drafts accepted |
|---|---|---|---|
| "Explain in a few sentences why the sky is blue." | 47.70, 47.91, 47.64 tok/s | **52.09, 53.00, 52.99** | 72% → 65% |
| "Write a Python function that returns the n-th Fibonacci number…" | 47.6 (earlier build) | **61.23, 60.99, 60.69** | 95% |
| "用几句话解释为什么天空是蓝色的。" | – | 44.55, 44.68, 44.72 | 43% → 40% (offline) |
| 2,120-token prompt, then the sky question | 41.67, 41.37, 41.65 | **46.62, 47.25, 46.46** | |
| 2,120-token prompt: time to first token | 40.3–40.8 s | 32.9 s | |

Offline acceptance by draft vocabulary size (`gufo-npu bench mtp`, NPU
greedy tokens, CPU MTP head restricted to the top K of the ranking):

| Prompt | Full | 8k | 16k | 32k | 48k | 64k |
|---|---|---|---|---|---|---|
| English (sky) | 72.4% | 54.0 | 62.1 | **65.5** | 69.0 | 70.1 |
| Python (Fibonacci) | 95.0% | 92.0 | 92.0 | **93.5** | 94.5 | 94.5 |
| Math (3x + 7 = 22) | 89.5% | 85.0 | 86.5 | **89.0** | 89.0 | 90.0 |
| Chinese (sky) | 42.9% | 28.6 | 35.7 | **40.5** | 45.2 | 42.9 |
| Japanese (sky) | 53.4% | 43.9 | 48.0 | **49.3** | 52.0 | 52.0 |
| Spanish (sky) | 68.9% | 42.2 | 44.4 | **51.1** | 51.1 | 60.0 |

A step costs about 29.5 ms (verify) plus 1.7 ms (MTP block) plus the draft
LM head, so 48k ids would add 0.4 ms for 1–4 points: about even. A first
ranking that gave every source the same say (a token's best frequency in
any of them) reached only 51.7% on the English prompt: 20 Wikipedia
languages crowded out common English chat tokens.

### Compact q4nx scales

A q4nx block (32 rows x 256 inputs) carried a bf16 scale and a bf16 minimum
per row and 32 inputs: 5120 bytes, 5 bits per weight. It now carries a uint8
sub-scale and an int8 sub-minimum per row and 32 inputs and a bf16 d and dmin
per row (w = dmin * mn + d * sc * code): 4736 bytes, 4.625 bits per weight,
7.5% less to stream. Q4_K maps onto it code for code (its 6-bit scales and
minima are the sub-values; d and dmin round to bf16, where before each d * sc
did). The kernel weights each group's dot product and token sum by the
integer sub-values (exact in bf16) and multiplies the two totals by d and
dmin once per 256 inputs, on hi + lo halves. Same binaries otherwise, old and
new interleaved, warm, 2026-10-27.

`gufo-npu bench gemv`, mean µs for one token (two runs):

| Stage | MB before → after | 5120-byte blocks | 4736-byte blocks | Change |
|---|---|---|---|---|
| DeltaNet in (qkv + z) | 10.5 → 9.7 | 280.7, 274.0 | 265.6, 261.2 | −5% |
| Attention in (q + k + v) | 6.6 → 6.1 | 204.3, 204.6 | 201.0, 203.9 | −1% |
| Mixer out (ssm_out) | 2.6 → 2.4 | 123.0, 128.4 | 121.8, 126.1 | −1% |
| FFN in (gate + up) | 15.7 → 14.5 | 372.2, 370.1 | 356.8, 352.9 | −4% |
| FFN down | 7.9 → 7.3 | 222.6, 222.3 | 212.2, 214.0 | −4% |
| LM head | 318.5 → 294.6 | 5,820.9, 5,773.2 | 5,360.3, 5,361.0 | −7.5% |

The LM head stays at 55 GB/s, so its time drops with its bytes; the small
GEMVs keep their fixed per-call cost.

`gufo-npu generate --device npu`, three runs each:

| Prompt | 5120-byte blocks | 4736-byte blocks |
|---|---|---|
| Sky, `--draft none` | 37.15, 36.96, 36.90 tok/s | **38.99, 38.55, 39.08** |
| Fibonacci, `--draft mtp` | 61.06, 60.78, 60.60 | **63.65, 63.00, 63.41** |
| Sky, `--draft mtp` | 52.73, 52.56, 52.83 | 53.39, 53.66, 53.42 |

A no-draft token now takes 25.7 ms instead of 26.9 ms. The rounding of d
instead of d * sc moves the weights slightly, so the greedy replies differ
a little: the sky reply runs to 95 tokens instead of 87, with 61% of its
drafts accepted instead of 65%, which eats most of that prompt's gain.

### fp16 K/V cache and compact scales together

The `3549ab2` build (fp32 K/V cache, 5-bit q4nx) against fp16 K/V plus
compact scales, interleaved, warm, three runs each, 2026-10-27:

| Prompt | Before | After |
|---|---|---|
| Sky, `--draft none` | 36.75, 37.01, 37.04 tok/s | **39.33, 39.29, 39.20** |
| Sky, `--draft mtp` | 52.73, 53.26, 52.99 | **53.95, 53.66, 53.90** |
| Fibonacci, `--draft mtp` | 61.04, 60.86, 61.10 | **63.35, 63.30, 63.23** |
| Chinese sky, `--draft mtp` | 44.90, 44.94, 44.41 | **48.89, 48.83, 48.74** |
| 2,120-token prompt, `--draft mtp` | 46.82, 47.05, 46.48 | **48.47, 48.34, 48.44** |
| 9,050-token prompt, `--draft none` | 28.89, 28.78, 28.82 | **33.22, 33.14, 33.24** |
| 9,050-token prompt, time to first token | 159.7–160.2 s | 140.6–140.8 s |

The compact scales move the weights slightly, so replies differ from the
old build (sky: 95 tokens instead of 87, 61% of drafts accepted instead of
65%; Chinese: 70 instead of 42 tokens, 46% instead of 40%). Every run
printed the same reply as the other runs of its build, except one of the
three 9k runs of the new build: a DMA wait-order bug in the GEMV, since
fixed (QUALITY.md, Determinism).

### NPU power mode

`xrt-smi configure --pmode <mode>` (needs CAP_SYS_ADMIN; the box runs
`default`), same binaries and prompt, three generate runs each, 2026-10-27:

| Mode | Read GB/s | Single call, submit→done | Per call in a runlist | FFN in GEMV | Decode tok/s |
|---|---|---|---|---|---|
| default | 56.0 | 67–70 µs | 25–28 µs | 368 µs | 35.4–35.5 |
| performance | 55.9 | 69 µs | 27 µs | 370 µs | 35.4 |
| turbo | 56.0 | 55 µs | 25 µs | 358 µs | 35.6–35.8 |

Turbo shortens the single-call round trip by ~14 µs and the GEMVs by ~3%
(+0.6% decode). Bandwidth does not move, so the NPU clock is not the limit.
gufo-npu does not change the power mode.

## NPU prefill (Phase 4)

`gufo-npu generate --device npu` now runs every prompt token but the last
through NpuPrefill: a chunk of up to 1,024 tokens goes through each layer at
once, its projections as bfp16 GEMMs on the NPU and its norms, residual
adds, SwiGLU, DeltaNet and attention on the CPU, with two chunks in flight
(one chunk's GEMMs run while the CPU mixes the other). The MTP cache over
the prompt comes from the same GEMMs (eh_proj and the block's input
projection, no LM head). The decoder then runs the last prompt token, which
gives the first logits, and decodes on from the filled state. Prompts under
7 tokens skip prefill (see below).

**Requirements.** `generate --device npu` now also loads
`bfp16_gemm.xclbin` (built and installed with the other kernels) into a
second hardware context, and keeps a bfp16 copy of the projection weights
next to the decoder's q4nx copy: about **1.56 GB more pinned host memory**
(memlock unlimited, as before) and ~2.7 s more at load.

**Numerics.** The GEMMs round their activations to bfp16 inside the mmul (8
values per shared exponent) and write bf16 outputs, where the decode GEMV
rounds its input to bf16 and writes fp32. Against the decode path the
logits after prefill differ by 1.0–2.6e-2 (relative L2) up to 600 tokens and
3.3e-2 at 1–2k; greedy output after prefill agrees with the decode path's
for the first 26 of 48 tokens on the chat prompt (then a near-tie flips).
Details in QUALITY.md.

### Prefill GEMM

IRON's `flm.GEMM` design (`kernels/designs/bfp16_gemm.py`; A bf16, B the
decoder's q4nx weights rounded to bfp16 and resident, C bf16), one xclbin in
its own hardware context, one instruction sequence per (M, K, N).
`bfp16_gemm_test`, best of 50 warm calls, alone on the NPU, 2026-10-27:

| Projection | K → N | M = 256 | M = 512 | M = 1024 |
|---|---|---|---|---|
| DeltaNet in (qkv + z) | 2048 → 8192 | 752 µs, 11.4 TFLOPS | 1146 µs, 15.0 | 1953 µs, 17.6 |
| Attention in (q + gate, k, v) | 2048 → 5120 | 497 µs, 10.8 | 758 µs, 14.2 | 1253 µs, 17.1 |
| Mixer out | 2048 → 2048 | 241 µs, 8.9 | 348 µs, 12.3 | 539 µs, 15.9 |
| FFN gate + up | 2048 → 12288 | 1080 µs, 11.9 | 1689 µs, 15.3 | 2874 µs, 17.9 |
| FFN down | 6144 → 2048 | 585 µs, 11.0 | 1075 µs, 12.0 | 2065 µs, 12.5 |
| MTP eh_proj | 4096 → 2048 | 402 µs, 10.7 | 600 µs, 14.3 | 1002 µs, 17.1 |

- Compute-bound at every size: B is 9 bits per weight (1.56 GB for every
  projection of a pass) and read once per call, ~28 ms at the read ceiling
  against ~170 ms of compute per 1,024 tokens. At M = 1024 the projections
  of a token cost ~170 µs of NPU time (~5,900 tok/s for the GEMMs alone).
- The GEMM's minimum M is 256 (4 core rows x 64), so a short prompt pays for
  256 rows: ~68 ms for the 98 calls of one pass.

### Time to first token

`gufo-npu bench prefill --model Qwen3.5-2B-UD-Q4_K_XL.gguf --decode-path`:
the first N tokens of one long chat prompt ("Item i: the quick brown fox…"),
each length run once to warm up and then three times; TTFT is
SpeculativeGenerate's prompt phase (prefill, the MTP cache, the last token
through the decoder and the first MTP draft), as `generate` runs it. Alone
on the NPU (runs under the shared NPU lock), 2026-10-27:

| Prompt tokens | Decode path (2 tokens per pass) | **NPU prefill** | Prompt tok/s | FLM TTFT (tokens) | FLM tok/s |
|---|---|---|---|---|---|
| 35 | 0.504 s | **0.107 s** (0.107–0.107) | 327 | 0.60 s (35) | 58.6 |
| 476 | – | **0.182 s** (0.181–0.183) | 2,617 | – | – |
| 1,000 | 14.63 s (1,024 tokens) | **0.312 s** (0.311–0.313) | 3,204 | 1.80 s (1,055) | 586.6 |
| 2,048 | ~32.9 s (2,120 tokens) | **0.584 s** (0.573–0.591) | 3,504 | – | – |
| 4,096 | – | **1.237 s** (1.234–1.240) | 3,312 | 3.34 s (4,246) | 1,272 |

- Built on the decode branch's fp16 K/V cache and compact q4nx scales
  (e3605ce). The decode-path column is the same binary with the prompt token
  by token (two per pass) through NpuDecoder; the 2,120-token figure is
  from the draft-vocabulary section above (before the fp16 cache).
- Against FLM: 5.6x faster TTFT on short prompts (MVP target < 0.3 s: met),
  5.8x at ~1k, 2.7x at ~4k; prompt throughput 2.6–5.6x FLM's at the same
  lengths, and at 4k 2.4x FLM's best (1,388 tok/s, at 8.5k tokens).

**Chunk plan.** A pass runs one or two chunks of 256, 512 or 1,024 rows;
`PlanPasses` picks them for the least estimated time from the measured
GEMM throughput per size (11, 14.5, 17 TFLOPS) and ~0.14 ms of CPU work per
token, counting two lanes as overlapped (about max(GEMMs of both, CPU of
both)) and one lane as GEMMs then CPU. The previous rule (the largest size
two lanes fill, else 256) ran four 256-row chunks for 1,000 tokens (the
chunks cover the prompt but its last token):

| Prompt tokens | Previous rule | Planned | TTFT before → after |
|---|---|---|---|
| 476 | 256 + 219 (MTP cache: 256, 219) | same (MTP cache: one 512-row call) | 0.184 → 0.182 s |
| 1,000 | 256 + 256, 256 + 231 | 512 + 487 | 0.339 → 0.312 s |
| 2,048 | 512 + 512, 256 + 256, 256 + 255 | 1,024 + 1,023 | 0.655 → 0.584 s |
| 4,096 | 1,024 + 1,024, 512 + 512, 256 + 256, 256 + 255 | 1,024 + 1,024, 1,024 + 1,023 | 1.321 → 1.237 s |

A single 1,024-row chunk for 1,000 tokens would take the least NPU time
(~0.17 s against ~0.20 s) but loses the overlap with the CPU's ~0.14 s.

**Short prompts.** A prefill pass costs ~0.10 s however short the prompt
(98 GEMM calls of at least 256 rows), the decode path ~28 ms per two
tokens. `bench prefill --decode-path`, 3 warm runs each:

| Prompt tokens | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| Decode path | 0.030 | 0.056 | 0.058 | 0.084 | 0.086 | 0.115 | 0.142 | 0.170 | 0.226 s |
| Prefill | 0.100 | 0.101 | 0.101 | 0.101 | 0.102 | 0.102 | 0.104 | 0.102 | 0.103 s |

So the hook leaves prompts under 7 tokens to the decoder (now: 6 tokens
0.088 s, 7 tokens 0.105 s, 8 tokens 0.102 s). A chat prompt is at least ~10
tokens, so in practice this is the bare-prompt case.

`gufo-npu generate --device npu` end to end, the prompts of the
draft-vocabulary section above, 128 tokens at most; one warm-up run, then
three (the text was identical in all four runs of each prompt):

| Prompt | Prompt tokens | Time to first token | Decode after it |
|---|---|---|---|
| "Explain in a few sentences why the sky is blue." | 24 | 0.11, 0.11, 0.11 s | 53.53, 53.62, 53.50 tok/s |
| 2,120 tokens ("Item i: …", then the sky question) | 2,120 | 0.70, 0.68, 0.69 s (32.9 s on the decode path) | 49.28, 48.85, 49.10 tok/s |

Where the time goes (seconds per run, same runs; GEMM is the time the CPU
waits for the NPU, i.e. GEMM time not hidden behind CPU work):

| Prompt tokens | GEMMs | GEMM wait | Buffer syncs | DeltaNet (CPU) | Attention (CPU) | Other CPU |
|---|---|---|---|---|---|---|
| 35 | 98 | 0.068 | 0.001 | 0.003 | 0.001 | 0.004 |
| 476 | 194 | 0.077 | 0.011 | 0.028 | 0.005 | 0.029 |
| 1,000 | 194 | 0.125 | 0.022 | 0.064 | 0.016 | 0.055 |
| 2,048 | 196 | 0.217 | 0.042 | 0.130 | 0.053 | 0.111 |
| 4,096 | 392 | 0.430 | 0.086 | 0.258 | 0.199 | 0.225 |

- Short prompts are the GEMM pass at M = 256 (68 ms) plus one decode step
  for the last token and an MTP draft (~35 ms).
- At 4k the NPU is busy ~0.70 s of 1.24 s; the CPU side (attention O(n²),
  DeltaNet, and the bf16/fp32 conversions and norms between GEMMs) is the
  larger part. Next levers: NPU attention (IRON `PrefillAttention`) for long
  prompts, fusing the DeltaNet conv into the recurrence jobs, keeping GEMM
  outputs in bf16 for the mixers.

What mattered on the way (warm TTFT, same benchmark):

| Step | 1,024 tokens | 4,096 tokens |
|---|---|---|
| First version: one 256-token chunk at a time, CPU work on a `SpinPool` (the decoder's pool class: its 31 workers fall asleep during each GEMM, and every pool job waits for all of them to wake, ~0.13 s per chunk) | 1.38 s | – |
| A pool that spins while a prefill runs and waits only for its jobs, one worker per physical core (31 spinning workers: 1.41 s, the NPU calls' completions waited for CPUs) | 0.61 s | 3.13 s |
| Chunks of up to 1,024 tokens, two in flight, vectorized CPU side (GEMM wait at 4k 1.14 → 0.49 s), DeltaNet split into (head, 32-column) jobs on the shared state | 0.73 s | 3.61 s |
| The same jobs on cache-line-aligned job-local copies of the state (the shared, 16-byte-aligned rows put two jobs on one cache line: DeltaNet 2.38 → 0.26 s) | 0.36 s | 1.46 s |
| AVX-512 attention kernels, chosen at run time (attention at 4k 0.35 → 0.28 s) | 0.36 s | 1.39 s |
| The decoder's fp16 K/V cache, read by the attention kernels directly (widened in registers): attention at 4k 0.28 → 0.22 s | 0.35 s | 1.32 s |
| Chunk sizes planned by estimated time (above) | **0.31 s** (1,000 tokens) | **1.24 s** |

The fp16 cache is read directly rather than through an fp32 copy: the
kernels converting it in registers (0.216 s at 4k) beat the same kernels
over an fp32 cache (0.276 s, the previous build), so a float view, which
adds the conversion pass on top of that, cannot win.

### TTFT quick wins (2026-10-28)

Two changes from the TTFT research (`docs/research/TTFT.md` on
`research/ttft`), measured with `gufo-npu bench prefill` (each length once to
warm up, then three runs, back to back so the NPU never idles into its
5 s autosuspend), two interleaved rounds of three builds, alone on the NPU:

| Prompt tokens | main | + AVX-512 DeltaNet | + first token from prefill |
|---|---|---|---|
| 35 | 0.107, 0.106 s | 0.107, 0.106 | **0.086, 0.086** (−21 ms against main) |
| 1,024 | 0.317, 0.315 | 0.312, 0.311 | **0.291, 0.290** (−26 ms) |
| 4,096 | 1.235, 1.229 | 1.226, 1.223 | **1.195, 1.199** (−35 ms) |

- **First token from the prefill.** PrefillPrompt now prefills the whole
  prompt; the first token comes from the LM head alone on its last row
  (`NpuDecoder::Logits`, one submission, ~6 ms) instead of a decode step
  for the last prompt token (~29 ms). The research estimated −23 ms; the
  measured gain is 20 ms on short prompts. Quality in QUALITY.md ("First
  token from the prefill").
- **AVX-512 delta rule** in the prefill's DeltaNet (`prefill_mixer.cpp`,
  picked at run time like the attention kernels): bit-identical (same
  hashes), DeltaNet at 4k 0.258 → 0.246 s. The research estimated −0.1 s;
  the 32-column jobs are bound by their two passes over the 16 KB state
  slice more than by instruction count, so the gain is ~1% of TTFT.
- **Short prompts.** The crossover stays at 7 tokens (decode path for
  shorter prompts): with the prefill path forced on, 6 tokens 0.076 s on the
  decode path against 0.082 s with prefill, 7 tokens 0.097 against 0.082.
  Through the prefill, TTFT is now 0.081 s for any prompt up to 12 tokens.

Outputs (`gufo-npu generate --device npu`): the sky prompt without
drafting (`a49f1c4c`) and the 960-token prompt with MTP (`8edbce90`) are
unchanged. The Fibonacci prompt with MTP changes, `69e1c62d` → `fdb595fb`
(239 → 256 generated tokens). Its text is the same for the first 607 of
~700 characters, then a near tie picks "```\n\n**Example usage:**" over
"\n\n\n# Example usage"; the MTP output equals plain greedy from the new
state (`fdb595fb` both), as before the change (`69e1c62d` both). Decode
speed unchanged: sky 39.1 tok/s, Fibonacci 64.4–64.6, 960 tokens 50.9–51.2.

## Prompt cache across requests (2026-10-28)

`PromptCache` (`src/models/qwen35/prompt_cache.hpp`) keeps the decode state
after a prompt prefix, so a multi-turn chat runs only what is new each turn.
DeltaNet state is recurrent and cannot be cut back to an earlier position,
so reuse starts from checkpoints:

- **Where:** before the prompt's last `<|im_start|>`, the history through
  the last user message. The GGUF's chat template renders earlier
  assistant turns as `<|im_start|>assistant\n{content}`, without the think
  block the generation prompt opens (`<think>\n\n</think>\n\n`), so the
  next request matches the previous prompt only up to there.
- **How:** the prefill copies the DeltaNet layers' conv window and state
  as its pass goes past that position (`StateCapture`); the attention and
  MTP caches are cut to it afterwards. A checkpoint costs a copy, not a
  pass. Remainders below the prefill threshold run through the decoder.
- **Kept:** two checkpoints, least recently used out first (configurable,
  with an optional byte budget).

`qwen35_prompt_cache_test` (labels `qwen35_2b;npu;model`), warm (each conversation's second round),
2026-10-28. A conversation over a document of numbered notes, three turns of
24 greedy MTP tokens. Time to the first token (`SpeculativeStats`), per
turn, three ways: the model's own `PrefillPrompt` (no cache), through the
cache with nothing to restore (a miss), and through the cache (turn 1 is a
miss).

| History | Turn | Prompt tokens | Restored | No cache | Cache miss | Cache hit |
|---|---|---|---|---|---|---|
| ~1k | 1 | 1,124 | – | 0.385 s | 0.381 s | **0.382 s** |
| ~1k | 2 | 1,157 | 1,117 | 0.384 s | 0.386 s | **0.100 s** |
| ~1k | 3 | 1,190 | 1,150 | 0.389 s | 0.393 s | **0.096 s** |
| ~4k | 1 | 4,091 | – | 1.199 s | 1.193 s | **1.223 s** |
| ~4k | 2 | 4,124 | 4,084 | 1.282 s | 1.296 s | **0.130 s** |
| ~4k | 3 | 4,157 | 4,117 | 1.283 s | 1.303 s | **0.122 s** |

- **Turns 2 and 3** restore the previous turn's checkpoint and run ~40
  tokens (the previous reply, the next question and the generation
  prompt): time to the first token falls 4× at ~1k tokens of
  history and 10× at ~4k, and stays flat as the history grows.
- **A miss** costs the checkpoint copy on top of the plain prefill:
  0–20 ms.
- **Memory per checkpoint:** 34.5 MiB at 1,124 tokens, 75.1 MiB
  at 4,091 and 131.6 MiB at 8,221: 18 DeltaNet layers' states (~19 MiB) plus 14 KiB per token (6 attention layers' fp16 K/V, 12 KiB, and the MTP layer's, 2 KiB). Two checkpoints of
  8k-token conversations hold ~263 MiB of host memory.
- Two separate runs of the test agreed within 0.03 s.
