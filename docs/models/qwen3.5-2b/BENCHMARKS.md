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
