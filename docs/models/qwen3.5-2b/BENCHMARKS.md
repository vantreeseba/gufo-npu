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

The copy's 79 GB/s is ~40 GB/s each way at once. Reads alone stop at ~56 GB/s
however many channels issue them, so **56 GB/s is the decode ceiling**:
decode only reads weights.

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
stops after 64): every matrix product on the NPU (97 GEMVs per token), the
rest in the fp32 decoder on the CPU. Three runs in a row, 2026-10-27:

| Path | Decode tok/s | ms per token | TTFT (19 tokens) |
|---|---|---|---|
| CPU reference (GGUF weights) | 5.55 | 180 | 1.83 s |
| NPU projections + CPU reference math | 22.4 | 44.8 | 0.85 s |
| **NPU projections + fp32 decoder** | **32.5** (32.45, 32.52, 32.69) | **30.7** | 0.59 s |
| FLM (Lemonade), 35-token prompt | 28.5 | 35.1 | 0.60 s |

- FLM's number comes from its server (`tools/bench/openai_bench.py`, 256
  generated tokens); ours from the CLI with a 64-token reply. Both are warm
  and short-context; a like-for-like run waits for `gufo-npu serve`
  (Phase 5).
- Of the 30.7 ms: ~28.4 ms in GEMV calls (~21.4 ms streaming 1.2 GB at
  56 GB/s, ~7 ms fixed cost: ~62 µs per call from submit to completion,
  measured on the smoke kernel; polling instead of waiting saves ~3 µs,
  buffer syncs ~0.3 µs), ~2.3 ms in the decoder's CPU math. The LM head alone
  is 5.7 ms.
- A runlist dispatch costs ~20 µs, so chaining a layer's GEMVs (which needs
  the norms, residuals, SwiGLU, DeltaNet and attention on the NPU) is worth
  up to ~4 ms per token. The streaming floor is ~21.4 ms (~47 tok/s).
- Prompts run token by token through the decode path (prefill is Phase 4).
