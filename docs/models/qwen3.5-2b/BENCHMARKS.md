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

## NPU q4nx GEMV (Phase 3, step 1)

`gufo-npu bench gemv --model Qwen3.5-2B-UD-Q4_K_XL.gguf`: one real tensor per
decode shape, packed to q4nx, resident on the device; random bf16 token.
20 warm-up dispatches, then the mean over 200 (start + wait, no host copies).
Three runs agree within 1%. Measured 2026-10-27.

| Tensor | GGUF type | dim → rows | MB (q4nx) | Mean µs | GB/s | Err NPU | Err total |
|---|---|---|---|---|---|---|---|
| blk.0.ssm_out | Q8_0 | 2048 → 2048 | 2.6 | 108 | 24.4 | 0.56% | 8.2% |
| blk.0.ffn_gate | Q4_K | 2048 → 6144 | 7.9 | 202 | 38.8 | 0.48% | 0.70% |
| blk.0.ffn_down | Q6_K | 6144 → 2048 | 7.9 | 202 | 38.8 | 0.41% | 7.6% |
| token_embd (LM head, RMS norm fused) | Q6_K | 2048 → 248320 | 318.5 | 5,690 | 56.0 | 0.51% | 9.2% |

- "Err NPU" is the relative L2 error of the NPU output against a
  double-precision GEMV over the same packed weights: the kernel's own error.
  "Err total" compares against the GGUF weights, so it adds requantization.
- **Streaming runs at the DMA ceiling.** The LM head reads 318 MB at 56 GB/s.
- **Small matrices pay the dispatch.** A dispatch costs ~66 µs on its own
  (`bench dispatch`); 7.9 MB at 56 GB/s is ~140 µs, which with that overhead
  matches the 202 µs measured. Chaining dispatches in a runlist (~20 µs
  each) and fusing projections that share an input is the next step.
- **Q4_K converts exactly** (codes kept, scales and minima rounded to bf16):
  0.7% total error. Tensors that Unsloth kept at 5–8 bits lose ~8% when
  requantized to 4 bits with a min/max fit. Whether that costs model quality
  is measured with the teacher-forced argmax match once decode runs end to
  end; an 8-bit variant of the kernel is the fallback.
- The kernel rounds to nearest even. A core starts in floor (truncating)
  mode, which biased the sum to ~1.3% error at dim 2048.
- Per-token ceiling for an all-q4nx 2B (~1.2 GB at 5 bits per weight) at
  56 GB/s: ~21 ms, ~47 tok/s before dispatch and non-GEMV costs.
