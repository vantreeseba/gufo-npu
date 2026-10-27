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
