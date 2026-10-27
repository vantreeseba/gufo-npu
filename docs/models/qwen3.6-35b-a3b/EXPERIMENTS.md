# Qwen3.6-35B-A3B experiments

Short retained/rejected decisions; the runs behind them are in
[BENCHMARKS.md](BENCHMARKS.md), [QUALITY.md](QUALITY.md) and
[the MoE spike](../../research/NPU-MOE.md).

| Experiment | Decision / qualification |
| --- | --- |
| Expert selection by gathers with patched DDR offsets | Retained: two calls per layer (gate + up over 9 experts, then down with K = 9 × 512), their offsets and router scales rewritten in the instruction stream after CPU routing: 414 µs per layer in a 40-layer chain, against 948 µs for one run per expert matrix. |
| Every matrix in q4nx, LM head included | Retained: the same 4.625-bit class as FastFlowLM's pack (which keeps only its LM head at 8 bits). Top-1 against the GGUF 88.0% (non-expert matrices alone 92.0%); the NPU path matches the CPU over the same weights (263/301 against llama.cpp, CPU 260/301). Decode 19.8–20.3 tok/s against FLM's 16.9–17.1. |
| Non-expert matrices at 8 bits on the NPU | Not built: 2.75 GB per token as stored puts the NPU's ~56 GB/s ceiling near 15–18 tok/s, about FLM's speed. |
| CPU and NPU splitting each matrix by rows | Not pursued: together they read ~170 GB/s against ~120 for the CPU alone, but each split point costs a sync; the NPU-only path was chosen. |
