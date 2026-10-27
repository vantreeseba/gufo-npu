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
| Instruction-buffer flow for the 2B too | Rejected: it costs Qwen3.5-2B ~2% decode (39.1 → 38.2 tok/s, same output); only MoE models, whose gathers need patched instruction streams, use it. |
| MTP speculation with two-token passes over the expert union | Retained: a pair reads the dense weights once and the union of both tokens' experts (8 to 16 routed plus the shared one) once; each token weighs the other's experts 0 and experts sum in id order, so a two-token pass is bit-identical to two one-token passes and speculative output equals plain greedy. 26.0–28.9 tok/s at short and ~1k context with 75–83% of drafts kept, against 18.7–20.3 without and FLM's 16.9–17.1. The prompt runs two tokens per pass: 942 tokens in 31 s instead of 48 s. |
| Padding every pair's union to 17 experts (one gather variant) | Rejected: each expert costs ~37 µs per layer (480 µs at 9, 779 µs at 17), 4.5–7.6 ms per step at typical unions of 12–14; a variant per count (9 to 17) costs only instruction streams. |
| Down gather over 17 experts: `MAX_DIM` 8,704 or two calls | `MAX_DIM` raised from 6,144 to 8,704: no runtime cost (Qwen3.5-2B same outputs and speed); two calls would need an accumulating output stage and a further call per layer. |
