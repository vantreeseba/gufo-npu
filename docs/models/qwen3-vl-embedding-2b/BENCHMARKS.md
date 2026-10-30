# Qwen3-VL-Embedding-2B: benchmarks

`gufo-npu bench embed --model GGUF --tokens 128,512,2048 --batch 8 --runs 5`:
warm (2 warm-up calls per length), unique inputs per call, tokens per second
over the timed calls and the share of time in GEMM waits, buffer syncs, CPU
attention and the rest.

No measurements yet. Expected from the 2B's measured GEMM rates (11-18
TFLOPS at these shapes) and 2.8 GFLOP per token: at most ~4-6k tokens/s from
the GEMMs alone, less with the CPU's share.

FastFlowLM has no counterpart for this model, so comparisons are against the
CPU reference and earlier commits.
