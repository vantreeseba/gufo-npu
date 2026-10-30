# Qwen3-VL-Embedding-2B: benchmarks

`gufo-npu bench embed --model GGUF --tokens 128,512,2048 --batch 8 --runs 5`:
warm (2 warm-up calls per length), unique inputs per call, tokens per second
over the timed calls and the share of time in GEMM waits, buffer syncs, CPU
attention and the rest.

## 2026-10-30, AVX-512 attention, image `sha-86050a6`, F16 GGUF

Same settings as below (warm, 2 warm-up and 5 timed calls per length, 8
inputs per call). Two sweeps agree within 1%; the table is sweep 1. The box
was at load ~9 from other services when the run started (the run below was
near idle).

| Tokens per input | tok/s | before | GEMM wait | sync | CPU attention | other |
|---|---|---|---|---|---|---|
| 32 | 2,605 | 2,677 | 69% | 7% | 7% | 17% |
| 128 | 3,544 | 3,319 | 65% | 8% | 7% | 20% |
| 512 | **4,024** | 2,789 | 46% | 9% | 17% (was 42%) | 28% |
| 2,048 | **3,037** | 1,297 | 31% | 7% | 40% (was 72%) | 22% |

+44% at 512 tokens and 2.3x at 2,048; 32 tokens within noise of before.
Accuracy unchanged: NPU min cosine to HF 0.99808 (was 0.99794), CPU
reference 1.000000, rankings the same.

## 2026-10-30, image `sha-62acdfc`, F16 GGUF

Warm (2 warm-up calls per length, 5 timed), unique inputs per call; the
box's chat server was loaded but idle. Two sweeps agree within 2%; the
table is sweep 1.

| Tokens per input | Inputs per call | tok/s | GEMM wait | sync | CPU attention | other |
|---|---|---|---|---|---|---|
| 32 | 8 | 2,677 | 71% | 7% | 5% | 17% |
| 128 | 8 | 3,319 | 60% | 7% | 12% | 20% |
| 512 | 8 | 2,789 | 31% | 6% | 42% | 21% |
| 2,048 | 8 | 1,297 | 15% | 3% | 72% | 10% |
| 32 | 1 | 443 (~72 ms per input) | 92% | 1% | 1% | 5% |
| 128 | 1 | 1,484 | 78% | 4% | 7% | 11% |
| 512 | 1 | 2,306 | 50% | 6% | 28% | 15% |

- Peak ~3.3k tok/s at 128-token inputs, batched.
- From 512 tokens the CPU attention dominates (42%, then 72% at 2,048):
  it is a plain per-key loop (ops.cpp), not the prefills' blocked
  AVX-512 kernels. That is the next thing to speed up (EXPERIMENTS.md).
- A single short input is GEMM-bound on mostly empty 256-row chunks:
  ~72 ms for 32 tokens.
- The CPU reference (fp32, for tests) runs ~125 tok/s.

FastFlowLM has no counterpart for this model, so comparisons are against the
CPU reference and earlier commits.
