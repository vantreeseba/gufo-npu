# Qwen3-VL-Embedding-2B: experiments

## Retained

- **AVX-512 attention over transposed key tiles** (2026-10-30). Each key
  block is written once as a 128 x 64 tile (keys along the vector lanes),
  so scores are a register-blocked product (4 query vectors x 64 keys, no
  horizontal sums), with online softmax and P V accumulated 4 query vectors
  at a time; work items sorted longest first. Throughput at 8 inputs per
  call: 512 tokens 2,789 -> 4,024 tok/s, 2,048 tokens 1,297 -> 3,037;
  attention's share 42% -> 17% and 72% -> 40%. Accuracy unchanged
  (BENCHMARKS.md, QUALITY.md). The plain loop stays only as the fallback
  for CPUs without AVX-512 (hosted CI).
- **Layer-at-a-time over a packed batch** (2026-10-29). Inputs are packed
  back to back and each projection runs over the whole batch in chunks of
  up to 1,024 rows, two GEMM calls in flight. Attention keeps inputs apart,
  so an input's embedding does not depend on its batch (the NPU test checks
  alone = batched). No K/V cache: an input longer than a chunk is just more
  rows of the same batch.
- **Weights rounded to bfp16 once at load** from the BF16 GGUF; no q4nx copy
  (there is no decode, so nothing reads q4nx).

## Candidates

- Skip the last layer's output projection and FFN for every row but each
  input's last (~2.5% of the GEMM work).
- Overlap the end of one projection with the start of the next (today the
  NPU idles while the CPU reads the last chunk of one pass and writes the
  first of the next; this matters for a single short input).
- Attention on the NPU, for long inputs: still 40% of the time at 2,048
  tokens after the AVX-512 kernel; the NPU-first rule applies once it is
  measured faster.
