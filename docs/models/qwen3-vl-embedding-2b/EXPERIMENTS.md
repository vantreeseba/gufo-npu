# Qwen3-VL-Embedding-2B: experiments

## Retained

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
- Attention on the NPU, for long inputs.
