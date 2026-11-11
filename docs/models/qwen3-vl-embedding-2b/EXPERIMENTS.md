# Qwen3-VL-Embedding-2B: experiments

## Retained

- **The vision tower's linears on the NPU GEMM** (2026-11-11, VISION.md
  stage 2). Six (K, N) shapes added to the model's xclbin; the patch
  embedding (K = 768) is padded with zeros to K = 1024 and reuses the
  (1024, 1024) sequence rather than adding a K = 768 design. `NpuVision`
  shares the embedder's hardware context and its two-in-flight chunked
  pass (`npu/gemm_pass.hpp`). A page (7,200 patches) spends 0.15 s waiting
  on the NPU's 792 GEMM calls against 3.0 s of CPU attention.
- **The ViT's attention on the text attention's AVX-512 kernels**
  (2026-11-11): the kernels take a head layout (query heads, kv heads,
  stride, causal or full) instead of the text config, so the same
  transposed key tiles and register blocking serve both. Query blocks are
  64 rows for the ViT (16 for text, unchanged), since one image is one long
  segment. The fp32 reference's per-key loop is not a runtime path; this is
  the interim until NPU attention (stage 3).

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

## Rejected

- **The ViT's attention on IRON's MHA** (2026-11-11, spike for VISION.md
  stage 3; not merged). IRON's `mha` design (pinned commit) with mlir-aie
  v1.4.4's `mha.cc`: 16 heads of 64, 8 pipelines of 3 cores (QK, softmax,
  PV), q/k/v repacked per head on the CPU. What it took to run at all:
  `mha.cc` is causal only (the QK and PV skips and the diagonal mask had to
  be compiled out); K and V padded to 64 x 8 pipelines leave key blocks of
  pure padding, which skip the softmax's state update while `matmul_PV`
  still rescales O by the previous block's factor (24 patches gave an
  all-zero output), so K and V were padded to one key block instead; the
  pinned design binds `matmul_PV` without v1.4.4's last argument
  (`S_kv_eff`), harmless here. The kernel's exp2 scale uses log2(e) =
  1.4453125, corrected through q.

  | | NPU (IRON MHA) | CPU (AVX-512, `FullAttention`) |
  |---|---|---|
  | One layer, 7,200 patches | 58 ms best, 60 ms mean (3.7 TFLOP/s) | 120 ms (1.8 TFLOP/s) |
  | One layer, 1,200 patches | 2.3 ms | 3.7 ms |
  | Random q, k, v: relative error vs a double softmax | 2.9% (24), 4.2% (1,200), 4.7% (7,200) | 1e-6 |
  | Page, end to end | 2.57 s (attention 1.40 s) | 4.2 s (attention 2.97 s) |
  | 640x480 card: embedding vs the CPU reference | **0.9817** (fails 0.99) | 0.9964 |
  | 640x480 card: features / DeepStack 0 vs the reference | 0.742 / 0.956 | 0.990 / 0.9998 |
  | Oracle cards (24 to 48 patches) vs HF | 0.9952 to 0.9968 | 0.9946 to 0.9967 |

  Fast enough (2x the CPU, a page 1.6 s sooner) but not accurate enough
  past small images: the error compounds through the 24 blocks. Likely
  sources, untested: O accumulated in bf16 across key blocks (`matmul_PV`
  writes the bf16 tile it rescales), bf16 scores and weights, and
  `aie::exp2<bfloat16>`'s linear interpolation (up to 7% high per weight).
  A ViT kernel needs an fp32 O accumulator and a closer exp at least; that
  is a new kernel, not a configuration of this one.

## Candidates

- **Attention on the NPU for the ViT** (stage 3): 3.0 s of a page's 4.3 s.
  IRON's MHA is too inaccurate as is (Rejected, above); a kernel with an
  fp32 O accumulator and a closer exp is the next try.
- **Find which bfp16 product makes the ViT's few drifting tokens**
  (QUALITY.md, images on the NPU) before fusing more into the GEMMs.

- Skip the last layer's output projection and FFN for every row but each
  input's last (~2.5% of the GEMM work).
- Overlap the end of one projection with the start of the next (today the
  NPU idles while the CPU reads the last chunk of one pass and writes the
  first of the next; this matters for a single short input).
- Attention on the NPU, for long inputs: still 40% of the time at 2,048
  tokens after the AVX-512 kernel; the NPU-first rule applies once it is
  measured faster.
