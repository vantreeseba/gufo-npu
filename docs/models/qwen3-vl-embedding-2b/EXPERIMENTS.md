# Qwen3-VL-Embedding-2B: experiments

## Retained

- **The ViT's attention on its own NPU program** (2026-11-11, VISION.md
  stage 3; `kernels/vit_attention.py`, core program
  `kernels/aie/vit_attention.cc`, host side `npu/vit_attention.hpp`).
  - Each core runs flash attention for a 64-row query block of one head
    over the keys in blocks of 32, with the softmax state (row max, row
    sums) and O in fp32 and the output rounded to bf16 once.
  - Both products run on the bfp16 mmul as 3-term splits: x = hi + lo
    with hi = bfp16(x) and lo = bfp16(x - hi), A B ~ Ah Bh + Al Bh + Ah Bl.
    The host splits q, k and v once per block (AVX-512, in the order the
    cores read them); the cores split P.
  - The running max is an integer, round(row max + 1), so the rescale
    2^(m_old - m) is exact; row sums come from the mmul with a tile of
    ones; padding keys get a -inf bias.
  - exp2 is a degree-3 polynomial with bf16 limbs where they matter (2.2e-4
    max and 6.8e-5 rms relative error against float64, against 9.2e-6 for
    the full-precision version), after x is clamped at -64 by an unsigned
    min on its bits.
  - Column c computes heads 2c and 2c + 1. Its memtile splits Q to the 4
    cores, broadcasts K and V (one object per key block, a head's blocks
    sent once per group of 4 query blocks through a zero-stride shim
    pattern) and joins the outputs. A call is 1 Q, 2 KV and 1 O task per
    column; one sequence per padded length (256 to 7,424, 29 sequences,
    about 2 minutes of build, as the GEMM's 30).

  The output's error equals rounding it to bf16 at every length (1.65e-3
  to 1.68e-3 against float64, 24 to 7,200 patches; QUALITY.md) and the
  embeddings did not move. A layer takes 0.53 ms at 24 patches, 2.7 ms at
  1,200 and 61 ms at 7,200 (CPU: 120 ms), but every block now moves the NPU
  between the GEMM's and the attention's hardware contexts twice (~3 ms a
  switch). A page's vision tower went from 3.57 s to 2.42 s; below ~3,000
  patches an image is slower (BENCHMARKS.md). The NPU path is the default
  at every size anyway: the work belongs on the NPU, and the switches are
  the next thing to remove (Candidates).

  On one core (µs per key block of 32 keys x 64 queries, converted from
  the probe's block sizes):

  | Step | µs | What changed |
  |---|---|---|
  | v1 | 21.2 | 32 x 16 blocks, splits on the core, fp32 state; error 1.6e-3 from the first build |
  | v2 | 11.4 | 64-row blocks, four interleaved accumulator chains, two exps in lockstep |
  | v3 | 5.5 | q, k, v split on the host (on the core the split is redone for every query block), two input streams instead of three, row patterns by zips, the light exp, QK fully unrolled |
  | v4 | 4.6 | 32-key blocks, P split in place and read back as a stream in PV, f = x - k by subtraction |

  Peano's assembly predicted the times: bundles per block divided by the
  NPU clock (1,267 MHz, docs/research/STRIX-HALO-NPU.md) came within 5 to
  15% of the measured time, so changes were judged on bundle and spill
  counts per basic block before building.
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
  segment. The fp32 reference's per-key loop is not a runtime path. Since
  stage 3 the ViT's attention runs on the NPU; `FullAttention` stays as the
  NPU op test's CPU comparison.

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

- **Stage 3 variants of the attention kernel** (2026-11-11, one core or
  Peano's assembly):
  - The full-precision exp (degree 4, every limb): 3.0 to 3.15 µs per
    16-key block against 2.76 for the light one, with the same output
    error (1.660e-3 against 1.662e-3).
  - Unrolling the exp loop: Peano scheduled the copies one after the other
    (twice the bundles). Four exp chains in lockstep instead of two
    spilled (182 bundles per 128 lanes against 160).
  - The softmax tail (row max, m, S - m) as its own pass from memory:
    2,504 bundles per block against 1,624 inline.
  - Row sums added into l in memory per tile: 116 bundles per tile against
    92 with an accumulator.
  - Peano's software pipeliner (`clang loop min_iteration_count`,
    `pipeline(disable)` for the post-RA pipeliner, a larger
    `--aie-postpipeliner-maxii`): no change, or 24% worse.
  - QK fully unrolled in the same loop as O's rescale branch: Peano ran for
    two hours without finishing; the branch moved to another pass.
  - A smaller core program, to switch contexts faster: from two
    measurements a switch costs about 0.35 ms plus 10.8 µs per KB of the
    two designs' configuration (docs/research/STRIX-HALO-NPU.md; the
    attention's CDO is 361 KB, 307 KB of it the 32 core programs, the
    GEMM's 183 KB). Re-rolling loops to halve the program would save about
    40 ms per image by that estimate and cost a page more than that. Not
    built.
  - CPU attention below ~3,000 patches: faster today (vision 0.03 s
    against 0.16 to 0.20 s at 24 patches, all of it the switches), but the
    work stays on the NPU and the switches are what to remove.
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

- **One hardware context for the vision tower**: two context switches
  per block (~3 ms each) are ~0.15 s of every image and all of a small
  image's slowdown against CPU attention. The GEMM could become a mode of
  the attention program (the same channel shape: a per-core input, a
  column broadcast and a joined output), or the attention could join the
  GEMM's xclbin.
- **The ViT's q/k/v split in the qkv GEMM's output pass** instead of its
  own pass: 2.7 to 4.5 ms per block at 7,200 patches (~0.1 s a page).
- **Skip key blocks that are all padding**: lengths are padded to 256, so
  24 patches run 256 keys and the cap runs 7 blocks of padding.
- **The attention's exp is latency-bound**: 92 bundles per 64 lanes for
  ~25 mac-unit operations; the core's five 64-lane accumulator registers
  allow only two chains.
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
