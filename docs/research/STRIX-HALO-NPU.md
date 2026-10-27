# Strix Halo NPU (XDNA2): facts that shape gufo-npu's kernels

Research from 2026-10-27, mostly from source: IRON@b3ba3cf, mlir-aie@v1.4.4,
xdna-driver main (2026-10-26), plus the header of FLM's Qwen3.6 pack (read by
HTTP range requests). "Inferred" marks conclusions not read directly.

## Memory path and buffers

- Host BOs are `AMDXDNA_BO_SHARE`: shmem or pinned user pages
  (`drivers/accel/amdxdna/amdxdna_gem.c`, `amdxdna_ubuf.c`). With PASID/SVA
  the NPU uses process virtual addresses through the IOMMU; otherwise each BO
  gets a contiguous IOVA (`amdxdna_iommu.c`, `force_iova` parameter). No cap
  on shared BOs found beyond RAM, IOVA space and memlock.
- Inferred: in PASID mode, 2 MB transparent hugepages on weight buffers
  should cut IOTLB misses. Worth an A/B test (THP vs not, `force_iova=1`).
- **Layout decides bandwidth.** IRON's `flm.GEMM` pre-packs weights so each
  128 KB transfer is one contiguous run, using all 8 columns' shim DMAs and
  memtile staging: ~47 GB/s. The same data through strided descriptors
  (16 B inner runs) drops to ~10 GB/s (`iron/operators/flm/gemm/README.md`).
- Shim descriptors: 3 dimensions + repeat; size field in 4-byte units up to
  1023 (`iron/operators/flm/dequant/README.md`).
- Inferred: FLM decodes Qwen3.6 at the same speed on Kraken Point as on
  Strix Halo despite half the DRAM bandwidth, so the limit is NPU-side.
- Our own copy test reaches ~80 GB/s read + write combined
  (`docs/models/qwen3.6-35b-a3b/BENCHMARKS.md`).

## Clocks and power

- DPM table {npuclk, hclk} MHz: 396/792 … 1267/1800 (`npu4_regs.c`).
- `xrt-smi configure --pmode`: turbo = max level, no clock gating;
  performance = max level; balanced = half; powersaver = level 0; default =
  the context solver decides (`aie2_pm.c`). Raising the level takes 3.5 s.
- A hardware context created without QoS hints gets the max DPM level; the
  device runs at the highest level any open context asks for
  (`aie2_solver.c`). Keep one long-lived context.
- Runtime PM `auto` explains our 0.3–1.1 ms first-dispatch-after-idle; set
  `power/control` to `on`.

## Dispatch and data-dependent MoE

- At most 4 commands in flight per hardware context (`HWCTX_MAX_CMDS`); a
  chained submit fits in 4 KB (inferred ~30–60 commands per chain);
  `hwctx_limit` is 16 (`aie2_pci.h`, `aie2_ctx.c`, `aie2_message.c`).
- Runtime parameters cost ~66 ns per word per core (~2 µs per word over 32
  cores): pass few.
- IRON's `flm.DecodeLayer` runs a whole Gemma-4 decoder layer as one runtime
  sequence on all 32 tiles. That is the template for fusing.
- mlir-aie `DispatchTime` scalars and dynamic BD chains are built on the host
  (`aiecc --get-npu-cpp` emits a C++ sequence builder). Choosing 8 of 256
  experts works with one xclbin and 8 offset scalars, but it needs the host
  round trip router → top-8 → dispatch. No device-side way to drive shim BD
  addresses was found.

## Compute

- 512 int8 MACs per tile per cycle on XDNA2. The AIE2P matmul multiplies
  bfp16 (bfloat16 × bfp16ebs8, 8×8×8); bf16 B is converted inside each MAC
  anyway, so storing B as bfp16 costs no accuracy and saves bandwidth. Set
  rounding to `conv_even` (the power-up default rounds down, far less
  accurate).
- int4 unpack: `aievec.unpack` → `aie2p.unpack.I512.I8.I4` (64 or 128 nibbles
  to int8).
- Open int4 path: `aie_kernels/quant/q4nx_dequant.cc` (q4nx → bfp16) feeding
  `flm.GEMM` reaches ~2.1 TFLOPS, 1.3× FLM's shipped overlay.

## Reusable IRON FLM operators

`GEMM` + `DequantBFP` (prefill projections and grouped experts),
`PrefillAttention` (head dim 256 GQA), `LMHead` (fused RMS norm),
`DecodeLayer` (fusion template). IRON's Gemma q4nx uses 32×256 blocks with
bf16 scale and minimum per 32 values (5 bits/weight), not FLM's Qwen layout.
**No MoE/router/top-k and no Gated DeltaNet / SSM kernels exist** in IRON or
mlir-aie: both are new work.

## FLM's Qwen3.6 pack (`FastFlowLM/Qwen3.6-35B-A3B-NPU2@flm_q4k_high_precision`)

- `model.q4nx` (21.0 GB) is safetensors-framed: 733 tensors (371 packed
  "I8", 302 BF16, 60 F32). Embeddings, router, DeltaNet alpha/beta, conv1d and
  norms are BF16.
- Quantized tensors are `[N/32, K/256, 4736]` blocks. Inferred block layout:
  4096 B nibbles, 512 B of 8-bit sub-scales/mins, 32 bf16 `d`, 32 bf16 `dmin`
  (~4.6 bits/weight, Q4_K-like; to be checked against llama.cpp Q4_K).
- Experts are expert-major and contiguous: expert *e* of a projection starts
  at `e × 606208` bytes, so one linear DMA per expert.
- `lm_head` is 8-bit (540 MB).
- **No MTP tensors**, so FLM does not speculate.

## Open questions

- Read-only NPU bandwidth (decode's pattern) on Strix Halo.
- NPU vs iGPU memory contention when both run.
- What sets the ~20 µs per-command firmware floor.
