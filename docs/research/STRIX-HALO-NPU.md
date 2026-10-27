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
- Measured (2026-10-27, `docs/models/qwen3.5-2b/BENCHMARKS.md`): reads alone
  stop at ~56 GB/s. They scale ~14 GB/s per column to 4 columns, then
  flatten; a second shim channel per column changes nothing. A 16 MiB buffer
  (fits the 32 MB Infinity Cache/MALL) reads no faster than 256 MiB.
- The MALL is filled only by GPU requests (Chips and Cheese,
  https://chipsandcheese.com/p/evaluating-the-infinity-cache-in and
  https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling); no XRT
  BO flag routes NPU traffic through it. A GPU-warmed buffer read by the NPU
  is untested, but the ceiling looks like the NPU's port into the fabric,
  which cache hits would not widen.
- Shim BD knobs in mlir-aie v1.4.4: `burst_length` (npu2: 64/128/256/512 B;
  0 = largest, the default) and `axcache` (default 2, "upsizing in the
  NoC"). Neither is expected to move the read ceiling; AxCACHE is untested.
- DMA zero-compression exists on core and memtile DMAs, not the shim, and
  4-bit weights have few zero bytes: no use for weights.

## System RAM vs the GPU carve-out

- The test box has 128 GB of LPDDR5X; the BIOS reserves 64 GiB as GPU
  "VRAM" (`mem_info_vram_total`) and Linux sees ~61 GiB. Both halves are the
  same DRAM behind the same memory controllers (~256 GB/s peak).
- Everything gufo-npu loads for the NPU lives in system RAM: the GGUF is
  mmapped by the host, and NPU buffers are amdxdna BOs (`xrt::ext::bo`,
  shmem or pinned user pages), never amdgpu VRAM. XRT cannot allocate from
  the GPU carve-out.
- Which half a buffer sits in does not change its speed: same DRAM, and the
  NPU's ~56 GB/s read limit is on its own side of the fabric. What matters
  when the GPU and NPU run at once is total DRAM traffic: the NPU's 56 GB/s
  plus whatever the GPU reads share ~256 GB/s. A GPU job near the DRAM limit
  would slow NPU decode; untested.
- Footprint: the carve-out leaves ~61 GiB for the OS, the GGUF page cache
  and the NPU's pinned weights (~1.2 GB for Qwen3.5-2B, ~20 GB for the 35B).
  Shrinking the carve-out in the BIOS frees system RAM if the GPU is not
  needed.

## On-chip memory

- Core tiles: 64 KB data memory in 8 banks (allocator models 4); a core can
  also address its west, north and south neighbours (256 KB reach), not
  memtiles. An ObjectFifo between adjacent cores is shared memory, no DMA.
- Memtiles: 512 KB each, 16 banks; their DMAs reach the east and west
  neighbours' memory and locks.
- Within one loaded design, runtime sequences only rewrite BDs, RTPs and
  addresses; core and memtile memory should persist across calls of the same
  hardware context (not documented; needs a canary test across runs and with
  a second context active). Reloading a design (`load_pdi`) costs milliseconds
  (4.9 ms measured in arXiv 2512.13282) and resets tiles.
- Each core column's 4 cores use all 4 southbound stream channels between
  the core rows and the memtile for their outputs; an extra core-to-shim
  stream does not route (we send the GEMV's residual down the existing
  output streams instead).

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
- Measured (2026-10-27, `bfp16_gemm_test`): `flm.GEMM` alone, B already
  bfp16 in DDR, at the 2B's projections runs 9–12 TFLOPS at M = 256 and
  16–18 TFLOPS at M = 1024 (12.6 for K = 6144), compute-bound: B at 9 bits
  per weight is read once per call. The mmul also converts A to bfp16 (8
  values per shared exponent), which costs ~7e-3 relative error per GEMM
  (9e-3 with B's own rounding); a bf16 A would need a second (lo) pass.
  A first timing through IRON's Python runtime came out 3–10× slower and
  bimodal while another process used the NPU; time designs through our own
  runtime, alone on the NPU.

## Reusable IRON FLM operators

`GEMM` + `DequantBFP` (prefill projections and grouped experts),
`PrefillAttention` (head dim 256 GQA), `LMHead` (fused RMS norm),
`DecodeLayer` (fusion template). IRON's Gemma q4nx uses 32×256 blocks with
bf16 scale and minimum per 32 values (5 bits/weight), not FLM's Qwen layout.
gufo-npu's q4nx keeps the 32×256 blocks with 8-bit sub-scales and minima and
a bf16 d/dmin per row (4.625 bits/weight; `src/core/q4nx.hpp`).
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
