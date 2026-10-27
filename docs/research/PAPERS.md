# XDNA / XDNA2 literature notes

What other work on AMD NPUs offers gufo-npu, as of 2026-10-27. Summaries of
reading done for this project; check the sources before relying on a number.

## Directly useful

- **mlir-air `programming_examples/fused_decode`, `llms/`, `llms/dflash`**
  (AMD, https://github.com/Xilinx/mlir-air). One dispatch per decoded token
  (all layers, attention with KV append, LM head) via a "full ELF" with
  per-token scratchpad parameters (needs XRT >= 2026-05-19 / xdna-driver
  1.7). A KV-window "staircase" stops DMAing padded cache. LFM2 conv layers
  keep their recurrent state inside the fused dispatch: a template for
  DeltaNet. DFlash speculative decoding: an 8-position target pass unpacks
  each 4-bit block once into bf16 and reuses it (`aie::mmul<8,8,8>`), one
  pass costs ~1.7x a single-token step, Qwen3-4B 15.8 -> 42.2 tok/s. Backs our
  multi-token GEMV + MTP plan. Their +13% from a second shim channel ran
  below the read ceiling; ours (at the ceiling) gained nothing.
- **mlir-aie after v1.4.4** (PRs #3917, #3941, v1.4.5 dev wheels): new
  `gated_delta_rule` (fp32 state kept as bf16 limbs), `selective_scan`,
  `router_top_k`, `experts`, `sample_select` kernels; `repeat_parameter` for
  per-call DMA repeat counts; opt-in fp32 QK^T (bf16 scores lost accuracy
  past 2^15). The INT8 Llama-1B example notes cores come back reset after the
  driver's 5 s autosuspend: on-chip state must not be assumed to persist.
- **arXiv 2602.06063** (Gemma3 on XDNA2): GEMV on 4 columns, the other 4
  for attention, norms and RoPE so a layer stays resident; FlowKV pipelined
  decode attention (score core + value core per KV head); megatile bf16 GEMM
  for prefill. Their reads cap below 40 GB/s (DDR5 laptops).
- **arXiv 2606.11357** (TileFuse, W4A16 on XDNA2): unpack each 4-bit block
  once and reuse it across tokens / M sub-tiles (argues for a q4nx-native
  prefill GEMM over a bfp16 copy); short prompts lose on the GEMM path. Their
  Table 1 implies ~51 GB/s for 4-bit GEMV, matching our ceiling; their
  "millisecond" dispatch cost is their runtime's xclbin switching.
- **FastTPS** (arXiv 2607.11211, AMD): static in-place KV cache (no host
  concat), fused RoPE + QK^T + softmax + PV, interleaved gate/up layout.
- **FlashAttention on XDNA2** (arXiv 2609.21264): fully fused attention with
  scores in L1, 3.62 TFLOP/s; fuse until past the roofline ridge. For prefill.

## Background

- arXiv 2512.13282 (GEMM on XDNA/XDNA2), arXiv 2510.14871 (MLIR-AIR),
  arXiv 2606.07586 (kernel merging: prefill dispatches per layer 15 -> 3),
  HeteroMosaic (arXiv 2607.12839, MICRO 2026: NPU + iGPU GEMM split incl.
  Strix Halo; DMA reconfiguration cheap, PDI reload expensive; turbo DVFS
  shifts timings), Zen-Attention (arXiv 2508.17593: ~60 GB/s NPU on HX 370),
  STEEL (arXiv 2607.09385), GAMA (arXiv 2504.09688, Versal GEMM placement),
  Jena "hello xdna" (hand-written bfp16 GEMM at 95% of tile peak),
  persistent-state Gated DeltaNet on FPGA (arXiv 2603.05931).
- FLM issue #636: `xrt-smi validate` latency 35 us in turbo vs 63 us in
  performance mode (Ryzen AI 350). On our box turbo gave 55 vs ~68 us.

## Gaps

No published MoE or DeltaNet work on XDNA beyond the new mlir-aie kernels,
no independent bfp16 accuracy study, no published runlist/dispatch costs.
