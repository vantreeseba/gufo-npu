# MoE decode on the NPU (Qwen3.6-35B-A3B): choosing experts per call

Spike, 2026-10-27. Question: how to run each token's 8 routed experts and the
shared expert per layer on the NPU, with the experts chosen at run time,
without one NPU call per expert. Measured on the test box (Strix Halo,
XDNA2, XRT from the toolchain image); all NPU runs had the NPU to themselves.

## The model's MoE, from the GGUF

| | |
|---|---|
| Layers | 40 (30 Gated DeltaNet, 10 gated attention, every 4th) + 1 MTP |
| Hidden | 2048 |
| Experts | 256, 8 routed per token, FFN width 512 |
| Shared expert | width 512, sigmoid gate `ffn_gate_inp_shexp` (a 2048-vector) |
| Router | `ffn_gate_inp` 2048 × 256, F32 (BF16 in the MTP layer) |
| Expert types | gate, up: Q4_K; down: Q5_K (`[2048, 512, 256]`, `[512, 2048, 256]`) |
| Everything else | Q8_0: attention, DeltaNet projections, `ssm_out`, shared expert, embedding, output |
| DeltaNet | 16 key heads, 32 value heads, head dim 128, inner 4096 |

At q4nx's 4.625 bits per weight, one expert is 1.21 MB of gate + up
(1024 × 2048) and 0.61 MB of down (2048 × 512). A token reads 9 of them per
layer, 655 MB over 40 layers. The whole token reads 1.69 GB, a 30 ms
ceiling at the NPU's 56 GB/s.

## Options measured

One layer's FFN, 9 experts, one token:

| Option | Per layer | Notes |
|---|---|---|
| (a) One ELF run per expert matrix (9 gate + up, 9 down) in a runlist, weights as sub-BOs of an arena | 948 µs | 52.7 µs per run: the per-call cost swamps 0.6–1.2 MB of weights |
| (b) As (a), re-binding the weight sub-BOs with `set_arg` between runlist executions | +38 µs | Works and stays correct (a re-bound run equals a fresh run, bit for bit), 2.1 µs per run; only (a) benefits |
| **(c) Two gather calls, expert offsets patched into the instruction stream** | **414 µs** in a 40-layer chain (458 µs alone) | Patch 3.2 µs per layer; matches the CPU to 4.9e-4 |
| Dense GEMV for comparison (8192 × 2048, 9.7 MB) | 216 µs | 45 GB/s |

The gather streams 16.4 MB per layer at 39.6 GB/s. The 2B model's dense
SwiGLU GEMV of similar size runs at about 38 GB/s.

### (c) in detail

- **The instruction-buffer ("legacy") flow works in our hardware context.** A
  kernel from the xclbin (`xrt::kernel(hwctx, "MLIR_AIE...")`) gets the
  instruction stream (`.insts.bin`, which the build already emits) as a
  host-owned, cacheable BO argument.
  - Its output is bit-identical to the ELF flow.
  - The stream's DDR patches are 12-word ops (opcode 0x81: register, buffer
    argument, 64-bit offset `arg_plus`; `aie/Runtime/TxnEncoding.h`). The
    firmware adds the argument's address at run time.
  - Rewriting `arg_plus` retargets the call's reads; tested against the ELF
    flow on another matrix. Rewriting 8 patches and syncing the BO takes
    0.6–0.9 µs.
- **ELF runs and legacy runs cannot share a runlist.** A mixed runlist fails
  with `ERT_CMD_STATE_NEW`, so every call in a chained decode step has to use
  the legacy flow.
  - Cost per call, measured in a 16-call runlist of 2048 × 2048 GEMVs: 74.1 µs
    ELF against 80.8 µs legacy with a shared instruction BO, and 83.1 µs with
    one BO per run.
  - A dense 8192 × 2048 GEMV costs the same in either flow inside the MoE
    chain (about 216 µs).
  - Non-cacheable instruction BOs are refused (`err=95`).
- **Gather variants** (`MOE_VARIANTS` in `kernels/designs/q4nx_gemv.py`), in
  the same xclbin and core program as every other GEMV:
  - **Gate + up, `rounds` gather.**
    - Round r is expert r's gate + up (1024 rows = one round of the 8 × 4
      array). Each column reads its 151 KB slice of that expert through its
      own fill: 72 DDR patches per call.
    - The `swiglu_scaled` output stage multiplies by the expert's router
      weight in fp32 before narrowing to bf16. The scale comes from RTP words
      7 + round × 2 + token, which the sequence writes with placeholder NaN
      payloads (`0x7FA5xxxx`). The host finds them once and rewrites them:
      18 words × 32 cores.
    - The input stage is the usual resnorm: H + D goes out as the next
      residual.
  - **Down, `k` gather.**
    - K is the 9 experts' 512-wide SwiGLU outputs back to back (4608), the
      gate + up call's output as written.
    - Each (round, column) slice is gathered from all 9 down matrices:
      144 patches of 38 KB.
    - The sum over experts happens in the accumulator. The output goes to
      the next layer's D slot like any down projection.
  - Only placeholder offsets are compiled in. The host maps each patch site
    to its (round, column, expert) from the placeholder value and writes
    `expert base + slice offset` into a per-layer arena BO.
- **Each awaited DMA task costs about 4.5–5 µs** of sequence time.
  - The down gather started at 211 µs (25.9 GB/s) with one awaited fill per
    segment.
  - A channel's tasks finish in order, so a group of 3 segments now awaits
    only its last fill per column, within the 16-BD budget and in queue
    order: 138 µs (39.6 GB/s).
  - The same grouping does nothing for the gate + up gather (279 → 285 µs),
    which already runs at the dense SwiGLU rate.
- **Correctness:** `tests/models/qwen36_35b_a3b/q4nx_moe_gather_test.cpp` (label `npu`) runs 20
  random selections of 9 of 16 experts with random router weights. Worst
  relative error against the CPU over the same q4nx weights: 4.9e-4.
- **The shared core program still fits:** 11.5 KB of 16 KB of program memory
  before the scaled output stage. All existing GEMV, chain and 2B decode
  tests pass. The 2B decode output is unchanged (same text, 39.2–39.3 tok/s).

### Measured layer chains

| Chain (legacy flow) | Time |
|---|---|
| Gate + up gather, down gather: one submission | 458 µs |
| The same, 40 layers in one runlist (80 calls) | 16.55 ms (414 µs per layer) |
| Gate + up, down, then a dense 8192 × 2048 resnorm input projection: one submission | 718 µs |
| The same, 40 in one runlist (120 calls) | 25.21 ms (632 µs per layer) |
| Two tokens on the same 9 experts (gather `_t2`) | 493 µs |

A submission boundary costs about 86 µs. A second token is nearly free when
both tokens use the same experts.

## Recommended design

Per layer and decode step:

1. **After the CPU mixer** (as in the 2B): one submission for the mixer's
   output projection, which writes D.
2. **CPU router.**
   - Read H and D (the out projection's input slots), compute
     h = H + D, then RMSNorm with `post_attention_norm`.
   - Router logits in fp32 on the CPU (256 × 2048; 2 MB per layer, or 1 MB
     in bf16), softmax, top 8, renormalize. Track A confirms the exact
     semantics.
   - The shared expert's gate is sigmoid(w · x).
   - Patch the two gathers' 216 offsets and 576 scale words, then sync both
     instruction BOs: 3.2 µs.
3. **One submission:** gate + up gather, down gather, then the next layer's
   input projection.

That makes 2 submissions per layer, ~81 per token. Every chained call uses
the legacy flow, and each layer has its own patched instruction BOs for its
two gathers. Weights:
- per layer, one arena BO with 256 experts and the shared expert (466 MB),
  18.6 GB over 40 layers
- dense weights 1.03 GB
- about 19.7 GB pinned in all

The box has 61 GB, about 37 GB of it free with the user's services running.

### Per-token budget (projection, one token per step)

| Part | Bytes | Time |
|---|---|---|
| MoE gathers, 40 layers | 655 MB | 16.6 ms (measured, chained) |
| DeltaNet input projections, 30 × (12288 × 2048) | 436 MB | 9.7 ms (at the measured 45 GB/s) |
| Attention input projections, 10 × (9216 × 2048) | 109 MB | 2.4 ms |
| Output projections, 40 × (2048 × 4096) | 194 MB | 4.6 ms (~42 GB/s) |
| LM head, 248,320 × 2048 | 294 MB | 5.4 ms (measured on the 2B) |
| ~81 submission boundaries | | ~5 ms (60–86 µs each) |
| CPU: mixers (30 × 32 DeltaNet heads), router, patching | | ~4.5 ms |
| **Total** | **1.69 GB** | **~48 ms, ~21 tok/s** |

For comparison:
- the bandwidth ceiling is 30 ms (33 tok/s)
- FLM on the same box: 16.5 tok/s
- the 2B's MTP speculation is the next lever

With MTP, a 2-token verify reads the dense weights once but the union of
both tokens' experts, up to 17 (8 + 8 routed + shared). Its cost:
- the gate + up gather takes any round count
- the down gather's K is limited to `MAX_DIM` (6144 = 12 experts), so
  raise `MAX_DIM` to 8704 (the token buffer still fits: 34.8 of 36 KB) or
  split the down into two calls
- a step costs ~1.35× a single token, so at ~70% acceptance that's ~+25%,
  around 26 tok/s

## Risks and open questions

1. **Quality of 4.625-bit weights where the GGUF keeps more.** The GGUF stores
   attention, DeltaNet projections, the shared expert, the embedding and the
   LM head at Q8_0, and the expert down projections at Q5_K. q4nx
   requantizes them all. Track A's oracle has to show whether
   teacher-forced top-1 holds.
   - If it doesn't, an 8-bit GEMV format for those tensors adds about
     0.85 GB per token, roughly 30% slower decode.
2. **The router stays on the CPU in fp32.** Routing is sensitive to near
   ties. It costs the per-layer sync point counted above.
3. **Moving every decode call to the legacy flow.** Measured at 0–9 µs per
   call. The 2B decoder would need the same change to share code.
4. **GEMV efficiency** (39–45 GB/s for 5–15 MB calls, 55 GB/s only for the
   LM head) is the largest gap to the ceiling, shared with the 2B.
5. **Dynamic runtime sequences.** mlir-aie can also generate the instruction
   stream at run time from a C++ builder (`TxnEncoding.h`, "dynamic runtime
   sequences"). That is an alternative to per-shape variants if expert
   counts vary a lot. Patching precompiled streams is cheaper per call.
   There are no MoE examples in the installed IRON or mlir-aie.

## Code

- `kernels/designs/q4nx_gemv.py`: `MOE_VARIANTS` (`rounds` and `k`
  gathers), grouped and awaited segment fills, `swiglu_scaled`, the
  `RTP_WORDS` layout and the scale placeholders. Also per-expert dense
  shapes for option (a): 2048 × 1024 SwiGLU and 512 × 2048 act.
- `kernels/aie/q4nx_gemv.cc`: `gemv_store(y, y_acc, rtp, round)` with the
  scaled SwiGLU stage.
- `tests/models/qwen36_35b_a3b/q4nx_moe_gather_test.cpp`: correctness and patch/layer timing.
- `tools/bench/npu_moe_spike.cpp`: the comparisons above, options (a), (b)
  and (c), and the chains. Build it against a release tree:

  ```sh
  g++ -std=c++20 -O2 -march=x86-64-v3 -Isrc tools/bench/npu_moe_spike.cpp \
    build/release/libgufo_npu_runtime.a build/release/libgufo_npu_core.a \
    -lxrt_coreutil -pthread -o npu_moe_spike
  ./npu_moe_spike build/release/kernels 16
  ```
