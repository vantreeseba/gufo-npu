# Prompt processing (prefill) on the NPU for Qwen3.6-35B-A3B

Spike, 2026-10-27. Today the 35B prompt runs one token per decode pass:
47–49 s at ~1k tokens. FastFlowLM (FLM) on the same box takes 3.49 s at 35
tokens, 5.17 s at 125, 8.75 s at 1,055, 19.5 s at 4,246 and 34.2 s at 8,499.

The question: how should the 35B's prefill run on the NPU so that it beats
FLM's time to first token (TTFT) at every length? The 2B's design does not
carry over. It keeps a bfp16 copy of every weight on the device, and the
35B's experts alone would need ~36 GB for that. So the NPU has to work from
the decoder's own q4nx buffers: 18.9 GiB of expert arenas and dense matrices.

All numbers below were measured on the test box (Strix Halo), warm, alone
on the NPU unless noted. The code is `tools/bench/qwen36_prefill_spike.cpp`,
`tools/bench/qwen36_route_dump.cpp`, `kernels/designs/qwen36_dequant.py`
(with `kernels/aie/qwen36_q4nx_dequant.cc`) and
`kernels/designs/qwen36_prefill_gemm.py`.

## What was measured

### 1. q4nx → bfp16 on the NPU, then the GEMM

`qwen36_dequant` is IRON's `flm.dequant` operator rebuilt for our compact
q4nx block (8-bit sub-scales and minima under bf16 d/dmin, 4736 bytes).
- **Kernel:** a preamble expands d·sc and dmin·mn into the per-32 bf16
  tables IRON's loop reads. The 4-bit codes are already laid out the same
  way in both formats.
- **Fill pattern:** it reads the GEMV's own block order, so the decoder's
  matrices and expert arenas convert in place, with no host repack.
- **Output:** bfp16 in `flm.GEMM`'s packed B order.
- **GEMM:** `qwen36_prefill_gemm` is the 2B's bfp16 GEMM configuration with
  the 35B's shapes, in its own xclbin.

Random q4nx weights (σ 0.02) and activations (σ 1). "Pair" is the dequant
followed by the GEMM, the two in separate hardware contexts:

| K × N | Dequant | GB/s moved | M | GEMM | TFLOPS | Pair | Pair TFLOPS | C error |
|---|---|---|---|---|---|---|---|---|
| 2048 × 12288 (DeltaNet in) | 1.26 ms | 34.1 | 256 | 1.10 ms | 11.7 | 7.40 ms | 1.7 | 1.08e-2 |
| | | | 512 | 1.75 ms | 14.8 | 8.29 ms | 3.1 | 1.08e-2 |
| | | | 1024 | 2.99 ms | 17.2 | 11.0 ms | 4.7 | 1.10e-2 |
| 2048 × 9216 (attention in) | 0.95 ms | 33.8 | 1024 | 2.24 ms | 17.2 | 8.92 ms | 4.3 | 1.08e-2 |
| 4096 × 2048 (mixer out) | 0.46 ms | 31.4 | 1024 | 0.99 ms | 17.4 | 6.56 ms | 2.6 | 1.09e-2 |

- **The GEMM matches the 2B's:** 11.6 TFLOPS at M = 256, 17.2 at 1024. It is
  compute-bound, so reading a bfp16 B that was converted on the device costs
  nothing extra.
- **The dequant streams at ~34 GB/s** (4.6 bits read plus 9 bits written per
  weight), about 20 G weights/s. All 1.4B dense weights take ~70 ms per
  chunk, so a resident bfp16 copy of them (1.6 GB) is optional.
- **The pairs are slow because of the context switch**, not the work (next
  section).
- **Errors, against the CPU over the same q4nx weights:**
  - The weights the GEMM sees are off by 8.4e-3 (rel. L2).
  - Packing the same weights on the host gives 6.2e-3. The kernel rounds
    twice (to bf16, then to bfp16) and rounds d·sc to bf16. A kernel that
    rounds fp32 straight to bfp16 would close the gap.
  - The GEMM output is off by 1.08e-2. That is slightly above the 1e-2 op
    target and in the same class as the 2B's prefill (9.2e-3 against its
    q4nx weights), whose model-level check is argmax agreement with decode.

### 2. Switching between hardware contexts costs milliseconds

Back-to-back runs, each awaited before the next starts:

| | dequant + dequant | GEMM + GEMM | dequant + GEMM | One switch | Both started, then awaited |
|---|---|---|---|---|---|
| 8-column xclbins | 222 µs | 264 µs | 5,137 µs | **4.9 ms** | 4,961 µs |
| 4-column xclbins (`NPU2Col4`) | 275 µs | 293 µs | 2,793 µs | **2.5 ms** | 2,751 µs |

Moving between two xclbins costs about 5 ms, or 2.5 ms when each covers 4
columns. Two 4-column contexts still take turns: started together, they do
not overlap. This sets the design:
- **Batch the dequant:** never alternate the dequant and GEMM contexts per
  layer. Convert several layers ahead per visit, or fuse the conversion
  into the GEMM so a single xclbin remains.
- **The 2B pays this too:** its prefill-then-decode moves between the GEMM
  and GEMV contexts once per prompt (~5 ms).

### 3. Real routing is very uneven

`qwen36_route_dump` ran a 1,280-token mixed prompt (prose, Python, C++, a
list, Spanish and German passages) through the NPU decoder and recorded
every token's 8 experts in every layer:

| Chunk tokens | Active experts per layer (mean, min) | Busiest expert per layer: mean, max tokens | (token, expert) pairs | Rows when each expert is padded to 256 | Padded to 64 |
|---|---|---|---|---|---|
| 35 | 69, 50 | 27, 33 | 280 | 17,683 | 4,421 |
| 128 | 114, 88 | 81, 126 | 1,024 | 29,203 | 7,400 |
| 256 | 141, 111 | 149, 247 | 2,048 | 36,154 | 9,453 |
| 1,024 | 227, 201 | 344, 618 | 8,192 | 58,630 | 18,130 |
| 1,280 | 231, 204 | 456, 831 | 10,240 | 60,282 | 20,104 |

- **A few experts take most of the tokens.** On average a layer's busiest
  expert gets 11× the mean, and up to 21×. Short prompts touch few experts:
  69 of 256 at 35 tokens.
- **Padding is costly:** padding each expert to `flm.GEMM`'s minimum
  M = 256 multiplies the work 7× at 1k tokens. A 64-row granule would cost
  2.2×.
- Uniform random routing (8 distinct experts per token) would instead give
  a busiest expert of ~50 tokens at 1k.

### 4. One layer's routed experts: dequant, then one GEMM per expert

Per layer:
1. Dequantize the experts that have tokens (or all 256 in 96 batched calls).
2. Group the chunk's token rows by expert into slots of 256/512/1024 rows.
3. One gate + up GEMM per expert, on sub-buffers of the grouped activations
   and of the bfp16 scratch.
4. SwiGLU and the router weight on the CPU.
5. One down GEMM per expert.
6. Scatter-add on the CPU.

GEMMs run as runlists of 24. Recorded routing, random weights:

| Chunk | Layer | Experts | Busiest | Padded rows | Dequant (active) | Gate + up GEMMs | Down GEMMs | Calls | Useful TFLOPS | CPU group / SwiGLU / scatter (1 thread) | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0 | 226 | 49 | 57,856 | 38.5 ms | 25.7 ms | 18.7 ms | 452 | 0.29 | 2.5 / 5.7 / 5.4 ms | 1.9e-2 |
| 256 | 13 | 133 | 218 | 34,048 | 22.5 ms | 15.1 ms | 11.1 ms | 266 | 0.49 | 1.7 / 4.7 / 4.7 ms | 2.0e-2 |
| 256 | 39 | 132 | 131 | 33,792 | 22.7 ms | 15.0 ms | 11.0 ms | 264 | 0.50 | 1.8 / 4.7 / 4.6 ms | 1.9e-2 |
| 1024 | 0 | 256 | 135 | 65,536 | 43.9 ms | 29.0 ms | 21.2 ms | 512 | 1.03 | 4.7 / 17.4 / 16.4 ms | 1.9e-2 |
| 1024 | 13 | 234 | 446 | 61,184 | 39.8 ms | 26.7 ms | 19.5 ms | 468 | 1.12 | 4.0 / 17.2 / 16.3 ms | 2.0e-2 |
| 1024 | 26 | 229 | 608 | 59,904 | 39.2 ms | 26.1 ms | 19.2 ms | 458 | 1.14 | 4.0 / 17.2 / 16.3 ms | 1.9e-2 |
| 1024 | 39 | 223 | 290 | 57,344 | 37.9 ms | 25.3 ms | 18.3 ms | 446 | 1.18 | 4.0 / 16.9 / 15.9 ms | 1.9e-2 |

- **Dequantizing all 256 experts of a layer takes 40.9 ms** at 33.5 GB/s
  moved, about as long as the GEMMs.
- **The GEMMs are call- and padding-bound.** A gate + up call costs ~113 µs
  and a down ~82 µs whatever the expert's token count, about 8 TFLOPS on
  padded rows. Only 0.3–1.2 TFLOPS of that is useful work.
- **The CPU stages are single-threaded here.** On the prefill pool they
  shrink ~10× (2–4 ms per layer at 1k).
- **The error is 1.9–2.0e-2** for the whole expert FFN (two chained bfp16
  GEMMs with a bf16 intermediate) against the CPU over the same q4nx weights.
  The model-level check, top-1 against decode as for the 2B, decides
  whether that is enough.

## The design

### Per prompt (v1, buildable from the spike's parts)

Short prompts (under ~32 tokens) keep the decode path; once two-token MoE
steps land with MTP, those run about two tokens per 67 ms. Longer prompts
run layer by layer over the whole prompt as one chunk, up to 8k tokens, so
each expert's weights are converted once per prompt:

1. **NPU, GEMM context:** the layer's input projection (DeltaNet qkv + z,
   or attention q + gate, k, v) over all prompt rows, M ≤ 1024 per call.
   The weights are bfp16 converted for this prompt, or a resident 1.6 GB
   copy.
2. **CPU:** the mixer (chunked DeltaNet scan, or causal attention over the
   fp16 cache), batched as in the 2B's `NpuPrefill`.
   - The dequant of upcoming layers' experts does not depend on
     activations, so it can be scheduled while the CPU mixes.
3. **NPU:** the output projection. **CPU:** the residual, the
   post-attention norm, fp32 router logits, softmax, top 8 and
   renormalization (moe.hpp, exactly as decode), the shared-expert gate,
   and grouping rows by expert.
4. **NPU:** the shared expert's gate + up and the routed experts' gate + up
   GEMMs, one per expert with tokens. Hot experts take 512/1024-row slots
   or several calls.
5. **CPU:** SwiGLU times the router weight. **NPU:** the down GEMMs.
   **CPU:** scatter-add into the residual.
6. **Dequant context:** expert scratch for the next layers. Visit it every
   4 layers (a 3.6 GB ring of bfp16 scratch, 0.91 GB per layer) to pay ~20
   context switches per prompt instead of 80.
7. After the last layer, switch once to the decode GEMV context for the
   last token's LM head and the MTP cache, as the 2B does.

### v2: fuse the conversion into a grouped GEMM

v2 is a grouped expert GEMM that reads q4nx directly. It uses the dequant
kernel's loop in the GEMM core to unpack each 32 × 256 block to bfp16 in
data memory, then runs the 8×8×8 bfp16 mmul over that expert's rows.
- **One expert per core row:** 4 experts per call, each with a 64-row M
  tile. A is broadcast along the row as `flm.GEMM` does, and B streams to
  each core from its own expert's blocks.
- **Ends the dequant problem:** no scratch and no conversion pass, one
  xclbin (so no context switches), and 4.6 bits read per weight instead of
  4.6 + 9 + 9.
- **Cuts padding:** to 2.2× at 1k, against 7× with v1.
- **Estimate:** each layer's experts take ~10 ms to read (467 MB at
  ~45 GB/s), with compute on 64-row slots of ~11 ms at 1k, 26 ms at 4k and
  48 ms at 8.5k (at ~10 TFLOPS).
- **Risks:**
  - data memory: the q4nx input buffers (2 × 4.7 KB) next to `flm.GEMM`'s
    A, B and C buffers
  - the 16 KB program memory with both kernels in one core
  - a new B data path: per core instead of per column

## Projected TTFT against FLM

Parts are taken from the tables above:
- **Experts:** measured per-layer costs, scaled by real routing.
- **Dense GEMMs:** 2.8 GFLOP per token at the measured 11.6–17.2 TFLOPS.
- **CPU mixers:** the 2B's measured prefill CPU times scaled by this
  model's layer and head counts. DeltaNet ×3.3 (linear), attention ×3.3
  (quadratic in length), other work ×2.
- **Pool:** the CPU stages of section 4 on the pool.

Stages are summed one after another, with no overlap credited, not even
the dequant behind the CPU mixer.

| Prompt tokens | v1 experts | Dense | CPU | Switches + last token | **v1 TTFT** | **v2 TTFT** | FLM |
|---|---|---|---|---|---|---|---|
| 35 | 1.0 s (69 experts/layer) | 0.06 s | 0.02 s | 0.15 s | **~1.2 s** (decode path similar) | ~0.5 s | 3.49 s |
| 256 | 2.0 s | 0.06 s | 0.12 s | 0.15 s | **~2.3 s** | ~0.6 s | ~5.7 s (interpolated) |
| 1,024 | 3.4 s | 0.18 s | 0.5 s | 0.15 s | **~4.2 s** | ~1.2 s | 8.75 s |
| 4,096 | 4.4 s | 0.72 s | 2.6 s | 0.2 s | **~7.9 s** | ~4.3 s | 19.5 s |
| 8,500 | 4.3 s | 1.5 s | 7.1 s | 0.2 s | **~13 s** | ~10 s | 34.2 s |

- **Both versions beat FLM at every length:** v1 by 2.1–2.9×, v2 by
  3.4–7×. v1's narrowest margin is around 1k tokens.
- **Long prompts become CPU-bound:** at 8.5k, over half of v1 (7.1 of
  ~13 s) is CPU attention and DeltaNet, which grow faster than the expert
  work. Overlapping CPU and
  NPU across two half-prompt chunks cuts that only if the dequant is not
  repeated per chunk, which is another point for v2. NPU prefill attention
  (IRON's `prefill_attn`) is the lever after that.
- **The estimates for 4k and 8.5k** extrapolate the GEMM cost from padded
  rows. The CPU numbers are scaled from the 2B, not measured on the 35B.

## Memory

On top of the decoder's 18.9 GiB of device buffers:

| | v1 | v2 |
|---|---|---|
| Expert bfp16 scratch | 1.8 GB (2 layers) to 3.6 GB (4 layers ahead) | 0 |
| Dense bfp16 (optional: converting per prompt costs ~70 ms) | 0–1.6 GB | 0–1.6 GB |
| Prompt activations and grouped rows (8k tokens) | ~1 GB | ~0.7 GB |
| **Pinned total** | **~22–25 GB** | **~20–21 GB** |

The box keeps ~25 GB free for this with the user's services running, so v1
should convert the dense matrices per prompt and keep a 2-layer ring.

## Alternatives considered

- **CPU expert GEMMs, NPU dense ("hybrid").**
  - CPU and NPU read memory concurrently (~120 + ~55 GB/s measured).
  - Grouped fp32 expert GEMMs over Q4_K on 16 Zen 5 cores: ~2 TFLOP per 1k
    tokens at an assumed ~1.5 TFLOPS is ~1.3 s per 1k, or under 0.5 s with
    int8 VNNI and 8-bit activations, as llama.cpp does.
  - Competitive with v1. It is not measured here, though, and it competes
    with the CPU mixers, which already dominate long prompts. Worth a spike
    if v2 slips.
  - Splitting each layer's experts between CPU and NPU by token load is
    also possible.
- **Decode path for every prompt** (two-token MoE passes, ~34 ms per
  token): wins only under ~30–50 tokens. 125 tokens would take ~4.2 s
  against FLM's 5.17 s, and 256 tokens ~8.6 s against ~5.7 s.
- **A resident bfp16 copy of the experts:** ~36 GB, which doesn't fit.
- **Two 4-column contexts side by side** (dequant on one, GEMM on the
  other): measured above; they take turns and switch in 2.5 ms.
- **One xclbin holding both the dequant and the GEMM** on disjoint columns
  or modes: removes the switches without writing the fused kernel. It is a
  smaller step than v2, but it keeps the 4.6 + 9 + 9 bits of traffic per
  weight and the scratch.

## Building it

- **v1, about 1–1.5 weeks:**
  - turn the spike's dequant and GEMM designs into the 35B package's prefill
    xclbins, with the dequant rounding fixed
  - `NpuPrefill` for MoE: whole-prompt chunks; per-layer router, grouping,
    SwiGLU and scatter on the `BurstPool`; the dequant ring scheduled behind
    the mixers; hot-expert splitting
  - the MTP cache over the prompt
  - tests: prefill state (conv windows, DeltaNet states, K/V cache, MTP
    cache) against the decode path; teacher-forced top-1 against decode
    and llama.cpp
  - a TTFT sweep against FLM
- **v2, about 2–3 more weeks, the main risk:**
  - the fused grouped q4nx GEMM design (B per core from its row's expert,
    unpack in core, `fused_mm`)
  - variable experts per call by patched offsets, as the decode gathers do
  - L1 and program memory fit
  - op tests and a determinism stress run (DMA wait order)
- **Then:** NPU prefill attention for prompts past ~4k.

## Code

- `src/models/qwen36_35b_a3b/kernels/aie/prefill_dequant.cc` and
  `prefill_dequant.py`: compact q4nx to bfp16 in GEMM order, from the GEMV's
  block order (xclbin `qwen36_35b_a3b_dequant`). Since v1 it rounds each
  weight once (the spike's version rounded to bf16 first). The `cols`
  parameter builds narrower variants.
- `src/models/qwen36_35b_a3b/kernels/prefill_gemm.py`: the 35B GEMM shapes
  (M 256, 512, 1024; xclbin `qwen36_35b_a3b_prefill`).
- `src/models/qwen36_35b_a3b/npu_prefill.cpp`: v1 (below the design above).
- `tools/qwen36_35b_a3b/prefill_spike.cpp`: the dense, switch and expert
  measurements. `SPIKE_ONLY=dense|switch|experts`; a routing file as the
  second argument.
- `tools/qwen36_35b_a3b/route_dump.cpp`: records routing through the NPU
  decoder.
