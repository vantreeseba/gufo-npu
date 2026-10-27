# gufo-npu: an open NPU engine for Qwen3.6-35B-A3B on Strix Halo

## Context

gufo runs Qwen3.6-35B-A3B (GGUF arch `qwen35moe`) on the Strix Halo iGPU: ~51–60 tok/s decode, ~3k tok/s prefill.
The XDNA2 NPU sits idle. The only NPU runtime for this model is FastFlowLM (FLM, now `ROCm/FastFlowLM`, v1.0.10).
Its runtime is MIT, but the kernels and per-model engine (`.xclbin`, `libqwen3_6_moe_npu.so`) are closed binaries.
On Strix Halo it reaches about **16.5 tok/s decode, ~105 tok/s prefill, with a fixed ~2.7 s TTFT** (FLM issue #710).
It also currently fails to load the Qwen3.5/3.6 family on Strix Halo (`CREATE_HWCTX err=-22`, #705).
Because the kernels are closed, we can't fix FLM's slow parts. gufo-npu is a new, fully open engine that runs this one model on the NPU.
It copies gufo's conventions and CI, not its code.

Decisions made with the user:
- Build our own open engine: XRT host runtime plus our own kernels written with IRON / MLIR-AIE.
- Weights: load FLM's Q4NX pack first, then add a GGUF (Unsloth UD) path.
- Model: Qwen3.6-35B-A3B only.

## What limits speed (research summary)

| Fact | Value | Source |
|---|---|---|
| XDNA2 array | 4×8 compute tiles (64 KB L1 each), 8 memtiles (512 KB each), shim DMAs | arXiv 2512.13282 |
| NPU→DDR bandwidth on Strix Halo | **52–56 GB/s** measured (iGPU ~215) | dotLLM #1285 |
| Fixed cost per NPU dispatch | ~80–100 µs | dotLLM #1285 |
| Measured compute | ~38 TOPS int8, ~15 TFLOPS bf16 | arXiv 2512.13282 |
| Model | 40 layers (30 Gated DeltaNet + 10 gated full-attn GQA 16/2, hd256), 256 experts top-8 + 1 shared (ffn 512), hidden 2048, vocab 248k, 1 MTP layer | HF config |
| Bytes read per decoded token at ~4.5 bpw | ~1.7 GB (experts ~0.64, DeltaNet proj ~0.57, LM head ~0.29, attn ~0.15) | our arithmetic |

- **Decode is limited by DDR bandwidth.** 54 GB/s ÷ 1.7 GB gives a ceiling of ~30 tok/s. FLM uses only ~22 GB/s of that.
- **Per-call overhead is the first thing to fix.** Expert matrices are tiny (2048×512). Making one NPU call per expert, or per op, at ~90 µs each, wastes most of the budget.
- **Prefill on FLM is slow for two reasons.** It makes ~4,900 small expert calls one after another (#710), when experts should be batched by token into one grouped GEMM per layer. It also has a fixed TTFT overhead.
- **Batches of up to 4 tokens cost nothing extra in decode** (GEMV with M≤4 runs at the same speed). So MTP speculative decoding should give close to a free 1.5–2×.
- **DeltaNet recurrent state is ~2 MB per layer in fp32**, ~120 MB/token read+write over 30 layers. Keep it in bf16 and update it in place.

Targets, all on the user's Strix Halo, against FLM measured on the same machine:
- **MVP:** decode ≥ 22 tok/s, prefill ≥ 400 tok/s at 2k, TTFT < 1 s for short prompts.
- **Later:** ~28 tok/s decode, and 40+ tok/s with MTP.

## Architecture

```
gufo-npu serve ──> scheduler ──> model (qwen36moe) ──> npu runtime (XRT: BOs, hw context, runlists)
                                       │                       │
                         loaders: Q4NX | GGUF→NPU repack    kernels/*.xclbin + instr seqs (built from IRON/MLIR-AIE)
                                       │
                         cpu reference (fp32 oracle for tests)
```

- **Host:** C++20, CMake, Nix flake, the same layout as gufo:
  - `src/core` (gguf, q4nx, tokenizer, sampling)
  - `src/npu` (XRT wrapper)
  - `src/models/qwen36moe` (reference + NPU path)
  - `src/cli` (serve, bench, prompt)
- **Kernels:** `kernels/` holds Python IRON / MLIR-AIE designs plus AIE C++ core code, compiled with Peano + aiecc into `.xclbin` + instruction binaries at **build time**. Compiling needs no NPU, so the hosted CI runners can do it.
- **Fewer dispatches in decode.** One hardware context and one xclbin holding every op. Each layer runs as an XRT runlist of at most 2 dispatches:
  - **A.** Norm + DeltaNet-or-attention + router logits.
  - The host does top-8 on 256 logits (µs on Zen 5), then patches the expert offsets into the instruction stream (DMA addresses are offsets into one per-layer expert BO).
  - **B.** One fused kernel: 8 routed + 1 shared expert GEMV, gated sum, residual.
  - The LM head is a separate column-split GEMV. Per token that is about 81 dispatches. A spike will check whether the whole token can be chained into one runlist.
- **Weight layout:** pre-tiled to match the shim DMA access pattern, int4 with group scales (Q4NX-compatible). Dequant happens in-tile to bf16/bfp16. All weights live in pinned host BOs (memlock unlimited, as FLM does).
- **Prefill:** grouped-by-expert GEMM over a token chunk (each expert's weights read once per chunk), a chunked DeltaNet scan, and tiled flash-attention for the 10 full-attn layers.
- **CPU side:** tokenizer, sampling, top-k, scheduling, and anything too small to be worth a dispatch.

## Phases

**0. Repo + CI scaffold.** No hardware needed.
- `git init` gufo-npu as a fresh repo. Copy over adapted versions of gufo's:
  - `AGENTS.md`/`CLAUDE.md` symlink
  - `.clang-format`/`.clang-tidy`
  - `flake.nix` pattern
  - `CMakePresets.json` pattern
  - `release-please-config.json`
  - `tools/ci/check-pr-title.py`
  - `docs/models/<model>/{README,BENCHMARKS,QUALITY}.md` layout
  - MIT LICENSE + NOTICE crediting gufo.
- Workflows, modelled on `gufo/.github/workflows/`:
  - `ci.yml`: format, build, CPU tests on `ubuntu-latest`.
  - `linux-build.yml`: compile host + xclbins and upload artifacts.
  - `publish-image.yml`: build `ghcr.io/vantreeseba/gufo-npu:<branch>` and `:sha-<sha>`, as the fork's image workflow does.
  - `pr-title.yml` and `release-please.yml` (plain `GITHUB_TOKEN`, no App).
  - `npu-smoke.yml`: `workflow_dispatch` only, on self-hosted `[linux, strix-halo, npu]`. It stays idle until a runner is registered.
- You test locally with `podman run --device /dev/accel/accel0 --ulimit memlock=-1 ghcr.io/vantreeseba/gufo-npu:<branch> ...`.
- **Toolchain spike (risk #1):** pin mlir-aie + llvm-aie (Peano) wheels and XRT (lemonade PPA `libxrt-npu2` / amd/xdna-driver tag). Try to package them in Nix. If that fights back, use a pinned Ubuntu 24.04 build container for kernel compilation; Nix still builds the host.
- Replace the stale `fastflowlm-docker/dockerfile` approach (unpinned HEAD XRT) with pinned versions.

**1. Hardware microbenchmarks.** Needs your machine.
- `gufo-npu diagnose`: checks driver, firmware ≥1.1, memlock and pmode.
- `bench/npu`: DMA bandwidth, dispatch cost, runlist chaining, instruction-offset patching, and int4 GEMV for M=1..4.
- Gives us the measured ceilings for the targets. Also measure FLM on the same machine as the baseline (if #705 lets it load).

**2. CPU reference model.** The fp32 oracle. Inputs are GGUF + Q4NX readers, the tokenizer and the chat template. It runs:
- DeltaNet and the gated attention
- MoE routing
- MTP head

Teacher-forced logits are checked against HF transformers / llama.cpp dumps (see gufo's `tests/models/qwen35ba3b/target_test.cpp` and `src/models/qwen/modules/{moe,ssm,attention}.cpp` for how gufo does it; read only, no code copied wholesale).

**3. NPU decode, end to end.**
- Kernels: int4 GEMV, fused 9-expert MoE, DeltaNet step, GQA decode attention, RMSNorm/RoPE/gating, LM head.
- Each kernel gets an op test against the CPU reference (rel err < 1e-2), then a full-model greedy-argmax match under teacher forcing.

**4. NPU prefill.** Grouped expert GEMM, chunked DeltaNet, flash attention, and prefill-vs-decode consistency tests.

**5. Serve.** `gufo-npu serve`: `/v1/models`, `/v1/chat/completions` (streaming), `/health`, `/metrics`, single session, Q4NX. Uses a small vendored HTTP lib (cpp-httplib).

**6. Perf work past MVP.**
- GGUF→NPU repack (`gufo-npu convert`)
- MTP speculative decode, using the free M≤4 batch
- lower-bit experts as an option
- multi-session
- KV/state prefix cache

## Verification

- CI: `nix build .#checks.x86_64-linux.pr` (or the container equivalent) is green on hosted runners, and the image is pushed to GHCR.
- On your Strix Halo:
  - `gufo-npu diagnose`, then `ctest --preset npu` for the op tests. Model tests are gated on `GUFO_NPU_QWEN36_MODEL`; exit code 77 means skipped, which does not count as a pass.
  - `gufo-npu bench` for pp512/pp2048/tg128 against `flm` with the same prompts.
  - A `curl` streaming chat request against `gufo-npu serve`.
- Results go in `docs/models/qwen3.6-35b-a3b/BENCHMARKS.md` and `QUALITY.md`.

## Risks / open questions

1. Packaging the IRON/Peano toolchain reproducibly (Phase 0 spike).
2. Patching instruction offsets per layer: if this is slow or unsupported, fall back to keeping all 256 experts of a layer reachable with offsets passed as runtime args.
3. The same firmware hardware-context failure as FLM #705 could hit us too. We'll need your firmware/driver versions early.
4. Pinning all weights (~20 GB Q4) needs memlock unlimited.
5. When you provide machine access: SSH details, kernel/firmware/XRT versions, and whether it can host a GitHub self-hosted runner.
