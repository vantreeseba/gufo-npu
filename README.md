# gufo-npu

An open-source engine for running Qwen models on the **AMD XDNA2 NPU** in Strix
Halo (Ryzen AI MAX+ 395), on Linux. The MVP targets **Qwen3.5-2B**; the second
pass adds **Qwen3.6-35B-A3B** (MoE).

The only NPU runtime for this model today, [FastFlowLM](https://github.com/ROCm/FastFlowLM),
ships its kernels as closed binaries. gufo-npu writes its own kernels with
AMD's open [IRON](https://github.com/amd/IRON) / [MLIR-AIE](https://github.com/Xilinx/mlir-aie)
toolchain so every bottleneck can be fixed. See [docs/PLAN.md](docs/PLAN.md).

## Models and benchmarks

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
|---|---|---|---|---|
| [Qwen3.5-2B](docs/models/qwen3.5-2b/README.md) | NPU decode, MTP with a 32k draft vocabulary, NPU prefill | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF/blob/e05864f8066d874d5f85aaff007ae57a2a7d1efe/Qwen3.5-2B-UD-Q4_K_XL.gguf) | Up to **63.5 tok/s** decode (FLM 28.5); **0.09 s** to first token at 35 tokens, 1.20 s at 4k (FLM 0.60 s, 3.34 s); prompts at **3,400–3,500 tok/s** from 1k to 4k (FLM 590–1,270) · [Benchmarks](docs/models/qwen3.5-2b/BENCHMARKS.md) | [Quality](docs/models/qwen3.5-2b/QUALITY.md) |
| [Qwen3.5-4B](docs/models/qwen3.5-4b/README.md) | NPU decode on its own NPU programs, MTP with a 32k draft vocabulary, NPU prefill | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF/blob/86835bf9949e4d14d6860f7910b1340ad4f271a9/Qwen3.5-4B-UD-Q4_K_XL.gguf) | Up to **30.6 tok/s** decode with MTP, 18.8 without (FLM published: 15.9 at 1k†); **0.66 s** to first token at 1k tokens; prompts at **1,550 tok/s** at 1k, 1,160 at 9k (FLM published: 322 and 600†) · [Benchmarks](docs/models/qwen3.5-4b/BENCHMARKS.md) | [Quality](docs/models/qwen3.5-4b/QUALITY.md) |
| [Qwen3.6-35B-A3B](docs/models/qwen3.6-35b-a3b/README.md) | NPU decode (MoE), MTP with a 32k draft vocabulary, NPU prefill, CPU decode | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) | Up to **28.9 tok/s** decode on the NPU with MTP, ~20 without (FLM 17.1); **1.25 s** to first token at ~1k tokens, 10.7 s at 8.5k (FLM 8.75 s, 34.2 s); prompts at **800–900 tok/s** from 1k to 8.5k (FLM 121–249); 23.0 tok/s decode on the CPU · [Benchmarks](docs/models/qwen3.6-35b-a3b/BENCHMARKS.md) | [Quality](docs/models/qwen3.6-35b-a3b/QUALITY.md) |

### Against FastFlowLM on the same machine

| Model | Workload | gufo-npu | FastFlowLM |
|---|---|---|---|
| Qwen3.5-2B | Decode, short prompt | 53–64 tok/s with MTP, 39 without | 28.5 tok/s |
| | Decode, 2k context | 48.4 tok/s with MTP | 28.0 (1k), 26.4 (4k) |
| | Decode, 9k context | 33.2 tok/s without drafting | 24.6 (8.5k) |
| | Time to first token (prompt speed), 35 tokens | 0.086 s (407 tok/s) | 0.60 s (58.6 tok/s) |
| | Time to first token (prompt speed), ~1k tokens | 0.29 s (3,525 tok/s) | 1.80 s (587 tok/s) |
| | Time to first token (prompt speed), ~4k tokens | 1.20 s (3,422 tok/s) | 3.34 s (1,272 tok/s) |
| Qwen3.6-35B-A3B | Decode, short prompt | 26.5–28.9 tok/s with MTP, 19.4–20.3 without | 17.1 tok/s |
| | Decode, ~1k context | 26.0–26.5 tok/s with MTP, 18.7–19.8 without | 16.9 tok/s |
| | Time to first token (prompt speed), 35 tokens | 0.25 s (143 tok/s) | 3.49 s (10.1 tok/s) |
| | Time to first token (prompt speed), 125 tokens | 0.34 s (363 tok/s) | 5.17 s (24.2 tok/s) |
| | Time to first token (prompt speed), ~1k tokens | 1.25 s (844 tok/s) | 8.75 s (121 tok/s) |
| | Time to first token (prompt speed), ~4k tokens | 4.77 s (891 tok/s) | 19.47 s (218 tok/s) |
| | Time to first token (prompt speed), ~8.5k tokens | 10.69 s (795 tok/s) | 34.17 s (249 tok/s) |
| Qwen3.5-4B | Decode, short prompt | 28.0–30.6 tok/s with MTP, 18.8 without | – |
| | Decode, ~1k context | 25.4 tok/s with MTP, 18.3 without | 15.9 tok/s† |
| | Decode, ~2k context | 24.2 tok/s with MTP, 17.8 without | 15.6 tok/s† |
| | Decode, ~9k context | 17.7 tok/s with MTP, 14.7 without | 14.0 tok/s (8k)† |
| | Time to first token (prompt speed), 35 tokens | 0.19 s (186 tok/s) | – |
| | Time to first token (prompt speed), 1k tokens | 0.66 s (1,552 tok/s) | ~3.2 s† (322 tok/s†) |
| | Time to first token (prompt speed), 4k tokens | 2.84 s (1,440 tok/s) | ~7.5 s† (543 tok/s†) |
| | Time to first token (prompt speed), 9k tokens | 7.79 s (1,156 tok/s) | ~15 s† (600 tok/s at 8k†) |

- **How these were measured:** Strix Halo, Ryzen AI MAX+ 395, 2026-10-27 (the 2B's and 4B's times to first token 2026-10-28). Greedy decoding, warm, three or more runs each.
  - gufo-npu runs in-process: decode through `gufo-npu generate`, the 2B's and 4B's times to first token through `gufo-npu bench prefill` (back to back, so the NPU never idles into its 5 s autosuspend).
  - Prompt speed is prompt tokens divided by time to first token, for both engines, so it includes each request's fixed costs (the first token itself, and FLM's HTTP round trip).
  - FastFlowLM runs through its OpenAI-compatible server (streaming, temperature 0), so its time to first token includes a local HTTP round trip.
  - Each model's BENCHMARKS.md has the exact prompts and every run.
- **† Qwen3.5-4B's FastFlowLM numbers are FLM's own published results on Kraken Point** (Ryzen AI 7 350; FastFlowLM v0.9.38), **not measured on this box**: Lemonade, which serves FLM here, is stopped. Its times to first token are our prompt lengths at FLM's published prefill rates. FLM's 35B decodes at 17.5 tok/s on Kraken Point and 16.9 here.
- **Short-prompt decode with MTP** depends on how many drafts are accepted: more on code, fewer on prose.
- **The 35B** prefills prompts of 6 tokens or more layer by layer: its dense matrices as bfp16 GEMMs on the NPU (converted once from the decoder's 4-bit weights), its experts on the CPU straight from the decoder's 4-bit weights (AVX-512 BF16); shorter prompts run two tokens per decode pass, which is faster there.

### Where the work runs

The NPU's share of the FLOPs (and of the bytes read, in decode). These are **computed** from each model's shapes and where each op runs, not measured (`tools/bench/work_split.py`; per op group in each model's BENCHMARKS.md):

| Model | Decode, short context | Decode, 9k context | Prefill, ~1k tokens | Prefill, longest prompt |
|---|---|---|---|---|
| Qwen3.5-2B | 99% (97.5% of bytes) | 89% (89%) | 98% | 95% (4k) |
| Qwen3.5-4B | 99% (97%) | 87% (87%) | 98% | 91% (9k) |
| Qwen3.6-35B-A3B | 97% (91%) | 78% (83%) | 51%, experts on the CPU | 46% (8.5k) |

- **gufo-npu:**
  - **Decode:** every weight matrix runs on the NPU except two small fp32 ones on the CPU: DeltaNet's alpha/beta projections and the 35B's router. The CPU also runs DeltaNet's recurrence, attention over the K/V cache (its share grows with the context) and the argmax. MTP steps are within a point of these numbers.
  - **The 35B's prefill:** it runs its experts on the CPU (44% of the FLOPs), because that is faster than its NPU expert kernels at every prompt length.
- **FastFlowLM** keeps more than 99.9% of the FLOPs on the NPU in every mode, including the router, DeltaNet and attention. Its CPU only samples in decode and does elementwise steps (SwiGLU, gates, top-8) in prefill. That comes from its v1.0.10 engine libraries' symbols and call graph, not its docs.
- **Work is not time:** the CPU is slower per FLOP, so its share of the measured time is larger.

**How it works:**
- Every weight matrix is stored as q4nx (4.625 bits per weight) and read by NPU GEMVs that run at up to the NPU's ~56 GB/s read ceiling. The same GEMVs also do the norms, residual adds and SwiGLU.
- Calls are chained into a few NPU submissions per token.
- DeltaNet, attention and MoE routing run in fp32 on the CPU.
- Prompts go through bfp16 GEMMs on the NPU (2B, 4B; the 35B's dense matrices, with its experts on the CPU).
- The 4B runs on its own NPU programs (decode GEMV and prefill GEMM built with its geometry), so its kernels cannot change the 2B's.
- The model's MTP head drafts tokens that a 2-token pass verifies, with the same output as plain greedy.

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, CI, pinned toolchain, smoke kernel | done |
| 1 | Hardware microbenchmarks (dispatch, DMA bandwidth, int4 GEMV) | done |
| 2 | CPU reference models (GGUF): forward pass, tokenizer, chat, greedy generate | done: 2B matches HF transformers, 35B matches llama.cpp |
| 3 | NPU decode | 2B, 4B and 35B, all with MTP |
| 4 | NPU prefill | 2B and 4B: bfp16 GEMMs (11–18 TFLOPS), ~1.56 GB (2B) / ~4 GB (4B) more pinned memory; 35B: dense matrices as bfp16 GEMMs on the NPU (converted once), experts on the CPU from the decoder's q4nx, ~1.5 GB more pinned |
| 5 | OpenAI-compatible server | |

## Try it

```sh
docker run --rm --device /dev/accel/accel0 --ulimit memlock=-1:-1 \
  -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro \
  ghcr.io/vantreeseba/gufo-npu:main diagnose
# also: smoke, bench dispatch, bench bandwidth, bench gemv --model GGUF
```

Drop the `/opt/xilinx/xrt` mount to use the XRT bundled in the image. Host
requirements and pinned versions: [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md).
Testing and CI: [docs/TESTING.md](docs/TESTING.md).
Model docs: [docs/models](docs/models/README.md).

## License

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
