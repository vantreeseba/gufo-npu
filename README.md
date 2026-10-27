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
| [Qwen3.5-2B](docs/models/qwen3.5-2b/README.md) | NPU decode, MTP with a 32k draft vocabulary, NPU prefill | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF/blob/e05864f8066d874d5f85aaff007ae57a2a7d1efe/Qwen3.5-2B-UD-Q4_K_XL.gguf) | Up to **63.5 tok/s** decode (FLM 28.5); **0.11 s** to first token at 35 tokens, 1.23 s at 4k (FLM 0.60 s, 3.34 s) · [Benchmarks](docs/models/qwen3.5-2b/BENCHMARKS.md) | [Quality](docs/models/qwen3.5-2b/QUALITY.md) |
| [Qwen3.6-35B-A3B](docs/models/qwen3.6-35b-a3b/README.md) | NPU decode (MoE), CPU decode | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) | **20.3 tok/s** decode on the NPU (FLM 17.1); 23.0 tok/s on the CPU · [Benchmarks](docs/models/qwen3.6-35b-a3b/BENCHMARKS.md) | [Quality](docs/models/qwen3.6-35b-a3b/QUALITY.md) |

### Against FastFlowLM on the same machine

| Model | Workload | gufo-npu | FastFlowLM |
|---|---|---|---|
| Qwen3.5-2B | Decode, short prompt | 53–64 tok/s with MTP, 39 without | 28.5 tok/s |
| | Decode, 2k context | 48.4 tok/s with MTP | 28.0 (1k), 26.4 (4k) |
| | Decode, 9k context | 33.2 tok/s without drafting | 24.6 (8.5k) |
| | Time to first token, 35 tokens | 0.11 s | 0.60 s |
| | Time to first token, ~1k tokens | 0.31 s | 1.80 s |
| | Time to first token, ~4k tokens | 1.23 s | 3.34 s |
| Qwen3.6-35B-A3B | Decode, short prompt | 20.2–20.3 tok/s | 17.1 tok/s |
| | Decode, ~1k context | 19.8–19.9 tok/s | 16.9 tok/s |
| | Time to first token, short | 1.3–1.5 s | 3.49 s |
| | Time to first token, ~1k tokens | 47–49 s | 8.75 s |

- **How these were measured:** Strix Halo, Ryzen AI MAX+ 395, 2026-10-27. Greedy decoding, warm, three or more runs each.
  - gufo-npu runs in-process through `gufo-npu generate`.
  - FastFlowLM runs through its OpenAI-compatible server (streaming, temperature 0), so its time to first token includes a local HTTP round trip.
  - Each model's BENCHMARKS.md has the exact prompts and every run.
- **Short-prompt decode with MTP** depends on how many drafts are accepted: more on code, fewer on prose.
- **The 35B** has no MTP or batched prompt yet. Its prompt runs one token per pass at decode speed, so long prompts are slower than FastFlowLM's. A third short-prompt run measured 18.6 tok/s while another job shared the CPU.

**How it works:**
- Every weight matrix is stored as q4nx (4.625 bits per weight) and read by NPU GEMVs that run at up to the NPU's ~56 GB/s read ceiling. The same GEMVs also do the norms, residual adds and SwiGLU.
- Calls are chained into a few NPU submissions per token.
- DeltaNet, attention and MoE routing run in fp32 on the CPU.
- Prompts go through bfp16 GEMMs on the NPU (2B).
- The model's MTP head drafts tokens that a 2-token pass verifies, with the same output as plain greedy.

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, CI, pinned toolchain, smoke kernel | done |
| 1 | Hardware microbenchmarks (dispatch, DMA bandwidth, int4 GEMV) | done |
| 2 | CPU reference models (GGUF): forward pass, tokenizer, chat, greedy generate | done: 2B matches HF transformers, 35B matches llama.cpp |
| 3 | NPU decode | 2B with MTP; 35B (MTP next) |
| 4 | NPU prefill | 2B: bfp16 GEMMs (11–18 TFLOPS), ~1.56 GB more pinned memory; 35B not yet |
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
