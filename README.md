# gufo-npu

An open-source engine for running Qwen models on the **AMD XDNA2 NPU** in Strix
Halo (Ryzen AI MAX+ 395), on Linux. The MVP targets **Qwen3.5-2B**; the second
pass adds **Qwen3.6-35B-A3B** (MoE).

The only NPU runtime for this model today, [FastFlowLM](https://github.com/ROCm/FastFlowLM),
ships its kernels as closed binaries. gufo-npu writes its own kernels with
AMD's open [IRON](https://github.com/amd/IRON) / [MLIR-AIE](https://github.com/Xilinx/mlir-aie)
toolchain so every bottleneck can be fixed. See [docs/PLAN.md](docs/PLAN.md).

**Status: Phase 3 started.** The toolchain, CI and NPU runtime path work end
to end, and an fp32 CPU reference of Qwen3.5-2B (GGUF) matches HF
transformers. `gufo-npu generate --device npu` decodes at 54–63 tok/s on
short prompts (English, code), ~48 tok/s at 2k context and 33 tok/s at 9k
without drafting (FLM: 28.5): q4nx GEMVs (4.625 bits per weight) stream
weights at the NPU's 56 GB/s read ceiling and also run the norms, residual
adds and SwiGLU, chained so a step needs 25 NPU submissions; the model's MTP
head drafts a token (scoring a 32k draft vocabulary) that a 2-token pass
verifies (same output as plain greedy). DeltaNet and attention run in fp32
on the CPU, threaded and vectorized, with an fp16 K/V cache.

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, CI, pinned toolchain, smoke kernel | done |
| 1 | Hardware microbenchmarks (dispatch, DMA bandwidth, int4 GEMV) | dispatch, copy bandwidth, FLM baseline done |
| 2 | CPU reference model (GGUF): forward pass, tokenizer, chat, greedy generate; matches HF | done (MTP and FLM pack: later) |
| 3 | NPU decode | chained NPU decode with MTP speculation and a draft vocabulary, 54–63 tok/s, 72/73 top-1 vs CPU |
| 4 | NPU prefill | |
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
Model docs: [docs/models/qwen3.6-35b-a3b](docs/models/qwen3.6-35b-a3b/README.md).

## License

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
