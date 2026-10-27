# Qwen3.6-35B-A3B benchmarks

Hardware: Strix Halo (Ryzen AI MAX+ 395), 64 GB, Ubuntu 26.04, Linux 7.0,
xdna-driver 2.26, NPU runtime power control `auto`.

## Targets

| | FastFlowLM (measured below) | MVP target | Ceiling |
|---|---|---|---|
| Decode, 1k context | 16.9 tok/s | ≥ 22 tok/s | ~30 tok/s (est. 54 GB/s read ÷ 1.7 GB/token; read-only bandwidth not yet measured) |
| Prefill, 2k | ~178 tok/s | ≥ 400 tok/s | |
| TTFT, 35-token prompt | 3.48 s | < 1 s | |

## FastFlowLM baseline

`qwen3.6-moe:35b-a3b` (FLM Q4NX, `Qwen3.6-35B-A3B-NPU2`, footprint 22.1 GB)
served by Lemonade 2026.43.3. Measured 2026-10-27 with
`tools/bench/openai_bench.py`: streaming chat, temperature 0, unique prompt
per request (no prefix cache), model already loaded, 2 warm-up requests.
Two independent sweeps agree within ~1%; the table is the second sweep, mean
of 3 repeats, 256 generated tokens.

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 35 | 3.49 | 10.1 | 17.1 |
| 125 | 5.17 | 24.2 | 17.1 |
| 1,055 | 8.75 | 120.6 | 16.9 |
| 2,119 (sweep 1) | 11.91 | 177.9 | 16.6 |
| 4,246 | 19.47 | 218.0 | 16.1 |
| 8,499 | 34.17 | 248.8 | 15.0 |

- TTFT carries ~3.3 s of fixed cost per request even for a 35-token prompt;
  long prompts then cost ~3.5 ms per token (~285 tok/s marginal prefill).
- Decode falls slowly with context: 17.1 → 15.0 tok/s from 35 to 8.5k tokens.

Reproduce: `python3 tools/bench/openai_bench.py --base-url http://<host>:13305/api/v1 --model qwen3.6-moe-35b-a3b-FLM --prompt-tokens 16,128,1024,4096,8192 --repeats 3 --max-tokens 256 --warmup 2`.

## CPU decode (reference point)

The CPU decode path checked by `qwen36_decoder_test`: `Decoder` over
`GgufProjections` in fp32, straight from the UD-Q4_K_XL GGUF (2.75 GB read
per token, mostly Q8_0; a fused Q8_0 dot, other types dequantized per row),
16 threads, release build, GGUF in page cache. 2026-10-27: one decode pass
over 10 chat-prompt tokens to warm up, then the same 10 tokens timed; three
runs: 43.5, 43.6 and 43.4 ms per token, **23.0 tok/s** at short context.

- Matrix products take 39.5 ms of it: the LM head 5.2 ms (540 MB, ~104 GB/s),
  the 640 routed-expert products 18.4 ms (~29 µs each), DeltaNet and
  attention inputs 10.1 ms. The rest is the CPU mixers and routing.
- With one thread per hardware thread the same step took 635 ms: every
  `SpinPool::Run` waits for all its workers, and an oversubscribed CPU parks
  one of them for a time slice (~0.7 ms) per call. The pools now size
  themselves from the CPUs the process may use (affinity mask and cgroup
  quota, `src/core/cpus.hpp`), each at most half of them: limited to 8 CPUs
  (`docker --cpuset-cpus=0-7` or `--cpus=8`) the decoder test runs at
  9.3 tok/s instead of falling off that cliff, and the 2B NPU decode keeps
  its 7 mixer workers (sky prompt, no draft: 39.37, 39.13, 39.16 tok/s).
- The CPU reads these weights at ~70 GB/s averaged over the step, more than
  the NPU's read bandwidth (52–56 GB/s in the plan's research; ~40 GB/s
  each way in `bench bandwidth` below).

## DDR bandwidth (Phase 1)

`gufo-npu bench bandwidth --iters 30`: 64 MiB int32 copy through all 8
columns, both shim DMA channels each (`kernels/designs/bw_memcpy.py`). Three
runs: **79.5, 80.7, 80.8 GB/s** counting read + write (≈40 GB/s each way,
concurrently). A read-only probe (decode's access pattern) is next.

## Dispatch cost (Phase 1)

`gufo-npu bench dispatch --rounds 200`, smoke kernel (4096 × int32, one
tile), 2026-10-27, gufo-npu 0.1.0. Bundled XRT 2.25 and host XRT 2.26 agree
within noise.

| Mode | Depth | µs per dispatch |
|---|---|---|
| sync (dispatch, wait) | 1 | 66.9 |
| runlist (dependent chain) | 4 | 34.2 |
| runlist | 8 | 27.6 |
| runlist | 16 | 23.2 |
| runlist | 32 | 21.5 |
| runlist | 81 | 20.3 |
| async (independent, in flight) | 81 | 23.4 |

- About 45 µs of a sync dispatch is the host round trip (completion and
  wake-up); chaining in a runlist removes it.
- About 20 µs per command remains, which is the per-command firmware cost.
  Fewer, larger (fused) commands are the only way under it.
- At ~81 dispatches per token, chained dispatch costs ~1.6 ms per token,
  against ~5.4 ms when issued one at a time.
- The first dispatch after idle took 270–1,100 µs with runtime power control
  `auto`.
