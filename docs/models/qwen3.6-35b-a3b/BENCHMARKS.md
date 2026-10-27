# Qwen3.6-35B-A3B benchmarks

Hardware: Strix Halo (Ryzen AI MAX+ 395), 64 GB, Ubuntu 26.04, Linux 7.0,
xdna-driver 2.26, NPU runtime power control `auto`.

## Targets

| | FastFlowLM (measured below) | gufo-npu NPU (below) | MVP target | Ceiling |
|---|---|---|---|---|
| Decode, 1k context | 16.9 tok/s | 19.8 tok/s | ≥ 22 tok/s | ~33 tok/s (56 GB/s NPU read ÷ 1.69 GB per token with every matrix in q4nx) |
| Prefill, 2k | ~178 tok/s | ~20 tok/s (no prefill yet: one token per pass) | ≥ 400 tok/s | |
| TTFT, 35-token prompt | 3.48 s | 1.4 s (24 tokens) | < 1 s | |

## NPU decode

`generate --device npu` (`NpuDecoder`): every matrix in q4nx on the NPU
(1.69 GB read per token), each layer's 256 experts and its shared expert in
one device arena read through two gathers (`npu::Q4nxGather`), the mixers
and the router on the CPU, instruction-buffer flow. Per layer: the mixer's
output projection (one submission), routing on the CPU, then the gate + up
and down gathers chained with the next layer's input projection (a second
submission): 81 submissions per token. One token per pass, prompt included;
no MTP yet.

Sweep, 2026-10-27: release build, greedy, `--draft none`, 128
tokens at most, three runs each after a warm-up run, short and ~1k prompts
interleaved, the GGUF in page cache. Each run loads the model anew.

| Prompt | Run 1 | Run 2 | Run 3 | Prompt time |
|---|---|---|---|---|
| Sky (24 tokens), decode | 18.59 tok/s | 20.17 | 20.30 | 1.31–1.48 s |
| Lighthouse story ×4 + question (942 tokens), decode | 19.79 tok/s | 19.75 | 19.85 | 47.4–49.1 s |

- Against FLM on the same box: decode **19.8–20.3 tok/s against 16.9–17.1**
  (+17–19%), TTFT 1.4 s against 3.49 s for short prompts. Long prompts are
  far slower than FLM's (48 s against 8.75 s at ~1k tokens), since they run
  one token per pass at decode speed; that needs the MoE prefill.
- Against the CPU decoder (23.0 tok/s, below): 12–14% slower; the NPU path
  reads 1.69 GB per token where the CPU reads the GGUF's 2.75 GB, but the
  NPU reads at most ~56 GB/s.
- Run 1 of the short prompt (18.6) was slower than runs 2 and 3; other CPU
  jobs were running on the box during the sweep.
- Output: the short reply is the same text in every run (one hash), and
  reads correctly ("The sky appears blue because of a phenomenon called
  Rayleigh scattering…").

Where a token's ~49.5 ms go (short prompt, run 3: 121 steps): waiting on
the NPU 45.6 ms (including ~81 submission round trips), CPU mixers 3.0 ms,
routing 1.9 ms (router dot products ~34 µs per layer on a 7-worker pool,
patching both gathers ~10 µs per layer). At ~1k context the mixers take
3.8 ms per step (attention over the cache).

`gufo-npu bench gemv` on the GGUF (calls one at a time, so each includes a
submission's ~60–75 µs; mean µs over 200 warm calls, two runs agree within
3%):

| Stage | Rows | MB | 1 token µs | 2 tokens µs | GB/s (1) | Per token |
|---|---|---|---|---|---|---|
| DeltaNet in (qkv + z) | 12,288 | 14.5 | 359 | 372 | 40.5 | × 30 |
| Attention in (q + gate, k, v) | 9,216 | 10.9 | 290 | 305 | 37.6 | × 10 |
| Mixer out (ssm_out, attn_output) | 2,048 | 4.8 | 173 | 181 | 28.0 | × 40 |
| MoE FFN (gate + up and down gathers, 9 experts) | 9,216 | 16.4 | 464 | 481 | 35.3 | × 40 |
| LM head (output) | 248,320 | 294.6 | 5,361 | 5,382 | 55.0 | × 1 |

Summed per token that is ~44.5 ms, close to the 45.6 ms measured in the
decoder (chained calls skip some of each call's fixed cost; the CPU work
between submissions adds it back). The gathers and the small mixer output
projection run well below the 56 GB/s read ceiling: per-call fixed cost.

Memory: device buffers 18.9 GiB (the 40 expert arenas, 467 MB each, are
18.7 GB; the dense matrices ~1 GB), anonymous memory ~0.6 GB, and up to
~20 GB of the memory-mapped GGUF in page cache while loading. Loading
(packing every matrix to q4nx, Q4_K experts code for code) takes 19.8–24 s.

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
- The CPU reads these weights at ~70 GB/s averaged over the step. On its
  own it streams 115–120 GB/s on 16 threads; the NPU reads at most ~56 GB/s
  (below).

## DDR bandwidth (Phase 1)

`gufo-npu bench bandwidth --iters 30`: 64 MiB int32 copy through all 8
columns, both shim DMA channels each (`kernels/designs/bw_memcpy.py`). Three
runs: **79.5, 80.7, 80.8 GB/s** counting read + write (≈40 GB/s each way,
concurrently). Reading only, decode's access pattern, the NPU reaches
55.9–56.4 GB/s ([2B benchmarks](../qwen3.5-2b/BENCHMARKS.md)), the ceiling
for NPU decode.

CPU and NPU reading at the same time (one-off probe, 2026-10-27, 256 MiB
NPU reads against a 16-thread CPU read loop, each timed alone first): the
NPU alone 55.6–55.9 GB/s, the CPU alone 120.0 GB/s; together the NPU 51.0,
55.6 and 55.6 GB/s and the CPU 119.5 GB/s, ~170 GB/s combined. With 8 CPU
threads: NPU 55.9–56.6, CPU 81.7. Neither slows the other much, so splitting
large matrix products between them is possible in principle.

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
