# Qwen3.6-35B-A3B benchmarks

Hardware: Strix Halo (Ryzen AI MAX+ 395), 64 GB, Ubuntu 26.04, Linux 7.0,
xdna-driver 2.26, NPU runtime power control `auto`.

## Targets

| | FastFlowLM (reported, Strix Halo) | MVP target | Ceiling |
|---|---|---|---|
| Decode | ~16.5 tok/s | ≥ 22 tok/s | ~30 tok/s (54 GB/s ÷ 1.7 GB/token) |
| Prefill, 2k | ~105 tok/s | ≥ 400 tok/s | |
| TTFT, short prompt | ~2.7 s | < 1 s | |

FLM has not yet been measured on this machine.

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
