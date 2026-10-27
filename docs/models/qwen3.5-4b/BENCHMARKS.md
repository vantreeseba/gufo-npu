# Qwen3.5-4B benchmarks

Hardware: Strix Halo (Ryzen AI MAX+ 395), 64 GB, Ubuntu 26.04, Linux 7.0,
xdna-driver 2.26. Model: Unsloth UD-Q4_K_XL (README.md). Release build,
toolchain image `gufo-npu-toolchain:dev`.

## Against FastFlowLM

FLM could not be measured on this box (Lemonade, which serves it here, is
stopped). The FLM numbers below are **FLM's own published results for
`qwen3.5:4b` on Kraken Point** (Ryzen AI 7 350, 32 GB; FastFlowLM v0.9.38,
[qwen3.5_results.md](https://github.com/ROCm/FastFlowLM/blob/main/docs/docs/benchmarks/qwen3.5_results.md)),
**not measured on this box**. FLM's Qwen3.6-35B-A3B decodes at 17.5 tok/s
there and 16.9 here, so its decode on the two chips is close; a measured
baseline needs FLM run on this box.

| Context | gufo-npu decode, no draft | gufo-npu decode, MTP | FLM decode (published, Kraken) | gufo-npu prompt time | FLM prompt time (our prompt length at its published prefill rate) |
|---|---|---|---|---|---|
| 24-33 tokens (sky, fib) | 18.5-18.8 tok/s | 28.0 (sky), 30.6 (fib) | – | 0.37-0.39 s | – |
| ~1k (942) | 18.3 | 25.4 | 15.9 | 0.84 s | ~2.9 s (322 tok/s at 1k) |
| ~2k (2,092) | 17.8 | 24.2 | 15.6 | 1.72 s | ~4.7 s (446 tok/s at 2k) |
| ~4k (4,162) | – | 21.7 (32 tokens) | 15.0 | 3.29 s | ~7.7 s (543 tok/s at 4k) |
| ~9k (8,992) | 14.7 | 17.7 | 14.0 (8k) | 8.0 s | ~15 s (600 tok/s at 8k) |

## Decode and prompt time

`gufo-npu generate --device npu` (`qwen35_4b::NpuDecoder` with
`qwen35_4b::NpuPrefill`), greedy, 2026-10-27: a warm-up round, then three
rounds, each round interleaving every prompt with and without MTP (the
long prompts: the 230-token lighthouse passage repeated, then a question).
Each run loads the model. Runs agree within 1%; the table gives the range of
the three. The output hash is the same in every run and with or without MTP
(speculation is exact).

| Prompt | Draft | Decode tok/s | Drafts accepted | Prompt time |
|---|---|---|---|---|
| Sky (24 tokens), 74 generated | none | 18.77-18.83 | | 0.37-0.39 s |
| | MTP | 27.92-28.17 | 70% | |
| Fibonacci (33 tokens), 256 generated | none | 18.50-18.52 | | 0.37-0.39 s |
| | MTP | 30.59-30.65 | 89% | |
| 942 tokens, 95 generated | none | 18.27-18.36 | | 0.83-0.86 s |
| | MTP | 25.38-25.51 | 61% | |
| 2,092 tokens, 97 generated | none | 17.77-17.88 | | 1.69-1.74 s |
| | MTP | 24.07-24.34 | 57% | |
| 4,162 tokens, 32 generated | MTP | 21.46-21.84 | 52% | 3.28-3.29 s |
| 8,992 tokens, 97 generated | none | 14.69-14.77 | | 7.95-7.98 s |
| | MTP | 17.63-17.72 | 59% | 8.02-8.28 s |

- **Per token, no draft, short prompt:** 53 ms, of it 51.7 ms waiting on
  the NPU (33.5 submissions per token) and 2.7 ms in the CPU mixers. That is
  2.42 GB of weights at ~46 GB/s effective, against the 56 GB/s read ceiling
  (~43 ms, ~23 tok/s). The 2B runs at ~42 GB/s effective.
- **Context:** at 9k the CPU mixers (attention over the fp16 cache) take
  17 ms of a 68 ms token.
- **Prompt time** includes the prefill (bfp16 GEMMs, 128 per chunk pass),
  the MTP cache over the prompt and the first decode step: ~1,100-1,270
  prompt tokens per second from 1k to 9k.
- **Load:** the decoder packs and uploads in 1.7-3.0 s, the prefill's bfp16
  weights in 7.4 s (`qwen35_4b_npu_prefill_test`). Device memory, computed
  from the packed sizes (not measured): ~2.5 GB of q4nx decode weights plus
  ~4 GB of bfp16 prefill weights.

## Against the 2B

| | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|
| Weights read per token | ~1.09 GB | ~2.42 GB |
| Decode, short, no draft | 38.8-39.0 tok/s | 18.8 |
| Decode, fib, MTP | 63.7-63.9 | 30.6 |
| Prompt, ~1k | 0.31 s | 0.84 s |
