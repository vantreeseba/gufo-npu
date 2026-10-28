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

## TTFT quick wins (2026-10-28)

`gufo-npu bench prefill` (the 4B now runs on its own decoder and prefill
there): the first N tokens of one long chat prompt, each length once to warm
up and then three runs, back to back so the NPU never idles into its 5 s
autosuspend. TTFT is SpeculativeGenerate's prompt phase (prefill, MTP cache,
first token, first draft). Two interleaved rounds of three builds, then the
final build, alone on the NPU:

| Prompt tokens | main | + AVX-512 DeltaNet | + first token from prefill | + chunk planner (final) |
|---|---|---|---|---|
| 35 | 0.237, 0.236 s | 0.235, 0.236 | 0.188, 0.189 | **0.188** (−48 ms against main) |
| 1,024 | 0.719, 0.727 | 0.709, 0.710 | 0.661, 0.662 | **0.660** (−63 ms) |
| 4,096 | 2.930, 2.929 | 2.886, 2.915 | 2.854, 2.837 | **2.844** (−86 ms) |
| 9,000 | 7.930, 7.880 | 7.799, 7.841 | 7.772, 7.717 | **7.787** (−118 ms) |

- **First token from the prefill.** PrefillPrompt prefills the whole
  prompt; the first token comes from the LM head alone on its last row
  (`NpuDecoder::Logits`) instead of a decode step for the last prompt token
  (~57 ms): −47 ms. Quality in QUALITY.md ("First token from the prefill").
- **AVX-512 delta rule** in this model's prefill DeltaNet (its own copy of
  the mixers), bit-identical: DeltaNet at 9k 1.26 → 1.19 s, 1% of TTFT.
- **Short prompts:** the crossover stays at 7 tokens. With the prefill path
  forced on, 6 tokens take 0.177 s on the decode path against 0.179 s
  through prefill, 7 tokens 0.230 against 0.186. Through the prefill, TTFT
  is ~0.18 s for any prompt up to 12 tokens.

### Chunk plan

The prefill picks its chunks (one or two lanes of 256, 512 or 1,024 rows)
for the least estimated time. It used the 2B's model: GEMMs at 11.0, 14.5,
17.0 TFLOPS and 0.14 ms of CPU work per token. Measured on the 4B, one lane
of exactly one chunk (planner forced, 3 warm runs, 7.13 GFLOP per token):

| Chunk rows | GEMM time | TFLOPS | CPU time (DeltaNet, attention, other, syncs) | CPU per token |
|---|---|---|---|---|
| 256 | 0.160 s | 11.4 | 0.076 s | 0.30 ms |
| 512 | 0.260 s | 14.0 | 0.143 s | 0.28 ms |
| 1,024 | 0.454 s | 16.1 | 0.295 s | 0.29 ms |

So the 4B's CPU side is twice the 2B's per token (wider DeltaNet: 32 value
heads, and hidden 2560). With these constants (0.29 ms, 11.4/14.0/16.1)
the planner runs two lanes earlier (300 tokens: two lanes of 256 instead of
one of 512; 600: two of 512 instead of 256 + 344 one lane at a time).
Against the 2B's constants on the same build (with the two changes above),
3 warm runs each; a sweep of the CPU constant from 0.14 to 0.8 ms per token
found 0.25-0.35 ms best, and 0.5 ms and above slower at 2k and longer:

| Prompt tokens | 2B's constants | 4B's constants |
|---|---|---|
| 300 | 0.355 s | **0.329** (−7%) |
| 600 | 0.606 | **0.562** (−7%) |
| 800 | 0.602 | 0.606 |
| 1,024 | 0.662 | 0.665 |
| 1,500 | 1.082 | **1.042** (−4%) |
| 2,048 | 1.283 | 1.292 |
| 4,096 | 2.826 | 2.856 |
| 9,000 | 7.771 | 7.781 |

From 800 tokens up both pick the same passes; the differences there are
run-to-run noise (the final build: 1,024 0.660 s, 4,096 2.844 s).

Outputs: with the min/max fit the table above was measured on, the sky
prompt without drafting (`2eb62ce4`) and the Fibonacci prompt with MTP
(`e5fd099a`) kept their hashes after all three changes; decode speed
unchanged (18.8 and 30.8 tok/s). After merging main's search fit for the
4B (#15): sky `f99190db` (main's), Fibonacci `a965fad4` (main: `64f4400a`),
MTP equal to plain greedy for both, and TTFT the same (35 tokens 0.188 s,
1,024 tokens 0.667 s).

## Against the 2B

| | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|
| Weights read per token | ~1.09 GB | ~2.42 GB |
| Decode, short, no draft | 38.8-39.0 tok/s | 18.8 |
| Decode, fib, MTP | 63.7-63.9 | 30.6 |
| Prompt, ~1k (`generate`) | 0.31 s | 0.84 s |
| TTFT, 1,024 tokens (`bench prefill`, warm, 2026-10-28) | 0.29 s | 0.66 s |
