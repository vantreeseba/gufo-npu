# Qwen3.5-4B experiments

What was tried while bringing the 4B up, what it showed, and what is open.

## q4nx fit: search, not min/max (2026-10-27)

Retained: `qwen35_4b::kQ4nxFit` is `q4nx::Fit::kSearch` (llama.cpp's
weighted least-squares Q4_K fit) for the decoder, the MTP block, the draft
head and the prefill's weights; main packed them with min/max. Most of this
GGUF's matrices are Q5_K, Q6_K or Q8_0 and requantize. On the 35B's
held-out evaluation (1,515 positions of six pinned texts,
`tools/reference/fetch_text.py`; KL against the GGUF as stored, its
harness run on this model, 2026-10-27):

| Weights through q4nx | min/max: KL, top-1 | search: KL, top-1 |
|---|---|---|
| every matrix (what the NPU runs) | 0.0301, 91.82% | **0.0235, 93.14%** |
| every matrix but the tied LM head | 0.0230, 93.60% | 0.0187, 94.26% |

On the oracle prompts the set the NPU runs agrees with llama.cpp on 270 of
301 positions instead of 266, and `qwen35_4b_npu_decode_test` gives
269/301 instead of 266/301 (NPU against the CPU over the same weights
300/301). The prefill test still passes (170/170 after prefill, MTP equal to
greedy). Output hashes change, as the weights do: sky without a draft
`f99190db` (was `2eb62ce4`), Fibonacci with MTP `64f4400a` (was `e5fd099a`).
Speed is unchanged, 18.69-18.78 tok/s on sky and 31.4-31.6 on Fibonacci with
MTP. The decoder packs and uploads in 4.7 s. Importance weighting is the
next candidate (qwen3.6-35b-a3b/EXPERIMENTS.md).

## Decode GEMV geometry (2026-10-27)

The 2B's core program bakes its geometry in at compile time: the token
buffer holds two slots of [H | D] fp32 at hidden 2048 plus the norm weight
(36 KB), and the longest input is 6144. The 4B needs hidden 2560 and FFN
9216, so it gets its own copy of the design (`kernels/decode_gemv.py`,
xclbin `qwen35_4b_decode`) rather than a wider shared one; the 2B's
binaries do not change.

- **Core memory.** The 4B's token buffer is 46,080 B. With the 2B's 8 KB
  stack the core program would need ~66.8 KB of the 64 KB data memory. The
  toolchain measures the kernels' stack frames at 2,624 B (plus small
  soft-float helpers), so the copy reserves 4 KB. Layout of a core (tile
  1,3): stack 0-4 KB, the two weight buffers 4,096-13,568, one y buffer and
  y_acc to 14,080, the token buffer 16,384-62,464, the per-32 sums to 64,768,
  the other y buffer and the RTP words to 65,052.
- **Partial rounds.** A round is 8 columns x 4 rows x 32 rows per core =
  1,024 output rows. Hidden-sized outputs (mixer out, FFN down, eh_proj:
  2,560 rows) are 2.5 rounds: columns 0-3 run three rounds, 4-7 two, each
  core reading its round count from its RTP word, each column's weight fill
  and output drain covering only its rounds. The q4nx matrix stays padded to
  3,072 rows (the padding is never read), and nothing is written past row
  2,560, so the output can land in the next call's D slot. The op test
  (`qwen35_4b_decode_gemv_test`) fills every output buffer with a sentinel
  and checks it survives past the rows.
- **Weight transfer runs.** A column's slice per round is 4,736 x k_blocks
  words, not a multiple of 1,024 for the 4B's inputs (2,560, 5,120, 9,216):
  the copy splits it into the largest equal runs of at most 1,024 words
  (740, 740, 888 words).

## The two-token mixer output (2026-10-27)

The first NPU decode run matched the CPU on every top-1 token by token, but
MTP speculation diverged from plain greedy. Two-token passes gave the first
token's logits exactly and the second's with 0.3-0.6 relative error;
rejected drafts (only the first token kept) stayed exact. Every GEMV variant
passed its op test for two tokens, so the fault was on the host: the CPU
writes each token's mixer output into its slot of the output projection's
input and syncs a region of one slot plus a hidden vector. On the 2B the
mixer output (16 x 128, 8 x 256) is exactly the hidden size, so that covers
token 1; on the 4B it is 4,096 wide (32 value heads x 128, 16 heads x 256)
against a hidden of 2,560, and the last 1,536 values of token 1 were never
synced. The 4B decoder now syncs both slots in full; speculation then gives
plain greedy's output exactly.

The 2B's decoder has the same latent limit (it syncs one slot plus a hidden
vector) but its shapes never reach it. Qwen3.6-35B-A3B's do: its mixer
output is also 4,096 against a hidden of 2,048, so a two-token MoE step (MTP
verify, two-token prompt passes) would lose the same values.

## Prefill error growth: bfp16, not the batched path

The prefill (bfp16 GEMMs) against the decode path matches the 2B's accuracy
on short prompts (up to 39 tokens: hidden <= 3.1e-2, logits <= 1.9e-2) but
drifts about three times faster with prompt length: at 2,049 tokens hidden
1.3e-1, conv windows 2.8e-1, DeltaNet states 1.1e-1, K/V 1.8e-1, logits
8.8e-2 (2B: 4.1e-2, 4.4e-2, 5.8e-2, 4.4e-2, 3.3e-2).

The 4B is the first model to prefill with more DeltaNet value heads (32)
than key heads (16), so a batched-path bug was checked for by splitting the
error (a diagnostic build of the prefill whose GEMMs are exact fp32
projections over the same q4nx weights, against the fp32 CPU decoder token
by token; QUALITY.md has the table). With exact projections the states agree
to 2e-5 to 2e-4 at 256, 1,100 and 2,049 tokens, with no growth; with the
bfp16 GEMMs they reach 1.2-1.9e-1 at 2,049. The batched mixer path is right;
the growth is bfp16 rounding compounding over 32 layers of width 2560. The
end-to-end checks hold (decode after prefill agrees on every top-1, greedy
with and without prefill agree on all 48 tokens checked, where the 2B agrees
on 26), so `qwen35_4b_npu_prefill_test` keeps tolerances set from the 4B's
measurements. A tighter prefill would need a more precise GEMM, e.g. the
second, lo-part pass the 2B's prefill notes (about twice the GEMM time).
