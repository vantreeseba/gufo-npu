# Qwen3.5-4B quality

| Check | Tolerance | Measured |
|---|---|---|
| CPU reference vs llama.cpp (CPU), teacher-forced logits | as the 35B's: end to end rel. L2 < 0.15 per prompt, argmax >= 95%; layer by layer and LM head (below) | 293/301 end to end (llama.cpp vs itself: 293/301), LM head 301/301 |
| CPU decoder vs reference | logits rel. L2 < 2e-3, argmax all | 1.9e-4, 17/17 |
| Each NPU op vs CPU reference | relative error < 1e-2 | all 16 sequences, 8.2e-6 to 1.7e-3 (SwiGLU, bf16 output); nothing written past a partial round |
| NPU decode vs CPU decoder (same q4nx weights), teacher-forced | argmax >= 95%, logits rel. L2 < 1e-2 per prompt | 301/301, 3.4e-3 to 5.5e-3 |
| MTP speculative decode vs plain greedy (NPU) | identical | identical (48 tokens; 81% of drafts accepted), also after prefill |
| NPU prefill vs NPU decode | hidden < 0.2, states < 0.4, logits < 0.15 (from the 4B's measurements; the 2B's are tighter), argmax >= 95% | 170/170; error grows with the prompt (below) |

There is no BF16 / HF oracle here: llama.cpp on the same GGUF is the
independent implementation, as for Qwen3.6-35B-A3B (whose test,
`reference_llama_test`, the 4B's is a copy of, dense only). The 2B already
pinned the architecture against HF transformers.

## CPU reference vs llama.cpp

`tests/models/qwen35_4b/reference_llama_test.cpp` (label `qwen35_4b;model`).
Oracle: `tools/reference/qwen36_llama_oracle.py` (llama.cpp
0c1e57098bba43ac29e6e3b677cdceebdd22334f via llama-cpp-python 0.3.36, CPU)
over the same GGUF, the four prompts chat 17, code 15, prose 39 and long 230
tokens (301 positions). 2026-10-27:

| Prompt | End to end: logits rel. L2 (worst position) | Argmax | llama.cpp vs itself (one token per call): rel. L2, argmax | Layer contributions < 1e-5 | Worst | LM head rel. L2 |
|---|---|---|---|---|---|---|
| chat | 4.33e-2 (8.97e-2) | 17/17 | 4.44e-2, 17/17 | 451/544 (82.9%) | 1.4e-2 | 1.1e-7 |
| code | 3.83e-2 (6.40e-2) | 15/15 | 3.80e-2, 15/15 | 422/480 (87.9%) | 1.3e-2 | 1.1e-7 |
| prose | 2.27e-2 (3.86e-2) | 39/39 | 2.65e-2, 39/39 | 1138/1248 (91.2%) | 9.7e-3 | 1.1e-7 |
| long | 3.29e-2 (8.59e-2) | 222/230 | 3.37e-2, 222/230 | 5682/7360 (77.2%) | 1.5e-2 | 1.1e-7 |

- End to end the reference is as close to llama.cpp as llama.cpp is to
  itself: llama.cpp rounds every quantized matmul's input to 8 bits (see the
  35B's QUALITY.md); the reference keeps fp32.
- Layer by layer, on llama.cpp's own layer inputs with its rounding applied,
  the median contribution error is 2.5e-7; the lowest share of exact tokens
  in any layer over all prompts is 42.5% (layer 0; the test requires 10%).
- The eight long-prompt top-1 differences are near ties on one side or the
  other (top-2 gaps 0.004-0.17 logits).
- Tokenizer ids and the chat rendering match llama.cpp's; one token at a time
  reproduces the whole-prompt forward exactly (rel. err 0).
- Speed (double-precision reference, 16 threads): 4.3-5.1 tok/s whole
  prompt, 3.5 tok/s token by token; peak RSS 3.7 GB (2.7 GB of it the mapped
  GGUF).

This is the first dense model with more DeltaNet value heads (32) than key
heads (16); the tiled mapping (value head h reads key head h % 16, as for the
35B) is what matches.

## CPU decoder vs reference

`tests/models/qwen35_4b/decoder_test.cpp`: the fp32 `Decoder` (fp16 K/V
cache) against the reference, both over the q4nx weights on the CPU, token by
token over the chat prompt: worst logits rel. L2 1.93e-4 (2B: 6.2e-4),
argmax 17/17.

## NPU decode

`tests/models/qwen35_4b/decode_gemv_test.cpp` (labels
`qwen35_4b;npu;kernels`): every variant of the 4B's decode GEMV, one and two
tokens, alone and chained, against q4nx in double: relative error 8.2e-6 to
1.2e-5 for fp32 outputs, 1.6-1.7e-3 for SwiGLU (bf16 output); residuals
within 9.3e-8 of their scale. Hidden-sized outputs (2,560 rows, a partial
last round) leave a sentinel past their rows untouched.

`tests/models/qwen35_4b/npu_decode_test.cpp` (labels `qwen35_4b;npu;model`),
teacher forced over the oracle's four prompts, NPU against the CPU decoder
over the same q4nx weights, 2026-10-27:

| Prompt | Argmax vs CPU q4nx | Logits rel. L2 | Argmax vs llama.cpp (GGUF as stored) |
|---|---|---|---|
| chat (17) | 17/17 | 5.5e-3 | 16/17 |
| code (15) | 15/15 | 5.2e-3 | 15/15 |
| prose (39) | 39/39 | 3.4e-3 | 38/39 |
| long (230) | 230/230 | 4.5e-3 | 197/230 |
| all | 301/301 | | 266/301 (88.4%) |

The NPU adds only its bf16 inputs (as on the 2B); the 35 top-1 changes
against llama.cpp are q4nx's requantization of the GGUF's Q5_K, Q6_K (the
LM head among them) and Q8_0 tensors, as measured for the 35B
(qwen3.6-35b-a3b/QUALITY.md, "q4nx requantization"). MTP speculation (draft
over the 32k draft vocabulary, 2-token verify) gives exactly plain greedy's
48 tokens.

## NPU prefill

`tests/models/qwen35_4b/npu_prefill_test.cpp`: prefill (the 4B's bfp16
GEMM program, CPU mixers batched) against the decode path, same q4nx
weights, prompts of 1 to 2,049 tokens; states compared layer by layer, then
16 teacher-forced decode steps from both. 2026-10-27:

| Prompt | Hidden | Conv | DeltaNet state | K/V | MTP K/V | Logits | Argmax |
|---|---|---|---|---|---|---|---|
| chat (17) | 3.1e-2 | 2.6e-2 | 2.8e-2 | 2.0e-2 | 1.3e-2 | 1.9e-2 | 17/17 |
| code (15) | 1.9e-2 | 2.4e-2 | 2.1e-2 | 2.0e-2 | 1.3e-2 | 1.2e-2 | 17/17 |
| prose (39) | 1.8e-2 | 2.2e-2 | 2.4e-2 | 2.0e-2 | 1.3e-2 | 6.5e-3 | 17/17 |
| 2 | 4.4e-2 | 4.3e-2 | 5.7e-2 | 4.7e-2 | 3.9e-3 | 7.6e-3 | 17/17 |
| 256 | 5.1e-2 | 8.8e-2 | 4.8e-2 | 4.7e-2 | 1.4e-2 | 7.7e-2 | 17/17 |
| 257 | 5.1e-2 | 5.2e-2 | 4.7e-2 | 4.7e-2 | 1.4e-2 | 8.7e-2 | 17/17 |
| 600 | 9.8e-2 | 4.1e-2 | 1.0e-1 | 1.2e-1 | 1.4e-2 | 8.8e-2 | 17/17 |
| 1,100 | 1.2e-1 | 5.5e-2 | 9.2e-2 | 1.5e-1 | 1.4e-2 | 6.1e-2 | 17/17 |
| 2,049 | 1.3e-1 | 2.8e-1 | 1.1e-1 | 1.8e-1 | 1.4e-2 | 6.4e-2 | 17/17 |

The 2B's worst over the same prompts is about a third of this (hidden
4.1e-2, states 5.8e-2, logits 3.3e-2). The first token after the prompt is
the same every time, every decode step after prefill agrees on top-1
(170/170), speculative decoding after prefill equals plain greedy, and
greedy with and without prefill agree on all 48 tokens checked (the 2B: 26
of 48). The growth is open (EXPERIMENTS.md); the tolerances above are this
model's measurements with headroom, and a wrong offset or stale window still
gives errors of order 1.
