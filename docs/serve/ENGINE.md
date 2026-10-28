# Engine and model registry

`src/serve/engine.hpp` is the server's view of a model: token ids in, token
ids out, one generation at a time. `LoadEngine` opens the GGUF, picks the
model package and loads it once; every request after that reuses the loaded
weights and NPU programs and starts from a fresh decode state.

## Model registry

`src/serve/model_registry.hpp` picks the package from the GGUF's
architecture and shapes:

| GGUF | Package | Decoder | Prompt processing |
|---|---|---|---|
| `qwen35`, hidden 2,048 | Qwen3.5-2B | family `qwen35::NpuDecoder` | family `qwen35::NpuPrefill` |
| `qwen35`, hidden 2,560 | Qwen3.5-4B | `qwen35_4b::NpuDecoder` | `qwen35_4b::NpuPrefill` |
| `qwen35moe`, hidden 2,048 | Qwen3.6-35B-A3B | family `qwen35::NpuDecoder` (MoE) | `qwen36_35b_a3b::NpuPrefill` (experts on the CPU) |

Other GGUFs are refused on the NPU (`--device cpu` runs any model the CPU
reference reads, greedy and slow; it is there for tests).

- **Model id:** the GGUF's `general.name`, lowercased with spaces as `-`
  (`qwen3.5-2b`, `qwen3.5-4b`, `qwen3.6-35b-a3b`), or `EngineOptions::model_id`.
- **MTP drafts:** used when the GGUF has its MTP layer (`ModelInfo::can_draft`)
  and the request asks for them (`GenerateRequest::draft`, the default).

## A request

- **Prompt:** the whole token list, every time. Each NPU model keeps a
  `qwen35::PromptCache` (`EngineOptions::prompt_cache` checkpoints, default
  2, 0: off; `gufo-npu serve --prompt-cache N`): a request restores the
  longest checkpoint its prompt starts with and processes only the rest
  (`GenerateResult::cached_prompt_tokens`). A checkpoint is taken before the
  prompt's last `<|im_start|>`, the history through the last user message,
  which the next turn of a chat starts with (`prompt_cache.hpp`). The CPU
  reference has no cache.
- **Ends:** at a stop id (`kStop`, the id is reported), at `max_tokens` or the
  context limit (`kLength`), or when the token callback returns false or the
  request's cancel flag is set (`kCancelled`). Cancelling is checked after
  each generated token; a prompt being processed runs to its end first.
- **Sampling:** `GenerateRequest::sampling` goes to the model's generate
  loop (`Sampler`, `src/core/sampling.hpp`); temperature 0 is greedy. With
  MTP drafts, sampling keeps the plain decoder's output distribution.
- **Refused** (`std::invalid_argument`): an empty prompt, token ids outside the
  vocabulary, `max_tokens` < 1, a prompt that fills the context limit, and
  sampling settings out of range.
- **Cancelling:** the engine's token callback returns false, and the generate
  loop ends after that token. Every decoder resolves its speculative pass
  before it reports tokens, and the next request starts from a fresh state
  (or a checkpoint), so a cancelled request leaves no trace. Checkpoints are
  taken while the prompt is processed, before any token is reported, so a
  request cancelled or failing later leaves only checkpoints of fully
  processed prompt prefixes (`serve_engine_test`).

## Context length

The decoders keep the K/V cache in host memory and grow it with the
sequence; the prefills run any prompt length in passes. So the limits are
memory and time, not a buffer size. All three GGUFs are trained to 262,144
tokens. The engine's default limit (prompt + generated tokens) is
`kDefaultContextLength`, 32,768, the length it is checked at below; raise it
with `EngineOptions::context_length` up to the trained length.

Host memory per token of context (K/V in fp16, the MTP layer's cache, and the
prompt's final residuals kept for the MTP cache), computed from the shapes;
KB and GB are KiB and GiB:

| Model | Attention layers | K/V per token | + MTP + residuals | At 32k | At 262k |
|---|---|---|---|---|---|
| Qwen3.5-2B | 6 of 24, 2 KV heads | 12 KB | 22 KB | 0.7 GB | 5.6 GB |
| Qwen3.5-4B | 8 of 32, 4 KV heads | 32 KB | 46 KB | 1.4 GB | 11.5 GB |
| Qwen3.6-35B-A3B | 10 of 40, 2 KV heads | 20 KB | 30 KB | 0.9 GB | 7.5 GB |

Checked at ~31k tokens: two prompts of 30,630 and 30,930 tokens per model
through the engine (`gufo_npu_engine_soak --min-prompt 31000 --max-prompt
32500`, MTP drafts), box, 2026-10-28. Both ran to the end on every model,
including the 35B's two prefill passes (it runs 16,384 tokens per pass):

| Model | Time to first token | Decode after it | Anonymous memory after |
|---|---|---|---|
| Qwen3.5-2B | 18.9–19.0 s | 18–22 tok/s | 0.40 GB (0.32 after load) |
| Qwen3.5-4B | 49.5–50.5 s | 8.5–10.4 tok/s | 0.65 GB (0.44) |
| Qwen3.6-35B-A3B | 64.6–66.1 s | 11.8–14.4 tok/s | 1.17 GB (0.71) |

Prompts this long are slow (attention and DeltaNet on the CPU grow with the
prompt; BENCHMARKS.md per model), but they work; the default limit stays at
32,768 until longer ones are measured.

## Load once, generate many

`gufo_npu_engine_soak` (tools/serve) loads one engine and runs the
benchmarks' reference prompts (sky, Fibonacci, the ~1k prompt), then random
requests (prompts of 10–4,000 tokens, 16–256 tokens out, MTP drafts), then
the reference prompts again. Each request's output was then generated again
in a fresh `gufo-npu generate` process and the hashes compared. Box, warm,
under the shared NPU lock (taken per request), 2026-10-28.

| Model | Load | Requests in one engine | Same as a fresh process | Reference hashes before / after | Decode, first / last quarter | Anonymous memory: after load, first request, end |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 3.6 s | 56 | 53 of 53 | `a49f1c4c` `fdb595fb` `8edbce90`, unchanged | 50.1 / 52.6 tok/s | 0.32, 0.38, 0.41 GB |
| Qwen3.5-4B | 11.6 s | 56 | 53 of 53 | `f99190db` `a965fad4` `fbd339e4`, unchanged | 27.2 / 26.9 tok/s | 0.44, 0.53, 0.66 GB |
| Qwen3.6-35B-A3B | 18.8 s | 56 | 53 of 53 | `2c9f87d3` `9430fd17` `1de5ad9f`, unchanged | 25.9 / 24.9 tok/s | 0.71, 0.76, 0.96 GB |

(53 = 3 reference prompts + 50 random requests; the 3 repeats of the
reference prompts are compared with their first run.) The 35B's sky hash is
`2c9f87d3` on main since #13: its 24-token prompt now goes through the
prefill (CPU experts, from 6 tokens) instead of the decode path; the fresh
`gufo-npu generate` gives the same.

**30-minute soak, Qwen3.6-35B-A3B:** one engine, 448 random requests
(prompts of 17–3,989 tokens, mean 1,917; 21,104 tokens generated) between
the reference prompts, the NPU lock taken per request. No failures, no
errors in the logs, the reference outputs unchanged after it. Decode 24.3 tok/s in the
first quarter and 24.6 in the last, mean time to first token 2.18 s and
2.10 s. Anonymous memory reaches 0.96 GB within the first requests and stays
there (0.97 GB after 448): the buffers grow to the longest prompt seen, then
stop.

**Memory.** RSS after load counts the GGUF's mapped pages (read once to pack
the weights; the kernel can reclaim them) and the device buffers:

| Model | RSS after load | GGUF mapped | Anonymous | Device buffers (the rest) |
|---|---|---|---|---|
| Qwen3.5-2B | 4.32 GB | 1.29 GB | 0.32 GB | ~2.7 GB |
| Qwen3.5-4B | 9.71 GB | 2.79 GB | 0.44 GB | ~6.5 GB |
| Qwen3.6-35B-A3B | 41.86 GB | 21.28 GB | 0.71 GB | ~19.9 GB |

(GB here are GiB.) The 35B's ~19.9 GiB is ~21.3 GB: its decoder's device
buffers and the prefill's resident bfp16 matrices.

## Memory between requests

Prompt buffers are sized by each request's length, and glibc keeps freed
large buffers in its heaps, so a long-running server's anonymous memory
would grow with each new mix of lengths. Before the fix it reached 8.8 GB
after a 30-minute 35B soak, and 3.3 GB after 5 minutes on the 2B. After each
request, once the heaps hold more than 1 GiB unused, `malloc_trim` hands
their free pages back (`ReleaseFreeMemory`). The same soaks then stay at
1.25 GB (35B, 15 min) and 0.68 GB (2B). See docs/SERVING.md, "Measured".

## The CLI

`gufo-npu generate` and `bench prefill` keep their own model selection: they
print per-package profiles the engine does not expose.
