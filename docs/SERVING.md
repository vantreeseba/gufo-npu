# Serving

`gufo-npu serve` runs one model behind an OpenAI-compatible HTTP API, so
clients that talk to OpenAI (or to FastFlowLM through Lemonade) can point at
it instead.

```sh
gufo-npu serve --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# [loaded MODEL-ID in ... s]
# [serving http://127.0.0.1:8080/v1]

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "stream": true}'
```

In the container, publish the port and give it the NPU:

```sh
docker run --rm --device /dev/accel/accel0 --ulimit memlock=-1:-1 \
  -v /path/to/models:/models -p 8080:8080 ghcr.io/vantreeseba/gufo-npu:main \
  gufo-npu serve --model /models/MODEL.gguf --host 0.0.0.0
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model GGUF` | (required) | The model: Qwen3.5-2B, Qwen3.5-4B or Qwen3.6-35B-A3B (Unsloth UD-Q4_K_XL). |
| `--host ADDR` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` to accept other machines. |
| `--port N` | `8080` | Port. |
| `--device npu\|cpu` | `npu` | `cpu` runs the fp32 CPU reference (slow; for checks). |
| `--kernels DIR` | installed kernels | NPU kernel directory, as for `generate`. |
| `--model-id NAME` | from the GGUF | The model name `/v1/models` lists and replies carry. |
| `--max-queue N` | `8` | Requests that may wait behind the running one; more get 429. |
| `--api-key KEY` | none | When set, `/v1/*` needs `Authorization: Bearer KEY`. `/health` and `/metrics` stay open. |
| `--context-length N` | 32,768 | Most prompt + generated tokens per request, up to the model's trained 262,144. Memory per token: [docs/serve/ENGINE.md](serve/ENGINE.md#context-length). |
| `--prompt-cache N` | `2` | Checkpoints of earlier prompts kept for multi-turn reuse; `0` turns the cache off. |

`SIGINT` or `SIGTERM` stops the server.

## Endpoints

### `POST /v1/chat/completions`

The OpenAI chat completions API, streaming (`"stream": true`, server-sent
events) and not.

Request fields:

| Field | Support |
|---|---|
| `messages` | `system`, `user`, `assistant` and `tool` messages, rendered with the model's chat template. A final `assistant` message is continued (prefill). |
| `model` | Accepted and ignored: the server runs one model, and replies carry its id. |
| `tools`, `tool_choice` | Function tools. `tool_choice` may be `"auto"` (default) or `"none"`; `"required"` and named functions are refused. |
| `max_completion_tokens`, `max_tokens` | Most generated tokens (the first wins). Default and ceiling: the rest of the context. |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed` | Sampling. `temperature` 0 (the default) is greedy decoding. |
| `stop` | Up to 16 strings. Matched in the generated text, also across tokens; the stop string itself is never sent. |
| `stream`, `stream_options.include_usage` | Streaming, and a final usage chunk. |
| `chat_template_kwargs.enable_thinking`, `reasoning_effort` | Thinking (a think block before the answer, returned as `reasoning_content`). Off by default. `reasoning_effort` `none` or `minimal` turns it off, `low`, `medium` or `high` on; `enable_thinking` wins when both are set. |
| `n` | Only 1. |
| `response_format` | Only `{"type": "text"}`. |
| `logprobs`, `top_logprobs` | Not supported (400). |

Other fields (`user`, `metadata`, `parallel_tool_calls`, ...) are ignored.

The reply:

- `choices[0].message` has `content`, `reasoning_content` when the model
  thought, and `tool_calls` (`{"id": "call_...", "type": "function",
  "function": {"name", "arguments"}}`, arguments as a JSON string).
  `content` is `null` when the reply is only tool calls.
- `finish_reason` is `stop` (end of turn or a stop string), `length`
  (`max_tokens` or the context) or `tool_calls`.
- `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens` and
  `prompt_tokens_details.cached_tokens` (prompt tokens restored from a
  prompt-cache checkpoint).

Streaming sends `chat.completion.chunk` events: first `{"role":
"assistant", "content": ""}`, then `content`, `reasoning_content` and
`tool_calls` deltas as they are generated (a tool call arrives whole, in one
delta with its `index`), then an empty delta with `finish_reason`, the usage
chunk if asked for, and `data: [DONE]`. Text is sent in whole UTF-8
characters.

### `GET /v1/models`, `GET /v1/models/{id}`

The one loaded model.

### `GET /health`

`{"status": "ok", "model": "..."}` once the model is loaded (the server
listens only after loading).

### `GET /metrics`

Prometheus text: requests by HTTP status (499 counts clients that hung up),
prompt, cached and generated tokens, engine prefill and decode seconds,
decode steps and accepted MTP drafts, the queue (`gufo_npu_queue_waiting`,
`gufo_npu_generation_running`), and histograms of time to first token and
request duration (both from arrival, queue included).

## Errors

Errors use the OpenAI shape, `{"error": {"message", "type", "param",
"code"}}`:

| Status | When |
|---|---|
| 400 | Invalid JSON or fields, unsupported options, messages the template cannot take, or a prompt as long as the context (`code: context_length_exceeded`). |
| 401 | `--api-key` is set and the request lacks it (`code: invalid_api_key`). |
| 404 | Unknown endpoint or model id. |
| 429 | `--max-queue` requests are already waiting (`code: queue_full`). |
| 500 | The engine failed. In a stream, sent as a final `data:` event. |

## How requests run

- The NPU runs one generation at a time. Requests wait in arrival order;
  `--max-queue` bounds the wait list.
- A client that disconnects, while waiting or while its reply is generated,
  gives up its place or stops its generation at the next token.
- The engine keeps checkpoints of earlier prompts (`--prompt-cache`, the
  history through the last user message). The next turn of a chat restores
  one and processes only the previous reply and the new message
  (`usage.prompt_tokens_details.cached_tokens`). Least recently used
  checkpoints go first, so `N` conversations can alternate without losing
  their history.

## Measured (2026-10-28)

On the box, warm, greedy with MTP unless noted, the server driven by the
official `openai` Python client (3.40).

**End to end, all three models:** 18/18 checks each. They cover:
- `/health`, `/v1/models` and `/metrics`.
- Chat, streaming and not: the stream equals the non-streamed reply.
- Thinking on (`reasoning_content`) and off.
- A tool call, streamed and not, then a tool result and the final answer.
- Seeded sampling: the same seed repeats and another differs.
- A stop string.
- Hanging up mid-stream: counted as 499, and the next request is served in
  0.2–0.5 s.
- Queue full: 429.
- A second turn reusing 929 of 993 prompt tokens.

**Server overhead:** HTTP adds nothing measurable. The first token arrives in
the same time as `gufo-npu generate`'s prompt phase, and decode runs at the
same tok/s (median of 3 after a warm-up; `--prompt-cache 0`):

| Model | Short prompt (24 tokens): TTFT, decode | ~1k prompt (932 tokens): TTFT, decode | In process, ~1k |
|---|---|---|---|
| Qwen3.5-2B | 0.093 s, 53.4 tok/s | 0.293 s, 55.4 tok/s | 0.29 s, 55.3 tok/s |
| Qwen3.5-4B | 0.204 s, 27.6 tok/s | 0.661 s, 25.6 tok/s | 0.79 s*, 26.0 tok/s |
| Qwen3.6-35B-A3B | 0.255 s, 26.6 tok/s | 1.088 s, 25.7 tok/s | 1.11 s, 25.5 tok/s |

\* `generate`'s prompt time for the 4B also counts work after the first
token.

**Second turn of a chat** (~1k tokens of history), time to the first token:

| Model | Restored from a checkpoint | Without one |
|---|---|---|
| Qwen3.5-2B | 0.10 s | 0.29 s |
| Qwen3.5-4B | 0.22 s | 0.67 s |
| Qwen3.6-35B-A3B | 0.31 s | 1.13 s |

**Soak** (mixed load: 2–4-turn conversations with 0.1k–4k
tokens of history, 70% streamed, 30% sampled, 10% of streams hung up):

| Model | Run | Requests | Errors | TTFT on a cache hit / miss | Decode | Anonymous memory |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 30 min | 513 | 0 | 0.33–0.35 s / 1.5–2.4 s | 25.1–25.3 tok/s in every 10 min | 0.7 → 8.8 GB* |
| Qwen3.6-35B-A3B | 15 min, with the fix | 235 | 0 | 0.41–0.43 s / 2.0–2.7 s | 24.6–25.0 tok/s | 0.7 → 1.25 GB (peak 1.55) |
| Qwen3.5-4B | 5 min | 79 | 0 | 0.26 s / 1.6–1.7 s | 25.5 tok/s | 0.45 → 5.9 GB* |
| Qwen3.5-2B | 5 min, with the fix | 163 | 0 | 0.15 s / 0.65–0.76 s | 50.5–52.5 tok/s | 0.33 → 0.68 GB (peak 0.75) |

\* Before the fix: glibc kept the prompt buffers of every new mix of
lengths in its heaps. The engine now hands free heap pages back between
requests once more than 1 GiB is unused (`ReleaseFreeMemory` in
`src/serve/engine.cpp`). On a fixed second-turn prompt this costs nothing
(2B: 0.104 s against 0.103 s). The runs' cache-hit times differ with their
conversations' history lengths.

## Not supported yet

- More than one model per process, or switching models per request.
- `n > 1`, log probabilities, JSON mode and structured outputs, forced tool
  calls (`tool_choice: "required"` or a named function).
- Images and audio.
- The legacy `/v1/completions` and the `/v1/responses` APIs; embeddings.
- TLS (put a reverse proxy in front for that) and CORS headers.
