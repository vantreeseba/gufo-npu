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

In the container, publish the port, give it the NPU and mount the model
read-only. The image's entrypoint runs `gufo-npu`, so the command starts
with `serve`; the installed kernels are found by default:

```sh
docker run -d --name gufo-npu-serve --device /dev/accel/accel0 \
  --ulimit memlock=-1:-1 -p 8080:8080 \
  -v /path/to/MODEL.gguf:/models/model.gguf:ro \
  ghcr.io/vantreeseba/gufo-npu:main \
  serve --model /models/model.gguf --host 0.0.0.0
docker inspect -f '{{.State.Health.Status}}' gufo-npu-serve   # healthy once loaded
```

The image's health check polls `/health` on port 8080 (set `GUFO_NPU_PORT`
with `-e` when serving on another port). Add
`-v /opt/xilinx/xrt:/opt/xilinx/xrt:ro` when the host's amdxdna driver
needs the host's XRT (see the Dockerfile).

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model GGUF` | (required) | The model: Qwen3.5-2B, Qwen3.5-4B or Qwen3.6-35B-A3B (Unsloth UD-Q4_K_XL). Repeat it to serve several, one loaded at a time ([Several models](#several-models)). |
| `--host ADDR` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` to accept other machines. |
| `--port N` | `8080` | Port. |
| `--device npu\|cpu` | `npu` | `cpu` runs the fp32 CPU reference (slow; for checks). |
| `--kernels DIR` | installed kernels | NPU kernel directory, as for `generate`. |
| `--model-id NAME` | from the GGUF | The model name `/v1/models` lists and replies carry. Only with a single `--model`; several models keep their GGUFs' ids. |
| `--max-queue N` | `8` | Requests that may wait behind the running one; more get 429. |
| `--api-key KEY` | none | When set, `/v1/*` needs `Authorization: Bearer KEY`. `/health` and `/metrics` stay open. |
| `--context-length N` | 32,768 | Most prompt + generated tokens per request, up to the model's trained 262,144. Memory per token: [docs/serve/ENGINE.md](serve/ENGINE.md#context-length). |
| `--prompt-cache N` | `2` | Checkpoints of earlier prompts kept for multi-turn reuse; `0` turns the cache off. |
| `--draft mtp\|none` | `mtp` | MTP speculative decoding (when the GGUF has the MTP layer). Output is the same either way; `none` is for comparisons. |
| `--preload ID` | the first `--model` | The model loaded at start. |
| `--unload-after SECONDS` | `0` (never) | Unload a model that has been idle this long, freeing its device and pinned memory; the next request loads it again. |

`SIGINT` or `SIGTERM` stops the server.

## Endpoints

### `POST /v1/chat/completions`

The OpenAI chat completions API, streaming (`"stream": true`, server-sent
events) and not.

Request fields:

| Field | Support |
|---|---|
| `messages` | `system`, `user`, `assistant` and `tool` messages, rendered with the model's chat template. A final `assistant` message is continued (prefill). |
| `model` | With one model, accepted and ignored: replies carry the model's id. With several, the model to run (404 `model_not_found` if unknown; the loaded model when absent). |
| `tools`, `tool_choice` | Function tools. `tool_choice` may be `"auto"` (default), `"none"`, `"required"` (the reply is one or more tool calls) or `{"type": "function", "function": {"name": ...}}` (calls to that function). Forced calls are constrained token by token; see [Structured outputs](#structured-outputs). |
| `max_completion_tokens`, `max_tokens` | Most generated tokens (the first wins). Default and ceiling: the rest of the context. |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed` | Sampling. `temperature` 0 (the default) is greedy decoding. |
| `stop` | Up to 16 strings. Matched in the generated text, also across tokens; the stop string itself is never sent. |
| `stream`, `stream_options.include_usage` | Streaming, and a final usage chunk. |
| `chat_template_kwargs.enable_thinking`, `reasoning_effort` | Thinking (a think block before the answer, returned as `reasoning_content`). Off by default. `reasoning_effort` `none` or `minimal` turns it off, `low`, `medium` or `high` on; `enable_thinking` wins when both are set. |
| `n` | 1 to 8 choices, generated one after another (later ones reuse the prompt cache). With `seed`, choice *i* samples with `seed + i`, so choices differ and repeat. |
| `logprobs`, `top_logprobs` | Each generated token's log probability and up to 20 alternatives (see below). |
| `logit_bias` | Token id to a bias in [-100, 100], added to the logit before sampling (greedy too): -100 bans a token, 100 all but forces it. Ids outside the vocabulary are refused. |
| `parallel_tool_calls` | `false` ends the reply after its first tool call (anything after it is dropped). Default `true`. |
| `response_format` | `{"type": "text"}` (default), `{"type": "json_object"}` (any JSON object) or `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}` (JSON matching the schema). See [Structured outputs](#structured-outputs). |

Other fields (`user`, `metadata`, ...) are ignored.

The reply:

- `choices[i].message` has `content`, `reasoning_content` when the model
  thought, and `tool_calls` (`{"id": "call_...", "type": "function",
  "function": {"name", "arguments"}}`, arguments as a JSON string).
  `content` is `null` when the reply is only tool calls.
- `finish_reason` is `stop` (end of turn or a stop string), `length`
  (`max_tokens` or the context) or `tool_calls`.
- `usage` has `prompt_tokens` (the prompt counted once, whatever `n`),
  `completion_tokens` (all choices), `total_tokens` and
  `prompt_tokens_details.cached_tokens` (prompt tokens restored from a
  prompt-cache checkpoint, for the first choice).
- `choices[i].logprobs` is `null` unless `logprobs` is true; then it is
  `{"content": [{"token", "logprob", "bytes", "top_logprobs": [{"token",
  "logprob", "bytes"}, ...]}, ...]}`, one entry per generated token (the
  end-of-turn token has none). `token` is the token's text (bytes that are
  not whole UTF-8 characters show as U+FFFD; `bytes` has them exactly).

Streaming sends `chat.completion.chunk` events: first `{"role":
"assistant", "content": ""}`, then `content`, `reasoning_content` and
`tool_calls` deltas as they are generated, then an empty delta with
`finish_reason`; with `n` > 1 each choice does this in turn, its chunks
carrying its `index`. Then the usage chunk if asked for, and `data:
[DONE]`. Text is sent in whole UTF-8 characters. A tool call is sent in
OpenAI's fragment shape once the model has finished writing it: a delta
with its `index`, `id`, `type` and `function.name` (empty `arguments`), then
one with the `arguments`. With `logprobs`, each chunk carries the entries of
the tokens generated since the previous chunk (`logprobs.content`); entries
of text held back (a possible stop string or markup) ride on a later chunk.

#### Log probabilities

`logprob` is the model's own log-softmax over the whole vocabulary for that
position: before `temperature`, `top_k`/`top_p`/`min_p`, penalties and
`logit_bias`, as OpenAI reports them. So the values do not depend on the
sampling settings, and at temperature 0 the chosen token is the first of
`top_logprobs`. With MTP drafting, an accepted draft token's values come
from the verify pass that accepted it: the same distribution decoding
without drafts samples from, up to rounding. Computing them costs one pass over the vocabulary per token,
only when asked for.

### `POST /v1/completions`

The legacy text-completion API. The prompt goes to the model as it is,
without the chat template.

| Field | Support |
|---|---|
| `prompt` | A string, an array of strings, an array of token ids, or an array of such arrays. Each prompt gets `n` choices: choice `index` = prompt × `n` + choice; at most 8 in all. |
| `max_tokens` | Default 16 (OpenAI's for this API); ceiling the rest of the context. |
| `temperature`, `top_p`, `top_k`, `min_p`, penalties, `seed`, `logit_bias`, `stop`, `n`, `stream`, `stream_options.include_usage` | As for chat completions. `best_of` must equal `n`. |
| `echo` | Puts the prompt before the generated text. |
| `logprobs` | 0 to 5: the legacy shape, `{"tokens", "token_logprobs", "top_logprobs": [{token: logprob}], "text_offset"}`. Refused together with `echo` (prompt tokens have no logprobs here). |
| `suffix` | Refused (400). |

Generation also ends at the chat template's end-of-turn tokens
(`<|im_end|>`, `<|endoftext|>`). The reply is `{"object": "text_completion",
"choices": [{"text", "index", "logprobs", "finish_reason"}], "usage"}`;
streaming sends the same shape per piece of text, then `data: [DONE]`.

### Structured outputs

`response_format` json_object / json_schema and a forced `tool_choice`
constrain sampling: before each token, the tokens that cannot continue a
valid reply are masked out (their logits count as -inf, before penalties,
temperature and the filters), so the reply always parses unless it is cut
off by `max_tokens` (`finish_reason: "length"`).

- **JSON schema keywords enforced:** `type` (one or a list), `properties`,
  `required`, `additionalProperties` (boolean or a schema), `items`,
  `minItems`, `maxItems`, `minLength`, `maxLength` (characters), `enum`,
  `const`, `anyOf`, `oneOf` (as anyOf), `allOf` with one schema, and `$ref`
  to `#`, `#/$defs/...` or `#/definitions/...` (recursion included). Not
  enforced: `pattern`, `format`, `minimum`/`maximum`, `multipleOf`,
  `uniqueItems` and other keywords; they are accepted and ignored. With
  `strict: true`, an object without `additionalProperties` allows no other
  keys (OpenAI's strict mode); without it, other keys are allowed, as JSON
  Schema says. A schema the server cannot compile (an unresolved `$ref`, an
  empty `enum`, an unknown type, more than 128 properties or enum values)
  is a 400 with `param: "response_format"`.
- **Forced tool calls** follow the model's tool-call markup
  (`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
  ...</function>\n</tool_call>`): only the allowed function names, only
  declared parameters (any, when a function declares none), every required
  parameter before `</function>`. A string parameter is free text (it may
  not contain `</parameter>` or `</tool_call>`), a string enum one of its
  values, other types JSON matching their schema. More calls may follow,
  separated by a newline.
- **Thinking:** with thinking on, the think block is free text; the
  constraint applies after `</think>`.
- **MTP drafts** stay exact: a draft is kept only when the masked sample
  equals it, so a draft the grammar refuses is never kept.
- **Cost:** the mask is found by walking a trie of the vocabulary's bytes
  from the grammar's state; inside strings and free text only the tokens
  that could end them are walked. `serve_structured_test` prints the cost
  per token on a 248k-token vocabulary. JSON whitespace is limited to 48
  bytes in a row, integers to 18 digits (they fit in 64 bits), other
  numbers to 24 digits and a two-digit exponent, and nesting to 32 levels,
  so a constrained reply cannot run on in whitespace or digits. Strings,
  keys and raw parameter text are valid UTF-8.

### `GET /v1/models`, `GET /v1/models/{id}`

The served models. With several, each carries an extension field
`"loaded": true|false` (whether it is on the device now).

### `GET /health`

`{"status": "ok", "model": "..."}` once the model is loaded (the server
listens only after loading). With several models, `model` is the loaded
one (`null` after an idle unload) and `loading` the one loading now, if
any.

### `GET /metrics`

Prometheus text: requests by HTTP status (499 counts clients that hung up),
prompt, cached and generated tokens, engine prefill and decode seconds,
decode steps and accepted MTP drafts, the queue (`gufo_npu_queue_waiting`,
`gufo_npu_generation_running`), and histograms of time to first token and
request duration (both from arrival, queue included). With several models,
also per model: `gufo_npu_model_loaded` and `gufo_npu_model_loading`
(gauges), and `gufo_npu_model_loads_total`, `gufo_npu_model_unloads_total`
and their `_seconds_total`.

## Several models

```sh
gufo-npu serve --model 2b.gguf --model 4b.gguf --model 35b.gguf \
  --preload qwen3.6-35b-a3b --unload-after 600
```

- **Ids** come from each GGUF (`qwen3.5-2b`, `qwen3.5-4b`,
  `qwen3.6-35b-a3b`); two models with the same id are refused.
- **One model on the NPU at a time.** A request for another model first
  unloads the current one, releasing its device buffers and pinned memory
  (the 35B pins ~22 GB), then loads the requested one. The switch is part
  of that request's time to first token: ~4 s for the 2B, ~12 s for the 4B
  and ~19 s for the 35B (their load times).
- **Queue order** stays arrival order (FIFO) whatever model a request
  names, so a client never waits behind later requests. Alternating
  between models pays a load per switch; clients that mix models heavily
  are better served by one server per model.
- **Before loading**, the server already knows each model's tokenizer and
  limits (read from its GGUF at start, no weights), so requests are
  checked and tokenized without a load.
- Each model's prompt cache is dropped when it unloads.
- `--unload-after SECONDS` also works with one model, to free its memory
  while idle.

[docs/examples/docker-compose.yml](examples/docker-compose.yml) runs all
three models from one container (and has one service per model as the
alternative).

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

**Server overhead:** HTTP adds nothing measurable. The first token arrives
as fast as through `gufo-npu generate`, and decode runs at the same tok/s
(`tools/serve/serve_bench.py`, median of 3 after a warm-up, `--prompt-cache
0`):

| Model | Short prompt (24 tokens): HTTP | In process | ~1k prompt (932 tokens): HTTP | In process |
|---|---|---|---|---|
| Qwen3.5-2B | 0.089 s, 53.8 tok/s | 0.09 s, 53.6 tok/s | 0.284 s, 55.3 tok/s | 0.29 s, 55.4 tok/s |
| Qwen3.5-4B | 0.193 s, 27.6 tok/s | 0.33 s*, 27.6 tok/s | 0.643 s, 25.8 tok/s | 0.79 s*, 26.0 tok/s |
| Qwen3.6-35B-A3B | 0.243 s, 27.3 tok/s | 0.24 s, 26.0 tok/s | 1.046 s, 26.2 tok/s | 1.08 s, 26.2 tok/s |

\* `generate`'s prompt time for the 4B also counts work after the first
token.

**Second turn of a chat** (~1k tokens of history), time to the first token:

| Model | Restored from a checkpoint | Without one |
|---|---|---|
| Qwen3.5-2B | 0.10 s | 0.29 s |
| Qwen3.5-4B | 0.22 s | 0.67 s |
| Qwen3.6-35B-A3B | 0.31 s | 1.13 s |

**Soak** (`tools/serve/serve_soak.py`):
- **Traffic:** 2–4-turn conversations with 0.1k–4k tokens of history; 70% streamed, 30% sampled, 10% of streams hung up.
- **Memory:** anonymous memory grows only when a longer prompt than any before arrives, then stays flat (see docs/serve/ENGINE.md, "Memory between requests").

| Model | Run | Requests | Errors | TTFT on a cache hit / miss | Decode by window | Anonymous memory: start, peak, end |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 30 min | 518 | 0 | 0.33–0.35 s / 1.5–2.3 s | 25.1, 25.4, 25.5 tok/s (10 min each) | 0.72, 2.24, 2.24 GB (flat from 10 min) |
| Qwen3.5-4B | 5 min | 80 | 0 | 0.24–0.25 s / 1.6 s | 25.8, 25.9 tok/s | 0.45, 1.95, 1.95 GB (flat from 3 min) |
| Qwen3.5-2B | 5 min | 168 | 0 | 0.12 s / 0.62–0.66 s | 52.7, 51.2 tok/s | 0.33, 1.12, 1.12 GB |

Before buffer reuse, the same traffic grew without bound: the 35B reached
8.8 GB at 30 minutes, the 4B 5.9 GB and the 2B 3.3 GB at 5 minutes.

**Container:** the runtime image (`docker build .`) serves the 2B with the
`docker run` line above. It reports healthy 15 s after start and passes all
18 checks.

## Testing a server

`tools/serve/` drives a running server through the official `openai` Python
client (`pip install openai httpx`). Each takes `--base-url` (default
`http://127.0.0.1:8080`) and `--api-key` when the server has one:

- `serve_e2e.py`: 18 checks, printing one line each and exiting 1 on a
  failure:
  - `/health`, `/v1/models` and `/metrics`
  - chat, streamed and not
  - thinking on and off
  - a tool call round trip
  - seeded sampling and a stop string
  - a disconnect mid-stream (499)
  - a full queue (429)
  - a second turn served from the prompt cache

  Start the server with `--max-queue 1` for the queue check.
- `serve_bench.py`: warm time to the first token and decode tok/s for a
  short and a ~1k-token prompt. Start the server with `--prompt-cache 0`.
  With `--gufo-npu BIN --model GGUF` it also times `gufo-npu generate` on
  the same prompts (`--skip-http` times only that).
- `serve_soak.py --minutes N`: mixed multi-turn traffic (streamed, sampled
  and hung-up requests). It prints time to the first token and decode tok/s
  per `--window` seconds. With `--server-pid` (on the server's machine) it
  also prints the server's anonymous memory over time; `--out` keeps every
  record as JSON.

## Not supported yet

- More than one model on the NPU at once (models take turns;
  [Several models](#several-models)).
- Tool-call arguments streamed while the model writes them (a call is sent
  once complete).
- Images and audio.
- The `/v1/responses` API; embeddings.
- TLS (put a reverse proxy in front for that) and CORS headers.
