# Chat format fixtures

- `templates/`: the `tokenizer.chat_template` of the Qwen3.5-2B, Qwen3.5-4B
  and Qwen3.6-35B-A3B GGUFs (Unsloth UD-Q4_K_XL, the pinned revisions in the
  models' READMEs). Qwen's chat template (Apache-2.0) with Unsloth's
  developer-role and tool-calling fixes. Re-extract one with
  `tools/reference/qwen35_chat_template.py --extract MODEL.gguf OUT.jinja`.
- `chat_cases.json`: OpenAI-style requests (messages, tools, thinking).
  `reject: true` marks requests the server refuses before templating
  (images, audio).
- `chat_expected.json`: each case rendered by every template with jinja2
  (transformers' environment); `null` where the template or the server
  rejects it. All three templates give the same text, because the server
  always sets `enable_thinking` and never `preserve_thinking`, the only
  places they differ. Regenerate:

```sh
python3 tools/reference/qwen35_chat_template.py \
  --template tests/serve/fixtures/templates/qwen3.5-2b.jinja \
  --template tests/serve/fixtures/templates/qwen3.5-4b.jinja \
  --template tests/serve/fixtures/templates/qwen3.6-35b-a3b.jinja \
  --cases tests/serve/fixtures/chat_cases.json \
  --out tests/serve/fixtures/chat_expected.json
```
