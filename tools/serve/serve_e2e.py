"""End-to-end checks of `gufo-npu serve` through the official openai client.

Start the server with `--max-queue 1` (the queue check expects a 429 for
the third of three concurrent requests), then:

  python3 tools/serve/serve_e2e.py --base-url http://127.0.0.1:8080

Prints one line per check and a JSON summary; exits 1 on any failure.
Needs `pip install openai httpx`.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

import httpx
import openai

from textgen import filler

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--api-key", default="none")
parser.add_argument("--history-sentences", type=int, default=70,
                    help="filler sentences in the multi-turn check (~13 tokens each)")
args = parser.parse_args()
BASE = args.base_url.rstrip("/")
client = openai.OpenAI(base_url=BASE + "/v1", api_key=args.api_key, timeout=600)
results = []


def check(ok, what, **info):
    results.append({"ok": bool(ok), "what": what, **info})
    print(("ok   " if ok else "FAIL ") + what + (" " + json.dumps(info) if info else ""),
          flush=True)


def get(path):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": f"Bearer {args.api_key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode()


def stream(messages, **kw):
    """Streams a chat; returns (content, reasoning, tool_calls, finish, usage, ttft, chunks)."""
    t0 = time.perf_counter()
    ttft = None
    content, reasoning, finish, usage = "", "", None, None
    calls = {}
    n = 0
    s = client.chat.completions.create(messages=messages, model=MODEL, stream=True,
                                       stream_options={"include_usage": True}, **kw)
    for chunk in s:
        n += 1
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        rc = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get(
            "reasoning_content")
        piece = (d.content or "") + (rc or "")
        if (piece or d.tool_calls) and ttft is None:
            ttft = time.perf_counter() - t0
        content += d.content or ""
        reasoning += rc or ""
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"name": "", "arguments": "", "id": None})
            if tc.id:
                c["id"] = tc.id
            if tc.function and tc.function.name:
                c["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["arguments"] += tc.function.arguments
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    return content, reasoning, [calls[k] for k in sorted(calls)], finish, usage, ttft, n


# --- basics
status, body = get("/health")
check(status == 200, "/health", body=body[:80])
models = client.models.list()
MODEL = models.data[0].id
check(len(models.data) == 1, "/v1/models", id=MODEL)

sky = [{"role": "user", "content": "Explain in a few sentences why the sky is blue."}]
r = client.chat.completions.create(model=MODEL, messages=sky, max_tokens=96)
plain = r.choices[0].message.content or ""
check(len(plain) > 20 and r.usage.completion_tokens > 0, "non-streaming chat",
      finish=r.choices[0].finish_reason, tokens=r.usage.completion_tokens)
c, rs, _, fin, usage, ttft, _ = stream(sky, max_tokens=96)
check(c == plain and not rs, "streaming chat equals non-streaming (greedy)",
      ttft=round(ttft or -1, 3))
check(usage is not None and usage.completion_tokens == r.usage.completion_tokens,
      "stream usage chunk")

# --- thinking on / off
c, rs, _, fin, usage, _, _ = stream(
    [{"role": "user", "content": "What is 17 * 23? Think it through."}],
    max_tokens=400, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
check(len(rs) > 0, "thinking on: reasoning_content", reasoning_chars=len(rs),
      content_chars=len(c), finish=fin)
c, rs, _, fin, _, _, _ = stream(
    [{"role": "user", "content": "What is 17 * 23?"}], max_tokens=64)
check(not rs and len(c) > 0, "thinking off: no reasoning_content", content=c[:60])

# --- tool call round trip
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
r = client.chat.completions.create(model=MODEL, messages=msgs, tools=tools, max_tokens=200)
m = r.choices[0].message
ok = bool(m.tool_calls) and m.tool_calls[0].function.name == "get_weather"
call_args = json.loads(m.tool_calls[0].function.arguments) if ok else {}
check(ok and "paris" in str(call_args.get("city", "")).lower() and
      r.choices[0].finish_reason == "tool_calls", "tool call", args=call_args,
      finish=r.choices[0].finish_reason)
if ok:
    _, _, scalls, sfin, _, _, _ = stream(msgs, tools=tools, max_tokens=200)
    check(scalls and scalls[0]["name"] == "get_weather" and
          json.loads(scalls[0]["arguments"]) == call_args and sfin == "tool_calls",
          "tool call streamed")
    msgs2 = msgs + [m.model_dump(exclude_none=True),
                    {"role": "tool", "tool_call_id": m.tool_calls[0].id,
                     "content": "Sunny, 21 C, light wind."}]
    r2 = client.chat.completions.create(model=MODEL, messages=msgs2, tools=tools,
                                        max_tokens=120)
    ans = r2.choices[0].message.content or ""
    check("21" in ans or "sunny" in ans.lower(), "tool result -> final answer",
          answer=ans[:100], finish=r2.choices[0].finish_reason)

# --- sampling with a seed
story = [{"role": "user", "content": "Write two sentences about a lighthouse."}]
def sample(seed):
    return client.chat.completions.create(model=MODEL, messages=story, max_tokens=48,
                                          temperature=0.8, seed=seed).choices[0].message.content
a1, a2, b = sample(42), sample(42), sample(43)
check(a1 == a2, "sampling: same seed repeats")
check(a1 != b, "sampling: another seed differs")

# --- stop strings
r = client.chat.completions.create(model=MODEL, messages=sky, max_tokens=96, stop=["."])
sc = r.choices[0].message.content or ""
check("." not in sc and r.choices[0].finish_reason == "stop" and len(sc) > 0,
      "stop string", content=sc[:80])


def metric(name):
    _, text = get("/metrics")
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name):
            total += float(line.rsplit(" ", 1)[1])
    return total, text


# --- cancellation by disconnect
before, mtext = metric('gufo_npu_requests_total{status="499"}')
essay = [{"role": "user", "content": "Write a long essay about the history of bridges."}]
with httpx.Client(timeout=120) as h:
    with h.stream("POST", BASE + "/v1/chat/completions",
                  headers={"Authorization": f"Bearer {args.api_key}"},
                  json={"model": MODEL, "messages": essay, "max_tokens": 2000,
                        "stream": True}) as resp:
        n = 0
        for line in resp.iter_lines():
            if line.startswith("data: "):
                n += 1
            if n >= 6:
                break
t0 = time.perf_counter()
r = client.chat.completions.create(model=MODEL, messages=sky, max_tokens=8)
after_wait = time.perf_counter() - t0
after, _ = metric('gufo_npu_requests_total{status="499"}')
check(after == before + 1 and after_wait < 10, "disconnect cancels the generation",
      next_request_s=round(after_wait, 2), cancelled=after - before)

# --- queue full (server started with --max-queue 1)
codes = []
def long_req():
    try:
        client.with_options(max_retries=0).chat.completions.create(
            model=MODEL, messages=essay, max_tokens=120)
        codes.append(200)
    except openai.RateLimitError:
        codes.append(429)
    except Exception as e:  # noqa: BLE001
        codes.append(str(e)[:60])
ts = [threading.Thread(target=long_req) for _ in range(3)]
for t in ts:
    t.start()
    time.sleep(0.3)
for t in ts:
    t.join()
check(sorted(map(str, codes)) == ["200", "200", "429"], "queue full -> 429", codes=codes)

# --- multi-turn prompt cache
hist = filler(args.history_sentences, seed=7)
def turn2(nonce):
    t1 = [{"role": "user", "content": f"{nonce} Background: {hist} Summarize the notes in one line."}]
    r1 = client.chat.completions.create(model=MODEL, messages=t1, max_tokens=40)
    t2 = t1 + [{"role": "assistant", "content": r1.choices[0].message.content},
               {"role": "user", "content": "Now list three of the nouns."}]
    _, _, _, _, u, ttft, _ = stream(t2, max_tokens=24)
    return r1.usage.prompt_tokens, u, ttft
p1, u2, ttft_hit = turn2("A")
cached = u2.prompt_tokens_details.cached_tokens if u2.prompt_tokens_details else 0
check(cached > 0, "multi-turn: turn 2 cached_tokens", prompt=u2.prompt_tokens,
      cached=cached, ttft_hit=round(ttft_hit, 3))
# The same turn 2 with no checkpoint to restore: a different first message.
t2_cold = [{"role": "user", "content": f"B Background: {hist} Summarize the notes in one line."},
           {"role": "assistant", "content": "x"},
           {"role": "user", "content": "Now list three of the nouns."}]
_, _, _, _, u3, ttft_miss, _ = stream(t2_cold, max_tokens=24)
c3 = u3.prompt_tokens_details.cached_tokens if u3.prompt_tokens_details else 0
check(c3 == 0 and ttft_hit < ttft_miss, "multi-turn: cache hit is faster",
      ttft_hit=round(ttft_hit, 3), ttft_miss=round(ttft_miss, 3),
      prompt_miss=u3.prompt_tokens)

# --- metrics
_, mtext = metric("gufo_npu_requests_total")
check("gufo_npu_requests_total" in mtext and "gufo_npu_time_to_first_token" in mtext,
      "/metrics", lines=len(mtext.splitlines()))

fails = [r for r in results if not r["ok"]]
print("SUMMARY " + json.dumps({"model": MODEL, "checks": len(results),
                               "failed": [f["what"] for f in fails]}))
sys.exit(1 if fails else 0)
