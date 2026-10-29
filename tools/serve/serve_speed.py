"""Warm speed of `gufo-npu serve` per request mode, from the server's metrics.

For a short and a ~1k-token prompt, at temperature 0 and 0.7, runs each mode
(plain, json_schema, tool_choice required, logprobs, n=3) a warm-up then
--runs times and prints medians of:
  ttft     engine time to the first token (prefill + first step), per
           request (n=3: the first choice's)
  tok/s    generated tokens / engine decode seconds
  accept   MTP drafts accepted / decode steps
taken from /metrics deltas, so tool calls (sent whole) are timed like text.
Run the server with `--prompt-cache 0` so repeats are not served from a
checkpoint. Also times the first constrained request (vocabulary index build)
when it is the first one the server sees (--first-constrained).

  python3 tools/serve/serve_speed.py --base-url http://127.0.0.1:8080 \\
      [--runs 3] [--out speed.json]
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

import openai

from textgen import filler

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--api-key", default="none")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--modes", default="plain,json_schema,tool_required,"
                    "logprobs,n3")
parser.add_argument("--first-constrained", action="store_true")
parser.add_argument("--out")
args = parser.parse_args()

client = openai.OpenAI(base_url=args.base_url + "/v1", api_key=args.api_key,
                       timeout=900)
MODEL = client.models.list().data[0].id

PROMPTS = {
    "short": "Invent a short profile of a person who works in a bakery.",
    "1k": "Background: " + filler(70, seed=11) +
          " Invent a short profile of one person mentioned in the notes.",
}
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "job": {"type": "string"},
    "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["name", "age", "job", "hobbies"],
    "additionalProperties": False}
TOOLS = [{"type": "function", "function": {
    "name": "save_profile",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"},
        "job": {"type": "string"}, "bio": {"type": "string"}},
        "required": ["name", "age", "job", "bio"]}}}]
MODES = {
    "plain": {},
    "json_schema": {"response_format": {"type": "json_schema", "json_schema": {
        "name": "person", "schema": PERSON, "strict": True}}},
    "tool_required": {"tools": TOOLS, "tool_choice": "required"},
    "logprobs": {"logprobs": True, "top_logprobs": 5},
    "n3": {"n": 3},
}
COUNTERS = ["gufo_npu_prefill_seconds_total", "gufo_npu_decode_seconds_total",
            "gufo_npu_decode_steps_total", "gufo_npu_drafts_accepted_total",
            "gufo_npu_completion_tokens_total", "gufo_npu_requests_total"]


def metrics():
    text = urllib.request.urlopen(args.base_url + "/metrics").read().decode()
    out = {}
    for name in COUNTERS:
        out[name] = sum(float(v) for v in re.findall(
            rf"^{name}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M))
    return out


def run(mode, prompt, temperature, seed):
    kw = dict(MODES[mode])
    kw.update(max_tokens=args.max_tokens, temperature=temperature)
    if temperature > 0:
        kw["seed"] = seed
    before = metrics()
    start = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": PROMPTS[prompt]}],
        **kw)
    wall = time.perf_counter() - start
    after = metrics()
    d = {k: after[k] - before[k] for k in COUNTERS}
    choices = len(r.choices)
    return {"ttft": d["gufo_npu_prefill_seconds_total"] / choices,
            "tps": d["gufo_npu_completion_tokens_total"] /
            max(d["gufo_npu_decode_seconds_total"], 1e-9),
            "accept": d["gufo_npu_drafts_accepted_total"] /
            max(d["gufo_npu_decode_steps_total"], 1),
            "tokens": r.usage.completion_tokens, "wall": wall}


results = {"model": MODEL, "rows": []}
if args.first_constrained:
    first = run("json_schema", "short", 0.0, 0)
    second = run("json_schema", "short", 0.0, 0)
    results["first_constrained"] = {"first_wall": first["wall"],
                                    "second_wall": second["wall"]}
    print(f"first constrained request {first['wall']:.2f} s wall, the next "
          f"{second['wall']:.2f} s", flush=True)

print(f"{'prompt':6} {'T':>3} {'mode':14} {'ttft s':>7} {'tok/s':>6} "
      f"{'accept':>6} {'tokens':>6}", flush=True)
for prompt in PROMPTS:
    for temperature in (0.0, 0.7):
        for mode in args.modes.split(","):
            run(mode, prompt, temperature, 1)  # warm-up
            rs = [run(mode, prompt, temperature, 2 + i)
                  for i in range(args.runs)]
            row = {"prompt": prompt, "temperature": temperature, "mode": mode}
            for k in ("ttft", "tps", "accept", "tokens"):
                row[k] = statistics.median(r[k] for r in rs)
            results["rows"].append(row)
            print(f"{prompt:6} {temperature:3.1f} {mode:14} {row['ttft']:7.3f} "
                  f"{row['tps']:6.1f} {row['accept']:6.2f} "
                  f"{row['tokens']:6.0f}", flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
