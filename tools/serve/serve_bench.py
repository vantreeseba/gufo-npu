"""Warm time to first token and decode tok/s of `gufo-npu serve` over HTTP.

Streams greedy chat completions for a short prompt (the sky question) and a
~1k-token one (filler notes), each a warm-up then --runs runs, and prints the
medians. Run the server with `--prompt-cache 0` so repeats are not served
from a checkpoint. With --gufo-npu and --model it also runs the same prompts
through `gufo-npu generate` (in process) for comparison:

  python3 tools/serve/serve_bench.py --base-url http://127.0.0.1:8080 \\
      [--gufo-npu build/release/gufo-npu --model MODEL.gguf] [--out result.json]

Needs `pip install openai`. Stop the server before the in-process runs if
both would not fit in memory (`--skip-http` runs only those).
"""
import argparse
import json
import re
import statistics
import subprocess
import time

import openai

from textgen import filler

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--api-key", default="none")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--gufo-npu", help="gufo-npu binary for in-process runs")
parser.add_argument("--model", help="GGUF for the in-process runs")
parser.add_argument("--skip-http", action="store_true")
parser.add_argument("--out", help="write the results as JSON")
args = parser.parse_args()

PROMPTS = {
    "short": "Explain in a few sentences why the sky is blue.",
    "1k": "Background: " + filler(70, seed=11) + " Summarize the notes in two sentences.",
}


def http_run(client, model, text):
    start = time.perf_counter()
    first = last = None
    usage = None
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}],
        max_tokens=args.max_tokens, stream=True,
        stream_options={"include_usage": True})
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            first = first or now
            last = now
    n = usage.completion_tokens
    return {"ttft": first - start,
            "decode_tps": (n - 1) / (last - first) if last > first else 0.0,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": n}


def in_process_run(text):
    out = subprocess.run(
        [args.gufo_npu, "generate", "--device", "npu", "--model", args.model,
         "--prompt", text, "--max-tokens", str(args.max_tokens)],
        capture_output=True, text=True, check=True).stderr
    m = re.search(r"\[(\d+) prompt tokens in ([\d.]+) s; (\d+) generated in "
                  r"([\d.]+) s, ([\d.]+) tok/s", out)
    return {"ttft": float(m.group(2)), "decode_tps": float(m.group(5)),
            "prompt_tokens": int(m.group(1)), "completion_tokens": int(m.group(3))}


def measure(run, text):
    run(text)  # warm-up
    runs = [run(text) for _ in range(args.runs)]
    return {
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens": runs[0]["completion_tokens"],
        "ttft": [round(r["ttft"], 3) for r in runs],
        "decode_tps": [round(r["decode_tps"], 2) for r in runs],
        "ttft_median": round(statistics.median(r["ttft"] for r in runs), 3),
        "decode_tps_median": round(statistics.median(r["decode_tps"] for r in runs), 2),
    }


results = {}
if not args.skip_http:
    client = openai.OpenAI(base_url=args.base_url.rstrip("/") + "/v1",
                           api_key=args.api_key, timeout=600)
    model = client.models.list().data[0].id
    results["model"] = model
    for name, text in PROMPTS.items():
        results[f"http_{name}"] = measure(lambda t: http_run(client, model, t), text)
        print(f"http {name}", json.dumps(results[f"http_{name}"]), flush=True)
if args.gufo_npu and args.model:
    for name, text in PROMPTS.items():
        results[f"in_process_{name}"] = measure(in_process_run, text)
        print(f"in process {name}", json.dumps(results[f"in_process_{name}"]),
              flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
