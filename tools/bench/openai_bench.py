#!/usr/bin/env python3
"""Time prefill and decode on an OpenAI-compatible chat server.

Streams one request per prompt length and reports, per request:
time to first token (TTFT), prefill tok/s (prompt tokens / TTFT) and decode
tok/s (tokens after the first / time after the first). Uses only the
standard library so it runs anywhere.

  openai_bench.py --base-url http://HOST:PORT/api/v1 --model MODEL \
      --prompt-tokens 128,1024,4096 --max-tokens 128 --repeats 2
"""

import argparse
import json
import time
import urllib.request

# ~1 token per word for Qwen tokenizers on plain English filler.
FILLER = (
    "The quick brown fox jumps over the lazy dog while the river runs "
    "past old stone bridges and quiet fields of wheat under a pale sky. "
)


def build_prompt(target_tokens: int, salt: int) -> str:
    words = FILLER.split()
    body = " ".join(words[i % len(words)] for i in range(max(target_tokens - 40, 1)))
    # A unique prefix defeats prompt caching between requests.
    return (
        f"Request {salt}. Read the text, then write a long story about it.\n\n"
        f"{body}"
    )


def run_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    if first is None:
        raise RuntimeError("no tokens streamed")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or chunks
    ttft = first - start
    decode_time = last - first
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens else None,
        "decode_tok_s": (
            round((completion_tokens - 1) / decode_time, 2) if decode_time > 0 else None
        ),
        "total_s": round(time.perf_counter() - start, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", default="128,1024,4096")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print JSON lines")
    args = parser.parse_args()

    salt = int(time.time())
    for _ in range(args.warmup):
        run_once(args.base_url, args.model, build_prompt(64, salt), 16)
        salt += 1

    if not args.json:
        print(f"{'target':>7} {'prompt':>7} {'gen':>5} {'ttft s':>8} "
              f"{'prefill t/s':>12} {'decode t/s':>11}")
    for target in (int(t) for t in args.prompt_tokens.split(",")):
        for _ in range(args.repeats):
            result = run_once(args.base_url, args.model, build_prompt(target, salt),
                              args.max_tokens)
            salt += 1
            result["target_tokens"] = target
            if args.json:
                print(json.dumps(result), flush=True)
            else:
                print(f"{target:>7} {result['prompt_tokens'] or '?':>7} "
                      f"{result['completion_tokens']:>5} {result['ttft_s']:>8} "
                      f"{result['prefill_tok_s'] or '?':>12} "
                      f"{result['decode_tok_s'] or '?':>11}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
