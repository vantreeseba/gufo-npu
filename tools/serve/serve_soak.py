"""Soak `gufo-npu serve` with mixed multi-turn chat traffic.

Each conversation has 2-4 turns over a random filler history (up to
--max-history sentences, ~13 tokens each); 70% of requests stream, 30%
sample (temperature 0.7, a random seed) and 10% hang up after a few chunks.
Records each request's time to first token, decode tok/s and cached prompt
tokens, and, with --server-pid on the server's machine, its memory every
30 s; then prints a summary per --window seconds:

  python3 tools/serve/serve_soak.py --base-url http://127.0.0.1:8080 \\
      --minutes 30 [--server-pid PID] [--out soak.json]

Needs `pip install openai httpx`.
"""
import argparse
import json
import random
import statistics
import threading
import time

import httpx
import openai

from textgen import filler

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--api-key", default="none")
parser.add_argument("--minutes", type=float, default=5)
parser.add_argument("--max-history", type=int, default=280)
parser.add_argument("--server-pid", type=int, help="sample /proc/PID/status")
parser.add_argument("--window", type=float, default=300)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--out", help="write every record as JSON")
args = parser.parse_args()

BASE = args.base_url.rstrip("/")
HEADERS = {"Authorization": f"Bearer {args.api_key}"}
client = openai.OpenAI(base_url=BASE + "/v1", api_key=args.api_key, timeout=900,
                       max_retries=0)
MODEL = client.models.list().data[0].id
rng = random.Random(args.seed)
records, memory, errors = [], [], []
stop = threading.Event()
t_start = time.time()


def elapsed():
    return round(time.time() - t_start, 1)


def sample_memory():
    while not stop.is_set():
        try:
            sample = {"t": elapsed()}
            for line in open(f"/proc/{args.server_pid}/status"):
                if line.startswith(("VmRSS", "RssAnon")):
                    key, value = line.split(":")
                    sample[key] = int(value.split()[0])  # KiB
            memory.append(sample)
        except OSError as e:
            errors.append({"t": elapsed(), "memory": str(e)})
        stop.wait(30)


def cached(usage):
    details = usage.prompt_tokens_details
    return details.cached_tokens if details else 0


def request(messages, streamed, sampled, hang_up):
    kw = {"max_tokens": rng.choice([32, 64, 128, 256])}
    if sampled:
        kw.update(temperature=0.7, seed=rng.randrange(1 << 30))
    start = time.perf_counter()
    rec = {"t": elapsed(), "stream": streamed, "sampled": sampled,
           "hang_up": hang_up}
    if not streamed:
        r = client.chat.completions.create(model=MODEL, messages=messages, **kw)
        rec.update(total=time.perf_counter() - start, prompt=r.usage.prompt_tokens,
                   completion=r.usage.completion_tokens, cached=cached(r.usage))
        return rec, r.choices[0].message.content or ""
    if hang_up:
        with httpx.Client(timeout=900) as h:
            with h.stream("POST", BASE + "/v1/chat/completions", headers=HEADERS,
                          json={"model": MODEL, "messages": messages,
                                "stream": True, "max_tokens": 512}) as resp:
                events = 0
                for line in resp.iter_lines():
                    events += line.startswith("data: ")
                    if events >= rng.randint(2, 8):
                        break
        rec.update(total=time.perf_counter() - start, cancelled=True)
        return rec, None
    first = last = None
    usage, text = None, ""
    for chunk in client.chat.completions.create(
            model=MODEL, messages=messages, stream=True,
            stream_options={"include_usage": True}, **kw):
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            first = first or now
            last = now
            text += chunk.choices[0].delta.content
    rec.update(total=time.perf_counter() - start, prompt=usage.prompt_tokens,
               completion=usage.completion_tokens, cached=cached(usage))
    if first is not None:
        rec["ttft"] = first - start
        if last > first and usage.completion_tokens > 1:
            rec["decode_tps"] = (usage.completion_tokens - 1) / (last - first)
    return rec, text


def median(xs):
    return round(statistics.median(xs), 3) if xs else None


def summarize():
    print(f"{MODEL}: {len(records)} requests, {len(errors)} errors, "
          f"{sum(1 for r in records if r.get('cancelled'))} hung up, "
          f"{sum(1 for r in records if r.get('sampled'))} sampled, "
          f"{sum(1 for r in records if r.get('cached', 0) > 0)} cache hits")
    for e in errors[:5]:
        print("  error:", e)
    w = 0.0
    end = max((r["t"] for r in records), default=0)
    while w <= end:
        rs = [r for r in records if w <= r["t"] < w + args.window]
        hit = [r["ttft"] for r in rs if "ttft" in r and r.get("cached", 0) > 0]
        miss = [r["ttft"] for r in rs if "ttft" in r and r.get("cached", 0) == 0]
        tps = [r["decode_tps"] for r in rs
               if "decode_tps" in r and r.get("completion", 0) >= 32]
        print(f"  {w / 60:5.1f}-{(w + args.window) / 60:5.1f} min: {len(rs):4d} "
              f"requests, prompt median {median([r['prompt'] for r in rs if 'prompt' in r])}, "
              f"TTFT cache hit {median(hit)} s ({len(hit)}), miss {median(miss)} s "
              f"({len(miss)}), decode {median(tps)} tok/s")
        w += args.window
    anon = [m for m in memory if "RssAnon" in m]
    if anon:
        mb = [m["RssAnon"] // 1024 for m in anon]
        step = max(1, len(anon) // 12)
        print(f"  anonymous memory MB: start {mb[0]}, peak {max(mb)}, end {mb[-1]}; "
              + ", ".join(f"{m['t'] / 60:.0f}m {m['RssAnon'] // 1024}"
                          for m in anon[::step]))


if args.server_pid:
    threading.Thread(target=sample_memory, daemon=True).start()
deadline = time.time() + args.minutes * 60
conversation = 0
while time.time() < deadline:
    conversation += 1
    history = filler(rng.randint(1, args.max_history), seed=rng.randrange(1 << 30))
    messages = [{"role": "user",
                 "content": f"Conversation {conversation}. {history} Summarize briefly."}]
    for turn in range(rng.randint(2, 4)):
        if time.time() >= deadline:
            break
        streamed = rng.random() < 0.7
        try:
            rec, reply = request(messages, streamed, rng.random() < 0.3,
                                 streamed and rng.random() < 0.14)
        except Exception as e:  # noqa: BLE001 (record and go on)
            errors.append({"t": elapsed(), "conversation": conversation,
                           "turn": turn, "error": repr(e)[:200]})
            break
        rec.update(conversation=conversation, turn=turn)
        records.append(rec)
        messages = messages + [
            {"role": "assistant", "content": reply if reply is not None else "(no reply)"},
            {"role": "user", "content": rng.choice(
                ["Tell me more.", "Name two nouns from the notes.",
                 "Shorter, please.", "Which note mentions a river?"])}]
stop.set()
if args.server_pid:
    time.sleep(0.1)
    sample = {"t": elapsed()}
    try:
        for line in open(f"/proc/{args.server_pid}/status"):
            if line.startswith(("VmRSS", "RssAnon")):
                key, value = line.split(":")
                sample[key] = int(value.split()[0])
        memory.append(sample)
    except OSError:
        pass
if args.out:
    with open(args.out, "w") as f:
        json.dump({"model": MODEL, "minutes": args.minutes, "records": records,
                   "memory": memory, "errors": errors}, f)
summarize()
