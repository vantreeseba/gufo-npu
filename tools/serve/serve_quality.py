"""Quality checks of a running `gufo-npu serve` through the openai client.

Sections (pick with --checks, comma separated):
  hashes      greedy reference prompts: md5 of the reply (first 8 hex, as
              `gufo-npu generate | md5sum`), without and with logprobs
  structured  json_object, strict json_schema (nested, arrays, enums,
              optional fields, a recursive $ref) and forced tool calls:
              --cases requests per case (thinking off), --thinking-cases for
              two of them with thinking on; replies parsed and validated
              against their schema; samples kept for reading
  logprobs    temperature 0: the chosen token is top_logprobs[0]; per-token
              logprobs saved (compare two servers with --compare A.json B.json)
  functional  logit_bias +-100, parallel_tool_calls false, /v1/completions
  tool        a forced tool call whose string argument holds quotes and
              newlines (--cases times)
  switch      one chat per model id in --switch (several-model servers)

  python3 tools/serve/serve_quality.py --base-url http://127.0.0.1:8080 \\
      --checks hashes,structured --cases 50 --out quality.json

Needs `pip install openai jsonschema`; `functional` also needs `tokenizers`
and --tokenizer (the model's tokenizer.json) to find token ids.
"""
import argparse
import hashlib
import json
import random
import sys
import time

import jsonschema
import openai

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--api-key", default="none")
parser.add_argument("--checks", default="hashes,structured,logprobs,functional")
parser.add_argument("--cases", type=int, default=50)
parser.add_argument("--thinking-cases", type=int, default=0)
parser.add_argument("--only", default="", help="structured cases to run, "
                    "comma separated (default: all)")
parser.add_argument("--tokenizer", help="tokenizer.json for token ids")
parser.add_argument("--switch", default="", help="model ids, comma separated")
parser.add_argument("--compare", nargs=2, metavar="JSON",
                    help="compare the logprobs of two earlier runs and exit")
parser.add_argument("--out", help="write the results as JSON")
args = parser.parse_args()

if args.compare:
    a, b = (json.load(open(p))["logprobs"]["tokens"] for p in args.compare)
    for name in a:
        ta, tb = a[name], b[name]
        same = 0
        worst = 0.0
        for x, y in zip(ta, tb):
            if x[0] != y[0]:
                break
            same += 1
            worst = max(worst, abs(x[1] - y[1]))
        print(f"{name}: {same}/{min(len(ta), len(tb))} tokens equal before "
              f"any difference; max |logprob difference| {worst:.2e}")
    sys.exit(0)

client = openai.OpenAI(base_url=args.base_url + "/v1", api_key=args.api_key,
                       timeout=900)
MODEL = client.models.list().data[0].id
results = {"model": MODEL}
failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]


SKY = "Explain in a few sentences why the sky is blue."
FIB = ("Write a Python function that returns the n-th Fibonacci number, "
       "with a docstring and a short example.")
P1K = ("Here is some background text. " + " ".join(
    f"Item {i}: the quick brown fox jumps over the lazy dog number {i}."
    for i in range(50)) + " Now explain in a few sentences why the sky is blue.")
REFERENCE = [("sky", SKY, 200), ("fib", FIB, 256), ("1k", P1K, 128)]


def chat(text, **kw):
    return client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": text}], **kw)


# ---- hashes ----------------------------------------------------------------

def hashes():
    out = {}
    for name, text, n in REFERENCE:
        plain = chat(text, max_tokens=n).choices[0].message.content or ""
        lp = chat(text, max_tokens=n, logprobs=True, top_logprobs=3)
        with_lp = lp.choices[0].message.content or ""
        out[name] = {"plain": md5(plain), "logprobs": md5(with_lp)}
        check(plain == with_lp, f"hash {name}: {md5(plain)}, the same with "
              f"logprobs ({md5(with_lp)})")
    results["hashes"] = out


# ---- structured outputs ----------------------------------------------------

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"type": "string", "enum": ["engineer", "designer", "manager",
                                            "scientist"]},
        "skills": {"type": "array", "items": {"type": "string"},
                   "minItems": 1, "maxItems": 5},
        "address": {"type": "object",
                    "properties": {"city": {"type": "string"},
                                   "country": {"type": "string"}},
                    "required": ["city", "country"],
                    "additionalProperties": False},
        "nickname": {"type": "string"},
    },
    "required": ["name", "age", "role", "skills", "address"],
    "additionalProperties": False,
}
RECIPE = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "servings": {"type": "integer"},
        "ingredients": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string",
                         "enum": ["g", "ml", "pcs", "tbsp", "tsp", "cup"]}},
            "required": ["name", "quantity", "unit"],
            "additionalProperties": False}},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["title", "servings", "ingredients", "steps"],
    "additionalProperties": False,
}
TREE = {
    "$defs": {"node": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "title": {"type": "string"},
                       "reports": {"type": "array",
                                   "items": {"$ref": "#/$defs/node"},
                                   "maxItems": 4}},
        "required": ["name", "title", "reports"],
        "additionalProperties": False}},
    "$ref": "#/$defs/node",
}
SENTIMENT = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string",
                      "enum": ["positive", "negative", "neutral", "mixed"]},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"},
                    "maxItems": 3},
    },
    "required": ["sentiment", "confidence", "reasons"],
    "additionalProperties": False,
}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
]

PEOPLE = ["a senior backend engineer in Berlin", "a young designer from Lagos",
          "a retired manager living in Osaka", "a marine scientist in Hobart",
          "a game developer from Montreal", "a nurse turned data scientist in Lima"]
DISHES = ["pancakes", "a lentil soup", "guacamole", "a simple tomato pasta",
          "banana bread", "a chickpea curry"]
ORGS = ["a five-person startup", "a small bakery", "a university lab",
        "a hospital ward", "a newspaper desk", "a three-person band"]
REVIEWS = ["The battery lasts forever but the screen scratches easily.",
           "Absolutely loved it, would buy again!",
           "It arrived broken and support never answered.",
           "It's fine. Does what it says.",
           "Great food, terrible service, long wait.",
           "The update made everything slower and crash more."]
CITIES = ["Paris", "Tokyo", "Nairobi", "Reykjavik", "Buenos Aires", "Toronto"]

CASES = {
    "json_object": (lambda i: f"Describe {random.Random(i).choice(DISHES)} "
                    "as a JSON object.",
                    {"type": "json_object"}, None, None),
    "person": (lambda i: "Invent a profile for "
               f"{PEOPLE[i % len(PEOPLE)]}.", PERSON, None, None),
    "recipe": (lambda i: f"Give me a recipe for {DISHES[i % len(DISHES)]}.",
               RECIPE, None, None),
    "tree": (lambda i: f"Sketch the org chart of {ORGS[i % len(ORGS)]}.",
             TREE, None, None),
    "sentiment": (lambda i: "Classify the sentiment of this review: "
                  f"{REVIEWS[i % len(REVIEWS)]}", SENTIMENT, None, None),
    "tool_required": (lambda i: f"What's the weather like in "
                      f"{CITIES[i % len(CITIES)]} right now?", None, TOOLS,
                      "required"),
    "tool_named": (lambda i: f"I'm flying to {CITIES[i % len(CITIES)]} "
                   "tomorrow, anything I should check?", None, TOOLS,
                   {"type": "function", "function": {"name": "get_weather"}}),
}


def one_case(name, i, thinking):
    prompt, schema, tools, choice = CASES[name]
    kw = {"max_tokens": 3000 if thinking else 800,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
    # Every fifth request greedy, the rest sampled with their own seed.
    if i % 5 != 0:
        kw.update(temperature=0.7, seed=1000 + i)
    if tools:
        kw.update(tools=tools, tool_choice=choice)
    elif schema.get("type") == "json_object":
        kw["response_format"] = schema
    else:
        kw["response_format"] = {"type": "json_schema", "json_schema": {
            "name": name, "schema": schema, "strict": True}}
    start = time.perf_counter()
    try:
        r = chat(prompt(i), **kw)
    except openai.APIError as e:  # a server error counts as a failure
        return {"i": i, "prompt": prompt(i), "finish": "error", "tokens": 0,
                "seconds": round(time.perf_counter() - start, 2),
                "valid": False, "error": str(e)[:300]}
    c = r.choices[0]
    rec = {"i": i, "prompt": prompt(i), "finish": c.finish_reason,
           "tokens": r.usage.completion_tokens,
           "seconds": round(time.perf_counter() - start, 2),
           "reasoning_chars": len(getattr(c.message, "reasoning_content", None)
                                  or "")}
    try:
        if tools:
            calls = c.message.tool_calls or []
            rec["output"] = [{"name": t.function.name,
                              "arguments": t.function.arguments} for t in calls]
            ok = bool(calls)
            for t in calls:
                fn = next(f["function"] for f in tools
                          if f["function"]["name"] == t.function.name)
                jsonschema.validate(json.loads(t.function.arguments),
                                    fn["parameters"])
                if isinstance(choice, dict):
                    ok = ok and t.function.name == choice["function"]["name"]
            rec["valid"] = ok
        else:
            rec["output"] = c.message.content
            value = json.loads(c.message.content)
            if schema.get("type") == "json_object":
                rec["valid"] = isinstance(value, dict)
            else:
                jsonschema.validate(value, schema)
                rec["valid"] = True
    except Exception as e:  # parse or validation failure
        rec["valid"] = False
        rec["error"] = str(e)[:300]
    return rec


def structured():
    out = {}
    names = [n for n in CASES if not args.only or n in args.only.split(",")]
    plan = [(n, False, args.cases) for n in names]
    if args.thinking_cases:
        plan += [(n, True, args.thinking_cases)
                 for n in ("person", "tool_required") if n in names]
    for name, thinking, count in plan:
        recs = [one_case(name, i, thinking) for i in range(count)]
        valid = sum(r["valid"] for r in recs)
        cut = sum(r["finish"] == "length" for r in recs)
        key = name + (" +thinking" if thinking else "")
        out[key] = {
            "valid": valid, "requests": count, "cut_by_max_tokens": cut,
            "mean_tokens": round(sum(r["tokens"] for r in recs) / count, 1),
            "mean_seconds": round(sum(r["seconds"] for r in recs) / count, 2),
            "samples": recs[:4] + [r for r in recs if not r["valid"]][:6]}
        check(valid == count, f"{key}: {valid}/{count} valid ({cut} cut by "
              "max_tokens)")
    results["structured"] = out


# ---- logprobs --------------------------------------------------------------

def logprobs():
    tokens = {}
    consistent = True
    for name, text, n in REFERENCE:
        r = chat(text, max_tokens=n, logprobs=True, top_logprobs=5)
        content = r.choices[0].logprobs.content
        tokens[name] = [(e.token, e.logprob) for e in content]
        for e in content:
            top = e.top_logprobs[0]
            if top.token != e.token or abs(top.logprob - e.logprob) > 1e-6:
                consistent = False
            if e.logprob > 1e-6:
                consistent = False
        joined = "".join(e.token for e in content)
        check(joined == (r.choices[0].message.content or ""),
              f"logprobs {name}: tokens join to the reply ({len(content)})")
    check(consistent, "logprobs: at temperature 0 every chosen token is "
          "top_logprobs[0] with the same logprob")
    results["logprobs"] = {"consistent": consistent, "tokens": tokens}


# ---- functional ------------------------------------------------------------

def functional():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    out = {}
    base = chat(SKY, max_tokens=8, logprobs=True, top_logprobs=1)
    first = base.choices[0].logprobs.content[0].token
    first_id = tok.token_to_id(first) or tok.encode(
        first, add_special_tokens=False).ids[0]
    banned = chat(SKY, max_tokens=8, logit_bias={str(first_id): -100},
                  logprobs=True, top_logprobs=1)
    b_first = banned.choices[0].logprobs.content[0].token
    check(b_first != first, f"logit_bias -100 on {first!r}: first token now "
          f"{b_first!r}")
    forced_id = tok.encode("Banana", add_special_tokens=False).ids[0]
    forced = chat(SKY, max_tokens=8, logit_bias={str(forced_id): 100})
    text = forced.choices[0].message.content or ""
    check(text.startswith(tok.decode([forced_id])),
          f"logit_bias +100 forces {tok.decode([forced_id])!r}: {text[:40]!r}")
    out["logit_bias"] = {"first": first, "banned_first": b_first,
                         "forced": text[:60]}

    two = [TOOLS[0], {"type": "function", "function": {
        "name": "get_time", "description": "Local time in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    q = ("What's the weather and the local time in Paris and in London? "
         "Use the tools.")
    many = chat(q, tools=two, max_tokens=600).choices[0]
    single = chat(q, tools=two, max_tokens=600,
                  parallel_tool_calls=False).choices[0]
    n_many = len(many.message.tool_calls or [])
    n_single = len(single.message.tool_calls or [])
    check(n_single == 1 and single.finish_reason == "tool_calls",
          f"parallel_tool_calls false: {n_single} call(s) (default: {n_many})")
    out["parallel_tool_calls"] = {"default_calls": n_many,
                                  "false_calls": n_single}

    c = client.completions.create(model=MODEL, prompt="The capital of France is",
                                  max_tokens=8, temperature=0)
    t1 = c.choices[0].text
    check("Paris" in t1, f"completions string prompt: {t1!r}")
    ids = tok.encode("The capital of France is", add_special_tokens=False).ids
    c2 = client.completions.create(model=MODEL, prompt=ids, max_tokens=8,
                                   temperature=0)
    check(c2.choices[0].text == t1, "completions token-id prompt: same text")
    c3 = client.completions.create(model=MODEL, prompt="The capital of France is",
                                   max_tokens=8, temperature=0, echo=True)
    check(c3.choices[0].text == "The capital of France is" + t1,
          "completions echo")
    streamed = "".join(
        ch.choices[0].text for ch in client.completions.create(
            model=MODEL, prompt="The capital of France is", max_tokens=8,
            temperature=0, stream=True) if ch.choices)
    check(streamed == t1, "completions stream: same text")
    c4 = client.completions.create(model=MODEL, prompt="The capital of France is",
                                   max_tokens=8, temperature=0, logprobs=3)
    lp = c4.choices[0].logprobs
    check(lp is not None and len(lp.tokens) == len(lp.token_logprobs) and
          "".join(lp.tokens) == c4.choices[0].text and
          all(len(t) <= 3 for t in lp.top_logprobs),
          f"completions legacy logprobs ({len(lp.tokens) if lp else 0} tokens)")
    c5 = client.completions.create(model=MODEL, prompt="Once upon a time",
                                   max_tokens=12, temperature=0.8, seed=5, n=2)
    check(len(c5.choices) == 2, "completions n=2")
    out["completions"] = {"text": t1}
    results["functional"] = out


# ---- a forced tool call with a quoted, multi-line string -------------------

def tool():
    note = [{"type": "function", "function": {
        "name": "save_note",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title", "body"]}}}]
    prompt = ('Save a note titled "Shopping" whose body lists, one per line: '
              'eggs, "organic" milk, bread, and the brand "Dad\'s Best" coffee.')
    recs = []
    for i in range(args.cases):
        kw = {} if i == 0 else {"temperature": 0.7, "seed": 50 + i}
        r = chat(prompt, tools=note, max_tokens=400, tool_choice={
            "type": "function", "function": {"name": "save_note"}}, **kw)
        calls = r.choices[0].message.tool_calls or []
        rec = {"calls": [t.function.arguments for t in calls]}
        try:
            a = json.loads(calls[0].function.arguments)
            jsonschema.validate(a, note[0]["function"]["parameters"])
            rec["valid"] = True
            rec["quotes_and_newlines"] = '"' in a["body"] and "\n" in a["body"]
        except Exception as e:
            rec["valid"] = False
            rec["error"] = str(e)[:200]
        recs.append(rec)
    valid = sum(r["valid"] for r in recs)
    both = sum(r.get("quotes_and_newlines", False) for r in recs)
    check(valid == len(recs), f"forced save_note: {valid}/{len(recs)} valid, "
          f"{both} with quotes and newlines in body")
    results["tool"] = {"valid": valid, "requests": len(recs),
                       "quotes_and_newlines": both, "samples": recs[:5]}


# ---- several models ----------------------------------------------------------

def switch():
    out = []
    for mid in [m for m in args.switch.split(",") if m]:
        start = time.perf_counter()
        r = client.chat.completions.create(
            model=mid, max_tokens=16, messages=[{
                "role": "user",
                "content": "What is 17 + 25? Answer with just the number."}])
        secs = time.perf_counter() - start
        text = r.choices[0].message.content or ""
        check(r.model == mid and "42" in text,
              f"switch to {mid}: {text.strip()!r} in {secs:.1f} s")
        out.append({"model": mid, "reply": text, "seconds": round(secs, 1)})
    results["switch"] = out


for name in args.checks.split(","):
    globals()[name]()
results["failures"] = failures
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
print(f"{len(failures)} failure(s)")
