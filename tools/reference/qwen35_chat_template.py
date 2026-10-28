#!/usr/bin/env python3
"""Render chat cases with the Qwen3.5 / Qwen3.6 GGUF chat templates.

The fixtures for `chat_format_test` (src/serve/chat_format_qwen35.cpp) come
from here: every case in the cases file is rendered with each template given,
the renders must agree across templates, and the result is written as the
expected file.

Rendering follows Hugging Face transformers' chat template environment, the
one the models' template was written for: a sandboxed Jinja environment with
trim_blocks and lstrip_blocks, loop controls, `raise_exception`, and a
`tojson` that keeps key order and non-ASCII text (json.dumps defaults
otherwise). Requests are prepared the way an OpenAI-compatible server must:
a tool call's `arguments` given as a JSON string is parsed to an object (the
template renders parameters only from an object), and a final assistant
message is continued rather than closed (assistant prefill: the render
without the generation prompt, minus its closing `<|im_end|>\\n`).

    python3 tools/reference/qwen35_chat_template.py \\
        --template tests/serve/fixtures/templates/qwen3.5-2b.jinja \\
        --template tests/serve/fixtures/templates/qwen3.5-4b.jinja \\
        --template tests/serve/fixtures/templates/qwen3.6-35b-a3b.jinja \\
        --cases tests/serve/fixtures/chat_cases.json \\
        --out tests/serve/fixtures/chat_expected.json

A template can also be read from a GGUF's `tokenizer.chat_template`:
`--extract MODEL.gguf OUT.jinja` writes it and exits.
"""

import argparse
import json
import struct
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

IM_END = "<|im_end|>\n"


def read_gguf_string(path, key):
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}

    def rd(f, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, f.read(size))[0]

    def rstr(f):
        return f.read(rd(f, "Q")).decode("utf-8")

    def skip(f, kind):
        if kind in scalar:
            f.seek(struct.calcsize("<" + scalar[kind]), 1)
        elif kind == 8:
            f.seek(rd(f, "Q"), 1)
        elif kind == 9:
            inner, count = rd(f, "I"), rd(f, "Q")
            for _ in range(count):
                skip(f, inner)
        else:
            raise ValueError(f"unknown GGUF value type {kind}")

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        rd(f, "I")
        rd(f, "Q")
        for _ in range(rd(f, "Q")):
            name, kind = rstr(f), rd(f, "I")
            if name == key and kind == 8:
                return rstr(f)
            skip(f, kind)
    raise KeyError(f"{path}: no {key}")


def environment():
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def prepare(messages):
    out = []
    for message in messages:
        message = dict(message)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            prepared = []
            for call in calls:
                call = json.loads(json.dumps(call))
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and isinstance(
                        function.get("arguments"), str):
                    try:
                        parsed = json.loads(function["arguments"])
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        function["arguments"] = parsed
                prepared.append(call)
            message["tool_calls"] = prepared
        out.append(message)
    return out


def render(template, case):
    messages = prepare(case["messages"])
    prefill = bool(messages) and messages[-1].get("role") == "assistant"
    text = template.render(messages=messages, tools=case.get("tools") or None,
                           add_generation_prompt=not prefill,
                           enable_thinking=bool(case.get("thinking", False)))
    if prefill:
        if not text.endswith(IM_END):
            raise ValueError(f"{case['name']}: prefill render does not end "
                             "with <|im_end|>")
        text = text[:-len(IM_END)]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", nargs=2, metavar=("GGUF", "OUT"))
    parser.add_argument("--template", action="append", default=[])
    parser.add_argument("--cases")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.extract:
        text = read_gguf_string(args.extract[0], "tokenizer.chat_template")
        with open(args.extract[1], "w", encoding="utf-8") as f:
            f.write(text)
        return 0
    if not args.template or not args.cases or not args.out:
        parser.error("--template, --cases and --out are required")

    env = environment()
    templates = []
    for path in args.template:
        with open(path, encoding="utf-8") as f:
            templates.append((path, env.from_string(f.read())))
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)

    expected = {}
    failures = 0
    for case in cases:
        name = case["name"]
        if case.get("reject"):
            # Refused by the server before templating (e.g. images).
            expected[name] = None
            continue
        renders = []
        for path, template in templates:
            try:
                renders.append(render(template, case))
            except jinja2.exceptions.TemplateError as e:
                renders.append(f"<error: {e}>")
        if any(r != renders[0] for r in renders):
            failures += 1
            print(f"FAIL: {name}: templates disagree", file=sys.stderr)
            for (path, _), r in zip(templates, renders):
                print(f"--- {path}\n{r!r}", file=sys.stderr)
        if renders[0].startswith("<error:"):
            expected[name] = None
            print(f"note: {name}: {renders[0]}", file=sys.stderr)
        else:
            expected[name] = renders[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases, {len(templates)} templates, "
          f"{failures} disagreements -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
