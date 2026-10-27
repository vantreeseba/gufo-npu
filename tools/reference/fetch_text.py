#!/usr/bin/env python3
"""Fetches pinned public dataset rows as plain text for quantization checks.

Writes <out>/eval/<name>.txt (held-out evaluation text) and
<out>/calib/<name>.txt (calibration text for activation statistics). The two
sets come from different rows of each dataset, and neither overlaps the
oracle prompts (tools/reference/qwen36_llama_oracle.py), which are written by
hand. Rows come from the Hugging Face datasets-server rows API; every entry
pins dataset, config, split and row offset, so reruns fetch the same text.

  python3 tools/reference/fetch_text.py --out <dir>
"""

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

ROWS_URL = "https://datasets-server.huggingface.co/rows"


def assistant_turns(row):
    return "\n".join(m["content"] for m in row["messages"]
                     if m["role"] == "assistant")


def field(name):
    return lambda row: row[name]


WIKI = "wikimedia/wikipedia"
DATASETS = {
    "wiki-en": (WIKI, "20231101.en", "train", field("text")),
    "wiki-de": (WIKI, "20231101.de", "train", field("text")),
    "wiki-fr": (WIKI, "20231101.fr", "train", field("text")),
    "wiki-zh": (WIKI, "20231101.zh", "train", field("text")),
    "wiki-ja": (WIKI, "20231101.ja", "train", field("text")),
    "chat": ("HuggingFaceH4/ultrachat_200k", "default", "train_sft",
             assistant_turns),
    "code": ("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train",
             field("answer")),
    "math": ("project-numina/NuminaMath-CoT", "default", "train",
             field("solution")),
    "alpaca-zh": ("shibing624/alpaca-zh", "default", "train",
                  field("output")),
}

# file name: (dataset, first row, rows joined with blank lines)
EVAL = {
    "wiki-en": ("wiki-en", 3_000_001, 1),
    "wiki-de": ("wiki-de", 1_500_007, 4),
    "wiki-zh": ("wiki-zh", 700_003, 8),
    "chat": ("chat", 150_011, 1),
    "code": ("code", 150_013, 1),
    "math": ("math", 800_017, 1),
}
CALIB = {
    "wiki-en-1": ("wiki-en", 1_000_003, 1),
    "wiki-en-2": ("wiki-en", 2_000_029, 1),
    "wiki-en-3": ("wiki-en", 4_000_037, 1),
    "wiki-de": ("wiki-de", 500_009, 1),
    "wiki-fr": ("wiki-fr", 1_000_019, 1),
    "wiki-zh": ("wiki-zh", 300_007, 1),
    "wiki-ja": ("wiki-ja", 600_011, 1),
    "chat-1": ("chat", 10_007, 1),
    "chat-2": ("chat", 60_013, 1),
    "chat-3": ("chat", 110_017, 1),
    "code-1": ("code", 20_011, 1),
    "code-2": ("code", 80_021, 1),
    "code-3": ("code", 120_031, 1),
    "math-1": ("math", 100_003, 1),
    "math-2": ("math", 400_009, 1),
    "alpaca-zh": ("alpaca-zh", 5_003, 6),
}


def get_json(url, attempts=8):
    for attempt in range(attempts):
        time.sleep(1)  # stay under the rows API's rate limit
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if attempt + 1 == attempts or error.code not in (429, 500, 502,
                                                             503, 504):
                raise
            wait = error.headers.get("Retry-After")
            time.sleep(int(wait) if wait and wait.isdigit() else
                       min(120, 10 * 2**attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(120, 10 * 2**attempt))
    return None


def fetch(dataset, offset, rows):
    name, config, split, text_of = DATASETS[dataset]
    query = {"dataset": name, "config": config, "split": split,
             "offset": offset, "length": rows}
    data = get_json(ROWS_URL + "?" + urllib.parse.urlencode(query))
    texts = [text_of(r["row"]) for r in data["rows"]]
    if len(texts) != rows:
        raise RuntimeError(f"{dataset}: got {len(texts)} of {rows} rows "
                           f"at offset {offset}")
    return "\n\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for subdir, entries in (("eval", EVAL), ("calib", CALIB)):
        out = args.out / subdir
        out.mkdir(parents=True, exist_ok=True)
        for file, (dataset, offset, rows) in entries.items():
            path = out / f"{file}.txt"
            if path.exists():
                continue
            path.write_text(fetch(dataset, offset, rows), encoding="utf-8")
            print(f"{subdir}/{file}.txt: {dataset} rows {offset}+{rows}")


if __name__ == "__main__":
    main()
