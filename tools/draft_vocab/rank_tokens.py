#!/usr/bin/env python3
"""Ranks Qwen3.5 token ids by how often they occur in a multilingual sample.

The MTP draft's LM head only scores the top of this ranking (a draft
vocabulary), which is ~8x cheaper than all 248k rows; the main model still
checks every draft over the full vocabulary, so output does not change.

The sample is fetched from the Hugging Face datasets-server (rows API, no
large downloads) and cached under --cache: chat answers (English, Chinese,
Japanese), code answers, math solutions and Wikipedia in 20 languages. A
token's score is a weighted sum of its frequency in each source, weighted
toward chat, code and the most used languages. On held-out rows, the top
32k ids cover 94% of English chat tokens (81% when every source counts the
same), 94% of code, 98% of math and 85% of Chinese chat; Wikipedia in other
languages loses a few points. Outputs never change, only how often a draft
is accepted.

Writes the ranking (one token id per line, best first; special tokens
first) to --out.

  uvx --with tokenizers python3 tools/draft_vocab/rank_tokens.py \
      --cache /tmp/draft-vocab --out /tmp/draft-vocab/ranking.txt
"""

import argparse
import collections
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

from tokenizers import Tokenizer

TOKENIZER_URL = (
    "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer.json")
ROWS_URL = "https://datasets-server.huggingface.co/rows"
WIKI_LANGS = [
    "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "ar", "vi",
    "id", "tr", "pl", "nl", "hi", "th", "uk", "fa"
]


def assistant_turns(row):
    return "\n".join(m["content"] for m in row["messages"]
                     if m["role"] == "assistant")


# name: (dataset, config, split, text of a row, weight, pages of rows)
SOURCES = {
    f"wiki-{lang}": ("wikimedia/wikipedia", f"20231101.{lang}", "train",
                     lambda row: row["text"], 0.5, 5)
    for lang in WIKI_LANGS
}
SOURCES.update({
    "wiki-en": ("wikimedia/wikipedia", "20231101.en", "train",
                lambda row: row["text"], 4, 10),
    "wiki-zh": ("wikimedia/wikipedia", "20231101.zh", "train",
                lambda row: row["text"], 2, 5),
    "wiki-ja": ("wikimedia/wikipedia", "20231101.ja", "train",
                lambda row: row["text"], 1, 5),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "default", "train_sft",
                  assistant_turns, 8, 30),
    "code-feedback": ("m-a-p/CodeFeedback-Filtered-Instruction", "default",
                      "train", lambda row: row["answer"], 4, 20),
    "alpaca-zh": ("shibing624/alpaca-zh", "default", "train",
                  lambda row: row["output"], 4, 20),
    "numina-math": ("AI-MO/NuminaMath-CoT", "default", "train",
                    lambda row: row["solution"], 2, 10),
    "dolly-ja": ("kunishou/databricks-dolly-15k-ja", "default", "train",
                 lambda row: row["output"], 2, 10),
})


def get_json(url, attempts=10):
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


def fetch(cache, name, rows_per_page):
    """Rows spread evenly over the split, cached as JSON lines of text."""
    dataset, config, split, text_of, _, pages = SOURCES[name]
    path = cache / f"{name}-{pages}x{rows_per_page}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.open(encoding="utf-8")]
    query = {"dataset": dataset, "config": config, "split": split}
    first = get_json(ROWS_URL + "?" + urllib.parse.urlencode(
        {**query, "offset": 0, "length": 1}))
    total = first["num_rows_total"]
    texts = []
    for page in range(pages):
        offset = page * max(total // pages, rows_per_page)
        offset = min(offset, max(total - rows_per_page, 0))
        data = get_json(ROWS_URL + "?" + urllib.parse.urlencode(
            {**query, "offset": offset, "length": rows_per_page}))
        texts += [text_of(r["row"]) for r in data["rows"]]
    with path.open("w", encoding="utf-8") as out:
        for text in texts:
            out.write(json.dumps(text, ensure_ascii=False) + "\n")
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=100)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.cache / "tokenizer.json"
    if not tokenizer_path.exists():
        urllib.request.urlretrieve(TOKENIZER_URL, tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    score = collections.defaultdict(float)
    for name, source in SOURCES.items():
        weight = source[4]
        texts = fetch(args.cache, name, args.rows_per_page)
        counts = collections.Counter()
        for encoding in tokenizer.encode_batch(texts,
                                               add_special_tokens=False):
            counts.update(encoding.ids)
        total = sum(counts.values())
        print(f"{name}: {len(texts)} texts, {total} tokens, "
              f"{len(counts)} distinct")
        for token, count in counts.items():
            score[token] += weight * count / total

    # Added tokens (chat template, thinking, tool calls) rank first.
    special = sorted(tokenizer.get_added_tokens_decoder())
    special_set = set(special)
    ranked = special + [
        t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        if t not in special_set
    ]
    args.out.write_text("".join(f"{t}\n" for t in ranked))
    print(f"{len(ranked)} ranked tokens ({len(special)} special) -> "
          f"{args.out}")


if __name__ == "__main__":
    main()
