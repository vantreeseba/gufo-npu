#!/usr/bin/env python3
"""llama.cpp oracle for the Qwen3.6-35B-A3B CPU reference.

Runs llama.cpp on the CPU (llama-cpp-python, pinned in requirements-llama.txt)
over the same GGUF as the reference, on fixed prompts with teacher forcing,
and writes, per prompt, to --out:
  <name>.text          the prompt text (chat prompts rendered with the GGUF's
                       chat template)
  <name>.ids.json      token ids (llama.cpp's tokenizer)
  <name>.logits.f32    [tokens, vocab] float32, the logits at every position
  <name>.hidden.f32    [layers + 1, tokens, hidden] float32: the embeddings,
                       then each layer's output (graph tensors `l_out-<il>`)
plus manifest.json with shapes, library versions and run settings.

The whole prompt is one batch with an fp32 KV cache, flash attention off and
no weight repacking, so the weights stay in the memory-mapped GGUF.

  python -I qwen36_llama_oracle.py --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \\
      --out oracle-35b
"""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import jinja2
import jinja2.sandbox
import llama_cpp
import numpy as np

# requirements-llama.txt; this release vendors llama.cpp at LLAMA_CPP_COMMIT.
LLAMA_CPP_PYTHON = "0.3.36"
LLAMA_CPP_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"

# The prompts of qwen35_oracle.py, plus a longer one (over 200 tokens).
PROMPTS = {
    "chat": [{"role": "user", "content": "Write one sentence about the moon."}],
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "prose": (
        "The river ran past old stone bridges and quiet fields of wheat. "
        "In the evening the light turned gold, and the farmers walked home "
        "along the water, talking about the harvest and the coming rain."
    ),
    "long": (
        "The lighthouse keeper had kept the same routine for thirty-one years. "
        "Every evening, an hour before sunset, he climbed the one hundred and "
        "twelve steps of the tower, wiped the salt from the great lens with a "
        "soft cloth, trimmed the wick, and checked the clockwork that turned "
        "the light. Ships no longer needed him the way they once had; "
        "satellites told captains where they were to within a few meters, and "
        "the harbor authority had sent three letters suggesting that the light "
        "could be automated. He answered each letter politely and changed "
        "nothing. On clear nights he could see the lights of the fishing boats "
        "strung along the horizon like beads, and on foggy nights he listened "
        "to the horn and counted the seconds between its calls. His daughter, "
        "who lived in the city and worked on software for shipping companies, "
        "visited every summer. She teased him about the letters, but she also "
        "brought him a small weather station, which he mounted on the gallery "
        "rail and checked every morning with his coffee. When the storm came "
        "in October, the power failed across the whole coast, and the "
        "automated beacons to the north and south went dark one after another."
    ),
}


@llama_cpp.llama_log_callback
def LOG_WARNINGS(level, text, _user_data):
    if level in (3, 4):  # GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR
        print(text.decode(errors="replace"), end="", flush=True)


def meta(model, key: str) -> str:
    size = llama_cpp.llama_model_meta_val_str(model, key.encode(), None, 0)
    if size < 0:
        raise KeyError(key)
    buf = ctypes.create_string_buffer(size + 1)
    llama_cpp.llama_model_meta_val_str(model, key.encode(), buf, size + 1)
    return buf.value.decode()


def render_chat(template: str, messages) -> str:
    """Renders like transformers' apply_chat_template(add_generation_prompt=True)."""

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=True)


def tokenize(vocab, text: str) -> list[int]:
    data = text.encode()
    tokens = (llama_cpp.llama_token * (len(data) + 16))()
    n = llama_cpp.llama_tokenize(vocab, data, len(data), tokens, len(tokens), False, True)
    if n < 0:
        raise RuntimeError("tokenize failed")
    return list(tokens[:n])


def piece(vocab, token: int) -> str:
    buf = ctypes.create_string_buffer(256)
    n = llama_cpp.llama_token_to_piece(vocab, token, buf, len(buf), 0, True)
    return buf.raw[:max(n, 0)].decode(errors="replace")


class Capture:
    """Copies graph tensors out as llama.cpp computes them (the eval callback):
    the embeddings and every layer's output, by their graph names."""

    def __init__(self):
        lib = Path(llama_cpp.__file__).parent / "lib" / "libggml-base.so"
        self.ggml = ctypes.CDLL(str(lib))
        self.ggml.ggml_get_name.restype = ctypes.c_char_p
        self.ggml.ggml_get_name.argtypes = [ctypes.c_void_p]
        self.ggml.ggml_nelements.restype = ctypes.c_int64
        self.ggml.ggml_nelements.argtypes = [ctypes.c_void_p]
        self.ggml.ggml_nbytes.restype = ctypes.c_size_t
        self.ggml.ggml_nbytes.argtypes = [ctypes.c_void_p]
        self.ggml.ggml_backend_tensor_get.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.tensors: dict[str, np.ndarray] = {}
        self.callback = llama_cpp.ggml_backend_sched_eval_callback(self._eval)

    @staticmethod
    def wanted(name: str) -> bool:
        return name == "model.input_embed" or name.startswith("l_out-")

    def _eval(self, tensor, ask, _user_data) -> bool:
        name = self.ggml.ggml_get_name(tensor).decode()
        if ask:
            return self.wanted(name)
        if self.wanted(name):
            if ctypes.c_int.from_address(tensor).value != llama_cpp.GGML_TYPE_F32:
                raise RuntimeError(f"{name} is not F32")
            out = np.empty(self.ggml.ggml_nelements(tensor), dtype=np.float32)
            self.ggml.ggml_backend_tensor_get(tensor, out.ctypes.data, 0,
                                              self.ggml.ggml_nbytes(tensor))
            self.tensors[name] = out
        return True

    def hidden(self, tokens: int) -> np.ndarray:
        """[layers + 1, tokens, hidden]: embeddings, then each layer's output."""
        layers = sorted((int(n.split("-")[1]) for n in self.tensors if n.startswith("l_out-")))
        if layers != list(range(len(layers))):
            raise RuntimeError(f"missing layer outputs: {layers}")
        names = ["model.input_embed"] + [f"l_out-{il}" for il in layers]
        return np.stack([self.tensors[n].reshape(tokens, -1) for n in names])


def logits_for(model, ids: list[int], n_vocab: int, threads: int,
               capture: Capture | None, step: int) -> np.ndarray:
    """Teacher-forced logits for every position of `ids` in a fresh context,
    `step` tokens per llama_decode call."""
    cparams = llama_cpp.llama_context_default_params()
    if capture is not None:
        cparams.cb_eval = capture.callback
        cparams.cb_eval_user_data = None
    cparams.n_ctx = max(256, len(ids))
    cparams.n_batch = cparams.n_ctx
    cparams.n_ubatch = cparams.n_ctx
    cparams.n_seq_max = 1
    cparams.n_threads = threads
    cparams.n_threads_batch = threads
    cparams.type_k = llama_cpp.GGML_TYPE_F32
    cparams.type_v = llama_cpp.GGML_TYPE_F32
    cparams.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
    cparams.embeddings = False
    ctx = llama_cpp.llama_init_from_model(model, cparams)
    if not ctx:
        raise RuntimeError("llama_init_from_model failed")
    batch = llama_cpp.llama_batch_init(step, 0, 1)
    out = np.empty((len(ids), n_vocab), dtype=np.float32)
    try:
        for begin in range(0, len(ids), step):
            chunk = ids[begin:begin + step]
            batch.n_tokens = len(chunk)
            for i, token in enumerate(chunk):
                batch.token[i] = token
                batch.pos[i] = begin + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1
            if llama_cpp.llama_decode(ctx, batch) != 0:
                raise RuntimeError("llama_decode failed")
            for i in range(len(chunk)):
                row = llama_cpp.llama_get_logits_ith(ctx, i)
                out[begin + i] = np.ctypeslib.as_array(row, shape=(n_vocab,))
        return out
    finally:
        llama_cpp.llama_batch_free(batch)
        llama_cpp.llama_free(ctx)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if llama_cpp.__version__ != LLAMA_CPP_PYTHON:
        raise SystemExit(f"need llama-cpp-python {LLAMA_CPP_PYTHON}, have {llama_cpp.__version__}")
    args.out.mkdir(parents=True, exist_ok=True)

    llama_cpp.llama_log_set(LOG_WARNINGS, None)
    llama_cpp.llama_backend_init()
    mparams = llama_cpp.llama_model_default_params()
    mparams.n_gpu_layers = 0
    mparams.load_mode = llama_cpp.LLAMA_LOAD_MODE_MMAP
    mparams.use_extra_bufts = False  # no repacked copy of the weights
    mparams.load_mtp = False
    start = time.monotonic()
    model = llama_cpp.llama_model_load_from_file(str(args.model).encode(), mparams)
    if not model:
        raise RuntimeError(f"cannot load {args.model}")
    vocab = llama_cpp.llama_model_get_vocab(model)
    n_vocab = llama_cpp.llama_vocab_n_tokens(vocab)
    template = meta(model, "tokenizer.chat_template")
    print(f"loaded in {time.monotonic() - start:.1f} s; arch {meta(model, 'general.architecture')}, "
          f"vocab {n_vocab}")

    manifest = {
        "model_file": args.model.name,
        "llama_cpp_python": llama_cpp.__version__,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "system_info": llama_cpp.llama_print_system_info().decode().strip(),
        "settings": {"kv_cache": "f32", "flash_attn": False, "repack": False,
                     "threads": args.threads, "one_batch": True},
        "prompts": {},
    }
    for name, prompt in PROMPTS.items():
        text = render_chat(template, prompt) if isinstance(prompt, list) else prompt
        ids = tokenize(vocab, text)
        capture = Capture()
        start = time.monotonic()
        logits = logits_for(model, ids, n_vocab, args.threads, capture, len(ids))
        seconds = time.monotonic() - start
        hidden = capture.hidden(len(ids))
        # llama.cpp against itself, one token per call: other kernels and
        # accumulation orders, so other 8-bit roundings of the activations.
        stepwise = logits_for(model, ids, n_vocab, args.threads, None, 1)
        self_err = float(np.linalg.norm((stepwise - logits).astype(np.float64)) /
                         np.linalg.norm(logits.astype(np.float64)))
        self_argmax = int((stepwise.argmax(axis=1) == logits.argmax(axis=1)).sum())
        (args.out / f"{name}.text").write_text(text)
        (args.out / f"{name}.ids.json").write_text(json.dumps(ids))
        logits.tofile(args.out / f"{name}.logits.f32")
        hidden.tofile(args.out / f"{name}.hidden.f32")
        top = np.argsort(-logits[-1])[:5].tolist()
        manifest["prompts"][name] = {
            "tokens": len(ids),
            "vocab": n_vocab,
            "hidden_layers": int(hidden.shape[0]),
            "hidden_size": int(hidden.shape[2]),
            "seconds": round(seconds, 2),
            "last_top5": top,
            "last_top5_text": [piece(vocab, t) for t in top],
            "one_token_per_call": {"logits_rel_err": self_err, "argmax_match": self_argmax},
        }
        print(f"{name}: {len(ids)} tokens in {seconds:.1f} s, last-position top-5 {top}; "
              f"one token per call: logits rel err {self_err:.2e}, "
              f"argmax {self_argmax}/{len(ids)}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    llama_cpp.llama_model_free(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
