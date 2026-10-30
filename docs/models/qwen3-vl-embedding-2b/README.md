# Qwen3-VL-Embedding-2B

A text embedding model: one 2,048-value unit vector per input, for search
and retrieval (mcp-ragdown, Engrafo). It runs in its own package,
`src/models/qwen3vl_embed_2b/`, on its own NPU program (a bfp16 GEMM built
with its shapes); nothing is shared with the Qwen3.5 family. See
[benchmarks](BENCHMARKS.md), [quality](QUALITY.md) and
[experiments](EXPERIMENTS.md).

Text only on the NPU and in the server for now. Images have a CPU fp32
reference (`vision.hpp`, `Reference::EmbedImage`); the forward pass, costs
and plan are in [VISION.md](VISION.md). Video is not run.

## Weights

Tested with [`mradermacher/Qwen3-VL-Embedding-2B-GGUF`](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF),
`Qwen3-VL-Embedding-2B.f16.gguf` (3,447,351,456 bytes): llama.cpp's
conversion of [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
(Apache-2.0), the text model as arch `qwen3vl` (the vision tower is the
repo's separate mmproj file, not needed here). The original weights are
BF16; F16 holds them with no measurable loss (QUALITY.md: the CPU reference
on this file matches HF to cosine 1.000000). BF16, F32 and quantized GGUFs
load too; every matrix is rounded to bfp16 at load either way.

## Architecture (from the HF config)

| | |
|---|---|
| Layers | 28 |
| Hidden | 2048 |
| FFN (SwiGLU) | 6144 |
| Attention | 16 query heads over 8 KV heads, head dim 128; q/k RMS-normed per head; RoPE over the whole head, base 5,000,000 (interleaved MRoPE sections [24, 20, 20]: for text every section has the same position, which is plain RoPE) |
| Vocab | 151,936 (the Qwen2/Qwen3 tokenizer, GGUF pre-tokenizer `qwen2`), tied embeddings |
| RMS eps | 1e-6 |
| Context | 262,144 trained; 8,192 by default here (the reference embedder's limit) |

## What gets embedded

As the model's reference embedder (`scripts/qwen3_vl_embedding.py`) builds
it, checked against HF by the oracle (QUALITY.md):

```
<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>
```

- The instruction defaults to `Represent the user's input.`; a given one is
  trimmed and gets a `.` unless it ends in punctuation. Qwen recommends a
  task instruction for queries (e.g. "Retrieve passages that answer the
  question") and the default for documents.
- An empty text embeds `NULL`.
- The embedding is the final hidden state (after the last RMSNorm) of the
  trailing `<|endoftext|>`, L2-normalized.
- Matryoshka: the first n values (n from 64 to 2048), renormalized, are a
  smaller embedding.
- Inputs longer than the context limit are refused, not truncated (HF
  truncates from the right, which drops the pooled token).

## How an input runs

Orange runs on the NPU, blue on the CPU. Inputs are packed back to back into
a batch of up to 8,192 rows (a longer input runs alone), and the batch goes
through each layer at once:

1. RMSNorm, to bf16 (CPU) → **q | k | v projection, 2048 → 4096** (NPU)
2. q/k RMSNorm, RoPE, causal attention within each input (CPU)
3. **output projection, 2048 → 2048** (NPU), residual add (CPU)
4. RMSNorm (CPU) → **gate | up, 2048 → 12288** (NPU) → SwiGLU (CPU)
5. **down, 6144 → 2048** (NPU), residual add (CPU)

Each projection runs in GEMM chunks of up to 1,024 rows with two calls in
flight: the CPU writes the next chunk's input and reads the previous one's
output while the NPU multiplies. Pooling (final RMSNorm, L2) is on the CPU.

## Usage

```sh
gufo-npu embed --model qwen3-vl-embedding-2b-bf16.gguf \
  --text "How much did I spend at the hardware store?" \
  --text "Corner Hardware: screws and a hammer, 23.40 EUR" --similarity

gufo-npu serve --model qwen3-vl-embedding-2b-bf16.gguf --port 8081
curl -s localhost:8081/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["first passage", "second passage"], "dimensions": 512}'
```

`POST /v1/embeddings` is described in [docs/SERVING.md](../../SERVING.md).
`gufo-npu bench embed --model GGUF` times warm batches per input length.

## Tests

| Test | Label | Needs |
|---|---|---|
| `qwen3vl_embed_2b_ops_test` | `qwen3vl_embed_2b` | nothing (hosted CI): template, instruction rule, packing, attention against a direct softmax, pooling, Matryoshka |
| `qwen3vl_embed_2b_vision_test` | `qwen3vl_embed_2b` | nothing (hosted CI): smart_resize, Pillow's bicubic (fixtures), patch order, position-embedding taps, MRoPE positions, MRoPE attention = text attention at text positions |
| `qwen3vl_tokenizer_test` | `qwen3vl_embed_2b;model` | the GGUF: HF tokenizer fixtures |
| `qwen3vl_embed_2b_reference_oracle_test` | `qwen3vl_embed_2b;model` | GGUF + oracle: ids, CPU reference vs HF fp32 (cosine ≥ 0.999), retrieval ranks |
| `qwen3vl_embed_2b_vision_oracle_test` | `qwen3vl_embed_2b;model` | GGUF + mmproj + oracle `--images`: patch grid, ids, CPU vision + text reference vs HF fp32 (cosine ≥ 0.999) |
| `qwen3vl_embed_2b_gemm_test` | `qwen3vl_embed_2b;npu;kernels` | NPU: every GEMM shape vs a double reference (relative error < 1e-2) |
| `qwen3vl_embed_2b_npu_embed_test` | `qwen3vl_embed_2b;npu;model` | NPU + GGUF + oracle: cosine ≥ 0.99, retrieval ranks, alone = batched |

Variables: `GUFO_NPU_QWEN3VL_EMBED_2B_GGUF` (the BF16 GGUF) and
`GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE` (the output of
`tools/reference/qwen3vl_embedding_oracle.py`, with `--images` for the
vision test), plus `GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ` (the mmproj GGUF,
e.g. `Qwen3-VL-Embedding-2B.mmproj-f16.gguf`) for images.
