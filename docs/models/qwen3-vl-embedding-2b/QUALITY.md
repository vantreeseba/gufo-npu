# Qwen3-VL-Embedding-2B: quality

The oracle is HF transformers in float32 on the model's own weights
(`tools/reference/qwen3vl_embedding_oracle.py`): 3 queries with a retrieval
instruction, 6 documents (one German), and inputs for the default and custom
instructions, Unicode, code and a ~1,500-token text that crosses a GEMM
chunk. The script also checks that the chat template renders exactly what
`input.cpp` builds and that the tokenizer appends `<|endoftext|>`.

Measured 2026-10-30 on the Strix Halo box, image `sha-62acdfc`, the F16
GGUF (README), through `gufo-npu embed` on the oracle's 15 inputs (20 to
2,088 tokens):

| Check | Tolerance | Result |
|---|---|---|
| Token counts = HF's (same template, same tokenizer) | exact | 15/15 equal |
| CPU reference vs HF fp32 | cosine ≥ 0.999 per input | **1.000000** on every input |
| NPU vs HF fp32 | cosine ≥ 0.99 per input | **0.99794 to 0.99916** (the 2,088-token input 0.99827) |
| Retrieval: each query's first document | = HF's | same (0, 1, 3); scores within 0.012 of HF's |

The NPU's gap is the GEMMs' rounding (weights and activations to bfp16,
outputs to bf16); it does not change any ranking here: the right document
leads the next by 0.16 or more on both sides.

Not run yet (they need a source build on the box): the C++ model tests
(`qwen3vl_tokenizer_test`, `qwen3vl_embed_2b_reference_oracle_test`,
`qwen3vl_embed_2b_gemm_test`, `qwen3vl_embed_2b_npu_embed_test`, which adds
the alone-vs-batched check). `qwen3vl_embed_2b_ops_test` and
`serve_embeddings_test` pass in CI.

## Images (CPU reference)

Measured 2026-10-30 on the Strix Halo box from a source build (the
toolchain image) of the image-input branch: `qwen3vl_embed_2b_vision_oracle_test`
with the F16 GGUF, the F16 mmproj (README) and the oracle run with
`--images` (transformers 5.21, torch 2.13 CPU, torchvision 0.28, Pillow
13.0). Three generated test cards: 96x64 alone and 128x96 with text (both
kept at their size), and 100x70 alone, which smart_resize brings to
96x64 through Pillow's bicubic.

| Check | Tolerance | Result |
|---|---|---|
| Patch grid = HF's `image_grid_thw` | exact | 3/3 equal (4x6, 6x8, 4x6) |
| Token ids = HF's processor | exact | 3/3 equal (28, 41, 26 tokens) |
| CPU vision + text reference vs HF fp32 | cosine ≥ 0.999 per input | **1.000000** on all 3 |

The same build's `qwen3vl_embed_2b_reference_oracle_test` (text, after
the reference's layer loop was shared with the image path): cosine
1.000000 on all 15 inputs, alone = batched. Page-sized images are not in
the oracle yet: the fp32 CPU vision reference is slow at 7,200 patches.

## Known differences from the reference embedder

- Inputs over the context limit are refused rather than truncated.
- Images run only in the CPU reference (no NPU path, no API yet); video is
  not supported.
