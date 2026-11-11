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

## Images on the NPU

Measured 2026-11-11 on the Strix Halo box from a source build (the
toolchain image) of the NPU vision branch: `qwen3vl_embed_2b_npu_vision_test`
with the F16 GGUF, the F16 mmproj and a fresh oracle run with `--images`
(same pins as above). Two runs of the whole `qwen3vl_embed_2b` label gave
the same numbers; all 8 tests pass, including the text NPU test (cosine to
HF min 0.99808, mean 0.99881, alone = batched), the GEMM test at all 30
sequences (total relative error 9.5e-3 on each) and the CPU reference
tests (1.000000).

| Check | Tolerance | Result |
|---|---|---|
| NPU vision + NPU text vs HF fp32, the 3 cards | cosine ≥ 0.99 per input | **0.99463 to 0.99673** (96x64 0.99673, 128x96 with text 0.99557, 100x70 resized 0.99463) |
| NPU vs the CPU reference, 640x480 card (1,200 patches, two GEMM chunks; 329 tokens) | cosine ≥ 0.99 | **0.99645** |
| NPU vision features vs the CPU reference | cosine ≥ 0.95 per block (a bug check) | 0.986 to 0.9999 |

The features' error grows with depth and sits in a few tokens. On the
640x480 card, per feature block (cosine as a whole; worst token; tokens of
300 below 0.99):

| Block | Cosine | Worst token | Below 0.99 |
|---|---|---|---|
| DeepStack 0 (after vision block 5) | 0.99980 | 0.9995 | 0 |
| DeepStack 1 (after block 11) | 0.99533 | 0.422 | 12 |
| DeepStack 2 (after block 17) | 0.99585 | 0.748 | 33 |
| Merger (after block 23) | 0.98970 | 0.783 | 64 |

The same tokens are the worst in DeepStack 1 and 2; their output norms
are ordinary (9.5 against a median of 12.9), so it is not a large output
losing precision. The text model averages them out (embedding 0.996 to
the reference), but the cause is not pinned down: it is the bfp16 rounding
of weights and activations (the only difference from the reference), and
which product contributes most is untested (VISION.md, open questions).
Page-sized images are not in the HF oracle; the page is timed only.

## Image inputs through the CLI and the server

Measured 2026-11-11 on the Strix Halo box from a source build (the
toolchain image) of the image-API branch, with the F16 GGUF, the F16
mmproj and the same `--images` oracle as above. The oracle's three cards
were written as PNGs (the decoder's PNG path; `image_test` checks every
PNG and JPEG kind against Pillow byte for byte) and embedded three ways:

| Path | Token counts = HF's | Cosine to HF fp32 (bar 0.99) |
|---|---|---|
| `qwen3vl_embed_2b_engine_image_test` (the server's engine, `--mmproj`) | 3/3 | 0.996728, 0.995573, 0.994634 |
| `gufo-npu embed --mmproj --image` (the two uncaptioned cards) | 2/2 | 0.996728, 0.994634 |
| `gufo-npu serve --mmproj`, `/v1/embeddings` data URIs (caption and instruction as HF's) | 3/3 | 0.996728, 0.995573, 0.994634 |

The same values as `qwen3vl_embed_2b_npu_vision_test`, which calls
`NpuVision` and `NpuEmbedder::EmbedImage` directly, so the CLI and the
server add nothing to the error. An image in one request with a text input
gets the vector it gets alone (cosine 1.0000000), and so do the engine
test's text and three images in one batch. The same build passes
`image_test`, `serve_embeddings_test`, `qwen3vl_embed_2b_vision_test` and,
unchanged, `qwen3vl_embed_2b_npu_embed_test` and
`qwen3vl_embed_2b_npu_vision_test`.

## Known differences from the reference embedder

- Inputs over the context limit are refused rather than truncated.
- Image files are decoded to Pillow's pixels; a JPEG could still differ
  from a Pillow built on another libjpeg-turbo release (the fixtures match
  on the toolchain image's and on Pillow 12.1's). EXIF orientation is not
  applied, as the reference embedder does not apply it. Video is not
  supported.
