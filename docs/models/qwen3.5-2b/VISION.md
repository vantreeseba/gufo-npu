# Qwen3.5-2B: image input

The plan for image prompts on Qwen3.5-2B (chat with images, and page OCR as
Engrafo uses it): what the model does with an image, what that costs, and
the stages to run it here. Sources, as of 2026-11: HF transformers 5.21.0
(`models/qwen3_5/modeling_qwen3_5.py`, `models/qwen2_vl/image_processing_qwen2_vl.py`,
`image_processing_backends.py`, `models/qwen3_vl/processing_qwen3_vl.py`),
torchvision 0.28 (`transforms/v2/functional/_geometry.py`), the model repo's
`config.json`, `preprocessor_config.json` and `chat_template.jinja`
(`Qwen/Qwen3.5-2B`), and Unsloth's `mmproj-F16.gguf` for the 2B.

The vision tower is the Qwen3-VL ViT that Qwen3-VL-Embedding-2B already runs
here, without DeepStack; [the embedder's VISION.md](../qwen3-vl-embedding-2b/VISION.md)
describes the shared parts in full and this page only gives the
differences.

Status: stage 1 (the HF oracle, `tools/reference/qwen35_2b_vl_oracle.py`,
[QUALITY.md](QUALITY.md)) is done; the other stages below are plans.

## Input

Image parts are user-message content items (`{"type": "image"}`, or any
item with an `image` or `image_url` key); system messages cannot hold
images. The chat template renders each one as
`<|vision_start|><|image_pad|><|vision_end|>` in place, with no separator
from the text items around it (`add_vision_id`, off by default, would
prefix `Picture N: `), and trims the message's rendered content. The
processor then repeats `<|image_pad|>` once per merged token. One image and a
question, thinking off (the template's default, `<think>\n\n</think>\n\n`):

```
<|im_start|>user\n<|vision_start|> <|image_pad|> x N <|vision_end|>{question}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

Token ids: `<|vision_start|>` 248053, `<|vision_end|>` 248054,
`<|image_pad|>` 248056 (`<|video_pad|>` 248057, out of scope). A request
with images is still a chat request: the continuation is decoded as for
text.

## Preprocessing

`preprocessor_config.json`: `Qwen2VLImageProcessorFast` (torchvision
backend), patch 16, merge 2, temporal patch 2, mean = std = 0.5,
`shortest_edge` 65536 and `longest_edge` 16777216 (64 to 16,384 image
tokens). gufo-npu serves with `--max-image-tokens` 1800 by default, so the
bounds are 65,536 to 1,843,200 pixels; the oracle runs with those.

1. RGB (the processor flattens RGBA on white, as the embedder's
   `FlattenAlpha`).
2. `smart_resize(h, w, factor=32, min_pixels, max_pixels)`:
   `h' = round(h/32)*32` (Python round, half to even), same for w; if
   `h'w' > max`: `beta = sqrt(hw/max)`, `h' = max(32, floor(h/beta/32)*32)`;
   if `h'w' < min`: `beta = sqrt(min/hw)`, `h' = ceil(h*beta/32)*32`.
   Aspect ratio above 200 is an error. Two differences from the embedder's
   `SmartResize` (qwen_vl_utils): the minimum is 65,536 pixels, not 4,096,
   so small images are enlarged to at least 64 tokens (a 100 x 70 card
   becomes 320 x 224, 70 tokens), and the first rounding has no `max(32, ...)`
   (it only matters for a side under 16 pixels). `SmartResize` takes the
   bounds as arguments already.
3. Resize, when the size changes: `torchvision.transforms.v2.functional.resize`
   on the uint8 CHW tensor, bicubic, `antialias=True`. On the CPU this is
   torch's native uint8 antialiased kernel (the Pillow-SIMD port: separable,
   a = -0.5, support 2 x scale when shrinking, fixed-point int16 weights,
   rounded to uint8 between the passes), not a float resize.
4. `x / 255`, then `(x - 0.5) / 0.5`; patchify as the embedder: rows in
   2 x 2 merge-block order, each `[channel][temporal 2][y 16][x 16]` = 1536
   values with the frame in both temporal slots. `image_grid_thw =
   (1, h'/16, w'/16)`, N = h'w' / 1024 merged tokens.

**Resize parity, measured** (`qwen35_2b_vl_oracle.py --resize-only`, and
the oracle's manifest; uint8 values of the processor's resized image,
recovered from `pixel_values`, against Pillow's `Image.resize` bicubic,
which `ResizeBicubic` reproduces):

| Image | Resized to (w x h) | Tokens | max abs | values differing |
|---|---|---|---|---|
| 256 x 192 card (enlarged) | 320 x 224 | 70 | 1 | 0.037% |
| 333 x 251 text card (shrunk) | 320 x 256 | 80 | 1 | 0.044% |
| 288 x 224 card (enlarged) | 320 x 256 | 80 | 2 | 0.058% |
| 400 x 160 text card (enlarged) | 416 x 192 | 78 | 1 | 0.018% |
| 100 x 70 card | 320 x 224 | 70 | 1 | 0.066% |
| 1000 x 750 photo-like card | 992 x 736 | 713 | 2 | 0.21% |
| letter page, 1275 x 1650 | 1184 x 1536 | 1776 | 2 | 0.041% |
| A4 page, 1654 x 2339 | 1120 x 1600 | 1750 | 1 | 0.028% |
| A4 page, 2480 x 3508 | 1120 x 1600 | 1750 | 2 | 0.010% |
| 320 x 256, 1280 x 1440 | kept | 80, 1800 | (no resize) | |

A float antialiased bicubic (`F.interpolate` on float32, rounded once)
differs from the processor by up to 25 levels on the same images: the
processor's kernel is the fixed-point one, which Pillow's is close to.
Through the vision tower, Pillow-resized inputs give features at cosine
0.99973 or more per token to the processor's (relative L2 5.9e-3 at most,
on the four resized oracle images), well under what bfp16 GEMMs move them.
So the embedder's Pillow resize is kept; matching torch's int16 weights
exactly is not worth a second resampler unless a test says so.

Pages at the cap: 1280 x 1440 is exactly 1,843,200 pixels and keeps its
size (grid 90 x 80, 1,800 tokens); a 150-dpi letter scan is shrunk to
1184 x 1536 (1,776 tokens), any A4 scan at 200 dpi or more to 1120 x 1600
(1,750 tokens).

## Vision tower

As the embedder's (24 blocks, width 1024, 16 heads of 64, ffn 4096,
`gelu_pytorch_tanh`, LayerNorm eps 1e-6, patch embed as one K = 1536 (or
summed K = 768) product, the 48 x 48 learned position table bilinearly
resampled with `align_corners=True`, 2D RoPE theta 10000, full
bidirectional attention per image, merger LayerNorm(1024) -> 4096 -> exact
GELU -> 2048), with **no DeepStack**: `deepstack_visual_indexes` is empty,
the merger's output is the only feature, and nothing is added inside the
text layers. Several images in one prompt run as separate attention
sequences (`cu_seqlens`); their features are concatenated in prompt order.

## Into the text model

- The `<|image_pad|>` rows of the input embeddings are replaced by the
  merger's output, in order (`masked_scatter`).
- **MRoPE positions** (`get_rope_index`): text tokens advance one position
  each, the three axes equal. An image whose first pad token is at position
  `s` gives its merged token (r, c) `(t, h, w) = (s, s + r, s + c)`; the
  next token (`<|vision_end|>`) is at `s + max(grid_h, grid_w) / 2`. Only
  pads are image tokens; `<|vision_start|>` and `<|vision_end|>` are text.
  The page (grid 90 x 80, 1,800 pads from index 4) takes positions 4 to 48,
  and `<|vision_end|>` is at position 49 although its index is 1,804.
- **Interleaved MRoPE** on the 64 rotated dims of the six full-attention
  layers (`partial_rotary_factor` 0.25 of head 256, theta 1e7, sections
  [11, 11, 10]): of the 32 frequencies, i uses the h position when
  `i % 3 == 1` (i < 33), w when `i % 3 == 2` (i < 30), t otherwise;
  rotate-half pairs (i, i + 32) within the 64 dims. With t = h = w this is
  the RoPE the text path runs. The 18 DeltaNet layers take no positions.
- **The RoPE position diverges from the KV index.** After the prompt,
  `rope_delta = max(position) + 1 - prompt_len` (negative with images:
  -1,755 for the page) and every later token rotates at `index + rope_delta`
  on all three axes. Decode needs that per-sequence offset next to
  `DecodeState::position` (the KV index). The cached keys are already
  rotated, so only new tokens need it.

## mmproj tensors

Unsloth's `mmproj-F16.gguf` for the 2B: `clip.projector_type` =
`qwen3vl_merger`, `clip.vision.*` keys as the embedder's (`block_count` 24,
`embedding_length` 1024, `feed_forward_length` 4096, `head_count` 16,
`patch_size` 16, `projection_dim` 2048, `spatial_merge_size` 2,
`image_mean`/`image_std` 0.5, `layer_norm_epsilon` 1e-6,
`is_deepstack_layers` all false). 298 tensors: the embedder's table without
the `v.deepstack.*` rows (weights F16, biases, norms and the position table
F32). The embedder's mmproj loader finds DeepStack blocks by their
tensors (`v.deepstack.{i}.fc1.weight`), so this file gives an empty list.

## Where each piece runs

Strix Halo only, the NPU first (as the embedder's table): decoding the
file, smart_resize, the resize and patchify on the CPU (once per image,
byte work); every ViT and merger linear on the NPU GEMM; the ViT attention
on the NPU (the embedder's whole-array flash attention design); LayerNorm,
GELU, biases and residuals start on the CPU and move onto the NPU as the
embedder's fusion stage does. The text side is the 2B's NPU prefill
(projections on the NPU GEMM, DeltaNet and attention mixers on the CPU) and
NPU decode, unchanged except for embedding rows and positions.

## Cost of a page (1280 x 1440: 1,800 image tokens, 1,822 prompt tokens)

| Part | Estimate | From |
|---|---|---|
| decode PNG, resize, patchify | < 0.05 s | CPU, once |
| vision tower | ~2.2-2.4 s | the embedder's measured tower with NPU attention (same shapes, no DeepStack mergers) |
| prefill, 1,822 tokens | ~0.55 s | ~3,500 tok/s at this length (README: 0.58 s TTFT at 2,048) |
| decode | ~24 ms a token (~42 tok/s) | the 2B's decode; the oracle page (~330 words) is ~450 output tokens, ~11 s |

Decode dominates OCR-length outputs; the vision tower is the next part, so
the embedder's open work (one hardware context for the tower, fused
epilogues) pays off here too. The context changes the decode rate little:
attention runs in 6 layers only.

## Plan

1. **Oracle** (done): `tools/reference/qwen35_2b_vl_oracle.py`, HF fp32,
   five cases (three cards, two images in one prompt, a page at the cap),
   with pixels, grids, positions, features, teacher-forced logits and
   hidden states ([QUALITY.md](QUALITY.md)).
2. **Shared ViT library** `src/models/qwen3vl_vision/`: the embedder's
   preprocessing, mmproj loading (DeepStack optional), CPU fp32 tower and
   MRoPE position builder, split out of `qwen3vl_embed_2b` so both models
   use it; numerics unchanged (the embedder's vision tests must stay
   bit-identical). Its NPU parts (GEMM shapes, attention program) stay per
   model: each model keeps them in its own xclbins.
3. **Family CPU reference** (`src/models/qwen35`): input embedding rows in
   place of token embeddings, per-row (t, h, w) positions and interleaved
   MRoPE in the full-attention layers, the rope offset for later tokens.
   Check: hidden states and logits against the oracle (rows, per layer),
   the features against `features.f32` from `pixels.f32`.
4. **2B NPU path**: the ViT and merger (K, N) shapes in the 2B's prefill
   GEMM xclbin (`qwen35_2b_prefill`), so the tower and the prefill share a
   hardware context; the 2B's own copy of the ViT attention design;
   `NpuPrefill` taking embedding rows and per-row positions; a rope offset
   in `DecodeState`; MTP over image rows (its cache is filled from the
   prompt's hidden rows; see the risks). Checks: op tests, the
   teacher-forced argmax against the oracle's greedy rows, the page's
   transcription.
5. **API**: chat `image_url` content parts with base64 data URIs only (no
   fetching), `--mmproj` for the chat model, `--max-image-tokens` (default
   1800) setting the resize's maximum, the prompt cache keyed with the
   images' content (pad ids are all equal, so the token ids alone do not
   identify a prompt) and with the token cap.

## Open questions and risks

- **bfp16 drift per token**: the embedder's features keep cosine 0.995 or
  more pooled, but single tokens drift (64 of 300 merger tokens below 0.99
  on a 1,200-patch card). A pooled embedding averages that out; generation
  reads every image token, so OCR may lose characters where an embedding
  would not. Measure on the page first (transcription against the oracle's
  greedy rows), then find which product drifts most.
- **Resize parity**: Pillow against the processor is 1-2 levels on 0.01-0.2%
  of values and changes features by < 3e-4 in cosine; a test should keep
  that bounded on a page-sized shrink, not only cards. EXIF orientation is
  ignored by both.
- **Hardware contexts**: the embedder's tower switches between its GEMM and
  attention programs twice per block (~0.15 s an image). The 2B adds its
  decode program; putting the ViT shapes in the prefill xclbin avoids a
  third program, one tower program would remove the switches.
- **Image-token cap**: 1,800 tokens is what the oracle and the cost assume.
  Larger caps grow the ViT attention as n^2 in patches (the default
  `longest_edge`, 16,384 tokens = 65,536 patches, would be ~80 times a
  page's 5.1 TFLOP); smaller ones lose small print.
- **MTP over image rows**: the MTP layer pairs each hidden row with the
  next token's embedding; at image rows the next "token" is a pad. Whether
  it should see the pad's text embedding or the image feature, and which
  RoPE position its attention uses, has no HF reference (HF drops the MTP
  weights); judge by acceptance rate on image prompts.
- **Video** (`<|video_pad|>`, timestamps between frames) is out of scope.
