# Qwen3-VL-Embedding-2B: image input

How the model embeds an image (alone or with text), what that costs, and how
to run it here. Sources, as of 2026-10: HF transformers `main`
(`models/qwen3_vl/modeling_qwen3_vl.py`, `vision_utils.py`,
`models/qwen2_vl/image_processing_qwen2_vl.py`), the model repo's
`scripts/qwen3_vl_embedding.py` and `preprocessor_config.json`,
`qwen_vl_utils/vision_process.py` (Qwen3-VL repo), and llama.cpp `master`
(`conversion/qwen3vl.py`, `tools/mtmd/clip.cpp`, `tools/mtmd/models/qwen3vl.cpp`,
`gguf-py/gguf/constants.py`).

Status: stage 1 (CPU fp32 reference, `vision.hpp`) is in the package; the
NPU stages below are plans.

## Input

The reference embedder (`Qwen3VLEmbedder`) puts the image before the text in
the user message. The chat template renders an image as
`<|vision_start|><|image_pad|><|vision_end|>` and the processor repeats
`<|image_pad|>` once per merged token, so the token sequence is

```
<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n
<|vision_start|> <|image_pad|> x N <|vision_end|>{text}<|im_end|>\n
<|im_start|>assistant\n<|endoftext|>
```

(no newline around the vision tokens; `{text}` is omitted when there is
none, and "NULL" is used only when there is neither text nor image). The
embedding is still the last token's final hidden state, L2-normalized.
Token ids: `<|vision_start|>` 151652, `<|vision_end|>` 151653,
`<|image_pad|>` 151655.

## Preprocessing

The embedder resizes with `qwen_vl_utils.fetch_image` and calls the
processor with `do_resize=False`, so its numbers, not
`preprocessor_config.json`'s `max_pixels` (1310720), apply:

1. RGB (RGBA is composited onto white).
2. `smart_resize(h, w, factor=32, min_pixels=4096, max_pixels=1843200)`
   (factor = patch 16 x merge 2; 1843200 = 1800 x 32 x 32, so at most 1800
   image tokens): `h' = max(32, round(h/32)*32)` (Python round, half to
   even), same for w; if `h'w' > max`: `beta = sqrt(hw/max)`,
   `h' = floor(h/beta/32)*32`; if `h'w' < min`: `beta = sqrt(min/hw)`,
   `h' = ceil(h*beta/32)*32`. Aspect ratio above 200 is an error.
3. `PIL.Image.resize((w', h'))`: Pillow's default for RGB, antialiased
   bicubic (a = -0.5, support 2 x scale when shrinking), horizontal pass
   then vertical, each rounded to uint8 with 22-bit fixed-point
   coefficients. Skipped when the size is unchanged.
4. `x / 255`, then `(x - 0.5) / 0.5` per channel (mean = std = 0.5).
5. Patchify (`image_processing_qwen2_vl.patchify`): `grid_h = h'/16`,
   `grid_w = w'/16`; patches are ordered by 2x2 merge block
   (block row, block col, row in block, col in block), and each patch
   vector is `[channel][temporal 2][y 16][x 16]` = 1536 values, the image
   repeated in both temporal slots. `image_grid_thw = (1, grid_h, grid_w)`,
   `N = grid_h * grid_w / 4` merged tokens.

A ~1280 x 1440 page keeps its size (1280 x 1440 = 1843200 exactly):
grid 90 x 80 = 7200 patches, 1800 image tokens.

## Vision tower (24 blocks, width 1024, 16 heads of 64, ffn 4096)

Per image, `n = grid_h * grid_w` rows in merge-block order:

1. **Patch embed**: Conv3d(3 -> 1024, kernel = stride = (2, 16, 16), bias)
   = one matrix product per patch, K = 1536. Both temporal slots hold the
   same frame, so `W0 + W1` over K = 768 is the same product (llama.cpp
   does two conv2d and adds).
2. **Learned position embedding**: a 48 x 48 table (`pos_embed`, 2304 x
   1024) bilinearly resampled to the patch grid with `align_corners=True`:
   for row r, `src = r * 47 / max(grid_h - 1, 1)`, taps `floor(src)` and
   `floor(src) + 1` clamped to [0, 47], weights `1 - |src - tap|`; the
   same for columns; 4 taps per patch; added to the patch embedding.
3. **2D RoPE** (theta 10000, over the whole head): `inv_freq[j] =
   10000^(-2j/32)`, j < 16. For a patch at (row, col), pair i (< 32) of the
   rotate-half layout (`x[i]`, `x[i + 32]`) turns by `row * inv_freq[i]`
   for i < 16 and `col * inv_freq[i - 16]` for i >= 16. Rows and columns
   are patch coordinates (not merged).
4. **Blocks**, pre-norm: `h += proj(Attn(LN1(h)))`, `h += fc2(gelu_tanh(
   fc1(LN2(h))))`. LayerNorm with weight and bias, eps 1e-6. `qkv` is one
   1024 -> 3072 linear with bias (q | k | v, heads contiguous), `proj`
   1024 -> 1024 with bias; fc1/fc2 with biases; activation
   `gelu_pytorch_tanh`. **Attention is full and bidirectional** over all
   patches of one image (no window attention, unlike Qwen2.5-VL;
   `cu_seqlens` only separates images/frames), no q/k norm, scale 1/8.
5. **DeepStack**: the outputs of blocks 5, 11 and 17 (0-based) each go
   through their own merger (below, with the norm after the 2x2 shuffle).
6. **Merger** on the last block's output: LayerNorm(1024) per patch, view
   4 consecutive patches (one merge block) as 4096, fc1 4096 -> 4096 + bias,
   GELU (exact erf, `nn.GELU()`), fc2 4096 -> 2048 + bias. DeepStack
   mergers: view as 4096 first, then LayerNorm(4096), same fc1/GELU/fc2.

Result: `N x 2048` image embeddings plus three `N x 2048` DeepStack
features.

## Into the text model

- The `<|image_pad|>` rows of the input embeddings are replaced by the
  merger's output, in order.
- **DeepStack**: after decoder layer `k` (k = 0, 1, 2, i.e. on the residual
  stream that enters layer k + 1), DeepStack feature `k` is added to the
  image token rows. Nothing else changes in the layers.
- **MRoPE positions** (`get_rope_index`): text tokens advance one position
  each, all three axes equal. An image starting at position `s` (the token
  after `<|vision_start|>`) gives its token at merged grid (r, c)
  `(t, h, w) = (s, s + r, s + c)`; the next text token is at
  `s + max(grid_h, grid_w) / 2`. Only `<|image_pad|>` tokens are image
  type; `<|vision_start|>` and `<|vision_end|>` are text.
- **Interleaved MRoPE** (sections [24, 20, 20], head 128, theta 5e6):
  frequency i (< 64) uses the h position when `i % 3 == 1 and i < 60`, w
  when `i % 3 == 2 and i < 60`, t otherwise; rotate-half pairs (i, i + 64).
  With t = h = w this is the plain RoPE the text path runs, so text-only
  inputs are unchanged.

## mmproj tensors (llama.cpp `qwen3vl_merger`)

`Qwen3-VL-Embedding-2B.mmproj-f16.gguf`; keys under `clip.vision.*`
(`block_count` 24, `embedding_length` 1024, `feed_forward_length` 4096,
`attention.head_count` 16, `patch_size` 16, `projection_dim` 2048,
`spatial_merge_size` 2, `attention.layer_norm_epsilon`, `image_mean`,
`image_std`, `is_deepstack_layers` (bool per block)), `clip.projector_type`
= `qwen3vl_merger`.

| GGUF | HF | ggml shape (row first) |
|---|---|---|
| `v.patch_embd.weight`, `v.patch_embd.weight.1` | `patch_embed.proj.weight[:, :, t]` | 16, 16, 3, 1024 |
| `v.patch_embd.bias` | `patch_embed.proj.bias` | 1024 |
| `v.position_embd.weight` | `pos_embed.weight` | 1024, 2304 |
| `v.blk.{i}.ln1.{weight,bias}` | `blocks.i.norm1` | 1024 |
| `v.blk.{i}.attn_qkv.{weight,bias}` | `blocks.i.attn.qkv` | 1024, 3072 |
| `v.blk.{i}.attn_out.{weight,bias}` | `blocks.i.attn.proj` | 1024, 1024 |
| `v.blk.{i}.ln2.{weight,bias}` | `blocks.i.norm2` | 1024 |
| `v.blk.{i}.ffn_up.{weight,bias}` | `blocks.i.mlp.linear_fc1` | 1024, 4096 |
| `v.blk.{i}.ffn_down.{weight,bias}` | `blocks.i.mlp.linear_fc2` | 4096, 1024 |
| `v.post_ln.{weight,bias}` | `merger.norm` | 1024 |
| `mm.0.{weight,bias}` | `merger.linear_fc1` | 4096, 4096 |
| `mm.2.{weight,bias}` | `merger.linear_fc2` | 4096, 2048 |
| `v.deepstack.{5,11,17}.norm.{weight,bias}` | `deepstack_merger_list.{0,1,2}.norm` | 4096 |
| `v.deepstack.{5,11,17}.fc1.{weight,bias}` | `....linear_fc1` | 4096, 4096 |
| `v.deepstack.{5,11,17}.fc2.{weight,bias}` | `....linear_fc2` | 4096, 2048 |

DeepStack tensors carry the vision block index, not the merger's. llama.cpp
uses its tanh GELU for the mergers where HF uses the exact one; we follow HF.

## On the NPU GEMM

The package's bfp16 GEMM (`embed_gemm.py`, IRON's GEMM) takes (K, N) with
K % 512 == 0 and N % 64 == 0, chunks of M = 256/512/1024 rows, no bias.

| Product | M (page) | K | N | Tiles? |
|---|---|---|---|---|
| patch embed (W0+W1) | 7200 | 768 | 1024 | not as is: pad K to 1024 with zeros (or run K = 1536 unsummed); 11-15 GFLOP |
| ViT qkv | 7200 | 1024 | 3072 | yes |
| ViT attn proj | 7200 | 1024 | 1024 | yes |
| ViT fc1 | 7200 | 1024 | 4096 | yes |
| ViT fc2 | 7200 | 4096 | 1024 | yes |
| merger / DeepStack fc1 (x4) | 1800 | 4096 | 4096 | yes |
| merger / DeepStack fc2 (x4) | 1800 | 4096 | 2048 | yes |

Six new (K, N) shapes in the xclbin: (1024, 3072), (1024, 1024),
(1024, 4096), (4096, 1024), (4096, 4096), (4096, 2048); a patch embed
padded to K = 1024 reuses (1024, 1024). Every linear here has a bias.

### Where each piece runs

Target: Strix Halo only (Zen 5 with AVX-512, the XDNA2 NPU; never the
iGPU). The NPU takes the work; a piece runs on the CPU only where it is
measurably and clearly faster there, and the measurement goes in
EXPERIMENTS.md.

| Piece | Where | Why |
|---|---|---|
| decode, smart_resize, Pillow resize, patchify | CPU | irregular byte work, once per image, ~10 M pixel ops |
| position-embedding resample | CPU, cached per grid size | a gather of 4 rows per patch; constant for a grid |
| 2D RoPE / MRoPE tables | CPU, cached per grid | cos/sin tables, constant for a grid |
| all linears (patch embed, qkv, proj, fc1, fc2, mergers) | NPU GEMM | 98% of the FLOPs outside attention |
| bias adds | NPU, GEMM epilogue | interim: CPU add while widening C, until the epilogue exists |
| LayerNorm, GELU (tanh, erf), residual adds | NPU, fused into the GEMM's input or output side | elementwise over 7200 x 4096 per layer; on the CPU they would cost more than the NPU GEMMs they sit between |
| ViT attention (full, head 64) | NPU | 5.1 TFLOP per page, more than all GEMMs |
| text attention (causal, MRoPE) | NPU, with the text path | shared with text inputs |
| pooling, L2 norm | CPU | one 2048-vector per input |

Elementwise ops start on the CPU (AVX-512) in stage 2 only to get
correctness first; each one moves to the NPU unless the measured CPU time
is clearly lower than the NPU's for the same work, including the bf16
round trip.

## Cost of a page (1280 x 1440: 7200 patches, 1800 + ~30 tokens)

| Part | FLOP | NPU at 11-18 TFLOPS |
|---|---|---|
| ViT GEMMs (24 x 25.2 MFLOP x 7200) | 4.35 T | 0.24-0.40 s |
| patch embed + 4 mergers | 0.37 T | 0.02-0.03 s |
| text GEMMs (2.82 GFLOP x 1830) | 5.16 T | 0.29-0.47 s |
| **GEMMs total** | **9.9 T** | **0.55-0.90 s** |
| ViT attention (4 n^2 d x 24, 19.9 G exps) | 5.10 T | CPU today: ~12 s |
| text attention (causal, 28 layers) | 0.38 T | CPU today: ~0.9 s |

CPU attention rate: BENCHMARKS.md (2026-10-30) has 8 x 2,048-token inputs
at 1,297 tok/s with 72% in CPU attention, i.e. 3.85 TFLOP in ~9.1 s,
~0.42 TFLOPS effective with the current per-key loop. A blocked AVX-512
kernel (EXPERIMENTS.md's next step) may gain a few times; still several
seconds per page against ~0.7 s of NPU GEMMs.

The ViT attention is O(n^2) and larger than all the GEMMs together at page
size: 0.10 T at 1024 patches, 1.6 T at 4096, 5.1 T at 7200, 26 T at 16384.
With the GEMMs on the NPU and attention on the CPU a page would take
roughly 14 s, ~90% of it CPU attention. NPU attention (or a smaller
`max_pixels`; quality cost unmeasured) is what makes images practical.
Memory: the ViT keeps 7200 x 4096 fp32 (118 MB) of activations at most.

## Plan

1. **CPU reference** (`vision.hpp/.cpp`, done): mmproj loading,
   smart_resize + Pillow bicubic + patchify on raw RGB8 (no image decoder),
   the ViT and mergers in fp32, MRoPE positions and the DeepStack hook in
   `Reference::EmbedImage`. The test oracle, not a runtime path. Checks:
   `qwen3vl_embed_2b_vision_test` (model-free) and
   `qwen3vl_embed_2b_vision_oracle_test` against HF fp32 (QUALITY.md).
2. **NPU GEMMs**: the ViT and merger shapes in `embed_gemm_variants.py`
   (same xclbin, one hardware context), an `NpuVision` mirroring
   `npu_embedder.cpp`, MRoPE + DeepStack in the NPU embedder; elementwise
   ops on the CPU for now. Checks: op tests per new shape (relative error
   < 1e-2), cosine to the stage-1 reference per case, per-phase timings.
3. **NPU attention**: a bidirectional flash-attention kernel (head 64,
   online softmax on the core) for the ViT, then the text's causal
   attention with MRoPE-rotated q/k. This is the stage that decides the
   page time (CPU attention would be ~90% of it).
4. **Fusion**: bias + GELU epilogues and LayerNorm on the NPU, in
   runlists, so a ViT layer is a handful of commands; each move is kept
   only when the timing says so.
5. API: `/v1/embeddings` and `gufo-npu embed` take an image (decoded by
   the caller or a small decoder behind the CLI; the library takes RGB8).

## Open questions and risks

- **Resize parity**: Pillow's fixed-point bicubic is reproduced, but only
  unchanged-size images are checked against HF so far. JPEG decoding
  differences (libjpeg versions, EXIF rotation) are outside the library.
- **bfp16 for the ViT**: 8-bit mantissas with shared block exponents on
  LayerNorm outputs and GELU inputs; untested. The tower's 24 layers and
  the DeepStack injection may amplify error more than the text path does.
- **NPU attention**: no kernel in the tree; QK^T has K = 64, which the GEMM
  cannot tile (K % 512). Needs a dedicated design or a reshaped batched
  GEMM.
- **max_pixels**: the embedder's 1800-token cap is what quality numbers
  assume; a smaller cap trades quality for O(n^2) time, unmeasured.
- **Batching**: images of different sizes cannot share a ViT attention
  call; the GEMMs can (rows are independent).
- **Video** (`<|video_pad|>`, temporal pairs, timestamps) is out of scope.
- The merger GELU differs between HF (erf) and llama.cpp (tanh); we follow
  HF, so llama.cpp-made embeddings will differ slightly.
