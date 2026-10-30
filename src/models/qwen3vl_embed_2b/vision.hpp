#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/gguf.hpp"

// Qwen3-VL-Embedding-2B's image input (docs/models/qwen3-vl-embedding-2b/
// VISION.md): the reference embedder's preprocessing on a decoded RGB8
// image, the vision tower from llama.cpp's mmproj GGUF (projector
// "qwen3vl_merger") in fp32 on the CPU, and the text model's 3D (MRoPE)
// positions for an input holding images. The text model side is
// Reference::EmbedImage (reference.hpp).
namespace gufo_npu::qwen3vl_embed_2b {

inline constexpr char kProjectorType[] = "qwen3vl_merger";
// The reference embedder's resize bounds (scripts/qwen3_vl_embedding.py):
// sides a multiple of 32 (patch 16 x merge 2), 4 to 1800 image tokens.
inline constexpr int kImageFactor = 32;
inline constexpr int kImageMinPixels = 4 * 32 * 32;
inline constexpr int kImageMaxPixels = 1800 * 32 * 32;

struct VisionConfig {
  int layers = 0;
  int hidden = 0;
  int ffn = 0;
  int heads = 0;
  int patch = 0;
  int merge = 0;
  int out_hidden = 0;  // the text model's width
  int pos_side = 0;    // the learned position grid is pos_side^2
  float eps = 1e-6f;
  float rope_theta = 10000.0f;
  // The blocks whose outputs feed the DeepStack mergers, in order (the
  // k-th is added after text layer k).
  std::vector<int> deepstack;

  int HeadDim() const { return hidden / heads; }
  int MergedDim() const { return hidden * merge * merge; }
  int PatchValues() const { return 3 * patch * patch; }
};

// Whether `file` is a Qwen3-VL mmproj (clip.projector_type).
bool IsVisionProjector(const gguf::File& file);
// Throws std::runtime_error for another projector or inconsistent shapes.
VisionConfig ReadVisionConfig(const gguf::File& file);

struct ImageSize {
  int height = 0;
  int width = 0;
};

// qwen_vl_utils.smart_resize: the size the image is resized to, both sides
// multiples of `factor`, the area within [min_pixels, max_pixels] where it
// can be. Throws std::invalid_argument for a side < 1 or an aspect ratio
// above 200.
ImageSize SmartResize(int height, int width, int factor = kImageFactor,
                      int min_pixels = kImageMinPixels,
                      int max_pixels = kImageMaxPixels);

// Pillow's Image.resize with its default for RGB (antialiased bicubic,
// a = -0.5, horizontal pass then vertical, 8-bit fixed point): `rgb` is
// width x height RGB8, row-major and unpadded, as is the result. The same
// size returns a copy.
std::vector<std::uint8_t> ResizeBicubic(const std::uint8_t* rgb, int width,
                                        int height, int out_width,
                                        int out_height);

// An image cut into patches, as the processor's pixel_values with the two
// (equal) temporal slots folded: grid_h * grid_w rows of 3 * patch * patch
// values ([channel][y][x], x / 255 normalized with mean = std = 0.5), the
// rows in merge-block order (block row, block col, row in block, col in
// block).
struct Patches {
  int grid_h = 0;
  int grid_w = 0;
  std::vector<float> values;

  int Count() const { return grid_h * grid_w; }
};

// Patchifies an image whose sides are multiples of patch * merge. Throws
// std::invalid_argument otherwise.
Patches Patchify(const std::uint8_t* rgb, int width, int height, int patch,
                 int merge);

// SmartResize, ResizeBicubic and Patchify: what the reference embedder
// feeds the vision tower for a decoded RGB8 image.
Patches PreprocessImage(const std::uint8_t* rgb, int width, int height,
                        int patch, int merge);

// The patch coordinates (row, col) of row `index` of Patches (merge-block
// order) for a grid grid_w patches wide.
void PatchPosition(int index, int grid_w, int merge, int* row, int* col);

// The bilinear (align_corners) taps into a `side`-long learned table for
// coordinate `index` of `size`, as HF resamples pos_embed: taps[0..1],
// weights[0..1].
void InterpolationTaps(int index, int size, int side, int* taps,
                       float* weights);

// One image through the vision tower: `tokens` = Count() / merge^2 rows of
// out_hidden values for the <|image_pad|> tokens, and one such block per
// DeepStack layer. `rows` x `cols` is the merged token grid.
struct ImageFeatures {
  int rows = 0;
  int cols = 0;
  std::vector<float> embeds;
  std::vector<std::vector<float>> deepstack;

  int Tokens() const { return rows * cols; }
};

// The text model's (t, h, w) position of each id, 3 per id, as HF's
// get_rope_index: text advances one position per token on all three axes;
// the k-th run of `image_pad_id` (images[k].Tokens() long) starting at
// position s puts merged grid cell (r, c) at (s, s + r, s + c), and the
// text after it continues at s + max(rows, cols). Throws
// std::invalid_argument when the runs do not match `images`.
std::vector<int> MropePositions(const std::vector<int>& ids, int image_pad_id,
                                const std::vector<ImageFeatures>& images);

// The vision tower in fp32: norms, biases and embeddings dequantized at
// load, matrix rows as they are used. Slow; the oracle for an NPU path.
class VisionReference {
public:
  // Keeps pointers into `file`, which must outlive it.
  explicit VisionReference(const gguf::File& file);
  ~VisionReference();
  VisionReference(const VisionReference&) = delete;
  VisionReference& operator=(const VisionReference&) = delete;

  const VisionConfig& config() const;
  // Throws std::invalid_argument for an empty grid or one whose sides are
  // not multiples of the merge size.
  ImageFeatures Encode(const Patches& patches);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
