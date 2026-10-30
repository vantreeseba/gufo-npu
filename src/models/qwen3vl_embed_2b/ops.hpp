#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "models/qwen3vl_embed_2b/config.hpp"

// The CPU math both the reference (reference.hpp) and the NPU embedder
// (npu_embedder.hpp) run, so the two differ only in their matrix products.
//
// A batch packs its inputs back to back, one row per token: segment i holds
// rows [begin, begin + length), its tokens at positions 0..length-1. Rows of
// different segments never attend to each other, so a segment's embedding
// does not depend on what it was batched with.
namespace gufo_npu::qwen3vl_embed_2b {

struct Segment {
  int begin = 0;
  int length = 0;
};

// Runs fn(i) for i in [0, n), in parallel where it can, and returns when all
// have finished (BurstPool::Run's shape).
using ParallelRun = std::function<void(
    std::size_t n, const std::function<void(std::size_t)>& fn)>;

// The segments of `inputs` packed back to back from row 0. Throws
// std::invalid_argument for an empty input or an id outside [0, vocab).
std::vector<Segment> Pack(const std::vector<std::vector<int>>& inputs,
                          int vocab);

// RoPE cos/sin per position and frequency, as HF computes them (fp32
// inv_freq = 1 / theta^(2i / head_dim), rotate-half pairs i and i + half).
class RopeTable {
public:
  RopeTable(int head_dim, float theta);
  // Makes positions [0, count) available. Not thread-safe; call before
  // Apply runs in parallel.
  void Reserve(int count);
  // Rotates x (head_dim values) in place to `position` (< the reserved
  // count).
  void Apply(float* x, int position) const;

private:
  int half_ = 0;
  std::vector<float> inv_freq_;
  std::vector<float> cos_, sin_;  // [position][half]
  int count_ = 0;
};

// The attention mixer of one layer over a packed batch. `in` holds each
// row's input projection (row stride cfg.InRows(): q | k | v, fp32); its q
// and k are RMS-normalized per head (q_norm, k_norm), rotated and, for q,
// scaled by 1/sqrt(head_dim), in place. `out` (row stride cfg.QDim())
// receives each query head's softmax-weighted values, causal within the
// row's segment. `rope` must have every segment's length reserved.
void Attention(const Config& cfg, const RopeTable& rope, const float* q_norm,
               const float* k_norm, const std::vector<Segment>& segments,
               float* in, float* out, const ParallelRun& run);

namespace detail {
// Whether the CPU runs the AVX-512 attention kernel, which Attention picks
// when it does (otherwise a plain loop).
bool HasAvx512();
// Attention with the kernel chosen: the AVX-512 one (throws
// std::invalid_argument without HasAvx512()) or the plain loop. For tests.
void Attention(const Config& cfg, const RopeTable& rope, const float* q_norm,
               const float* k_norm, const std::vector<Segment>& segments,
               float* in, float* out, const ParallelRun& run, bool avx512);
}  // namespace detail

// The text model's interleaved MRoPE sections (t, h, w), in frequencies.
inline constexpr int kMropeSections[3] = {24, 20, 20};

// Interleaved MRoPE, as HF's Qwen3VLTextRotaryEmbedding: frequency i of the
// head's half turns by the h position when i % 3 == 1 and i < 3 *
// sections[1], by w when i % 3 == 2 and i < 3 * sections[2], by t
// otherwise; rotate-half pairs (i, i + half). Equal t, h and w give
// RopeTable's rotation.
class MropeTable {
public:
  MropeTable(int head_dim, float theta);
  int half() const { return half_; }
  // cos and sin (half() values each) for the position thw = {t, h, w}.
  void Angles(const int* thw, float* cos, float* sin) const;
  // Rotates x (2 * half() values) in place by Angles' output.
  void Apply(float* x, const float* cos, const float* sin) const;

private:
  int half_ = 0;
  std::vector<float> inv_freq_;
  std::vector<int> axis_;  // per frequency: 0 t, 1 h, 2 w
};

// Attention with MRoPE: as Attention (same kernels), but each packed row r
// is rotated to positions[3 r .. 3 r + 2] = (t, h, w) (an input with
// images, MropePositions in vision.hpp) instead of its index in the
// segment.
void AttentionMrope(const Config& cfg, const MropeTable& rope,
                    const int* positions, const float* q_norm,
                    const float* k_norm, const std::vector<Segment>& segments,
                    float* in, float* out, const ParallelRun& run);

// The embedding of each segment: its last row of `h` (row stride
// cfg.hidden) through the final RMSNorm, L2-normalized.
std::vector<std::vector<float>> Pool(const Config& cfg,
                                     const float* output_norm,
                                     const std::vector<Segment>& segments,
                                     const float* h);

// Keeps the first `dimensions` values of a unit embedding and scales them
// back to unit length (the model is trained for these Matryoshka
// truncations). 0, or at least the full size, keeps it as is.
void Truncate(std::vector<float>& embedding, int dimensions);

}  // namespace gufo_npu::qwen3vl_embed_2b
