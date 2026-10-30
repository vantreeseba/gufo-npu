#pragma once

#include <string>

#include "core/gguf.hpp"

// Qwen3-VL-Embedding-2B's text model, as llama.cpp's converter writes it
// (GGUF architecture "qwen3vl", the vision tower in a separate mmproj file):
// a dense Qwen3 decoder with grouped-query attention, per-head q/k RMSNorm,
// SwiGLU and tied embeddings. For text, the model's interleaved MRoPE gives
// every token the same temporal, height and width position, which is plain
// RoPE over the whole head.
namespace gufo_npu::qwen3vl_embed_2b {

inline constexpr char kArchitecture[] = "qwen3vl";
// The smallest embedding size the model is trained to be cut to
// (Matryoshka: the first n values of a vector, renormalized, for n from this
// up to the hidden size).
inline constexpr int kMinDimensions = 64;

struct Config {
  int layers = 0;
  int hidden = 0;
  int ffn = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  float rope_theta = 0;
  float rms_eps = 0;
  int vocab = 0;
  int context_length = 0;  // trained

  int QDim() const { return heads * head_dim; }
  int KvDim() const { return kv_heads * head_dim; }
  // The attention input projection's outputs per token: q | k | v.
  int InRows() const { return QDim() + 2 * KvDim(); }
};

// Whether the GGUF holds a model this package reads (its architecture).
bool IsEmbeddingModel(const gguf::File& file);

// Reads and checks the shape metadata. Throws std::runtime_error for
// another architecture or inconsistent shapes. The output head (tied or
// not) is never read: an embedding is the last hidden state.
Config ReadConfig(const gguf::File& file);

}  // namespace gufo_npu::qwen3vl_embed_2b
