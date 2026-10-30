#include "models/qwen3vl_embed_2b/config.hpp"

#include <cstdint>
#include <stdexcept>

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

int Int(const gguf::File& file, const std::string& key) {
  return static_cast<int>(file.GetUInt(key));
}

int IntOr(const gguf::File& file, const std::string& key, int fallback) {
  return file.Find(key) == nullptr ? fallback : Int(file, key);
}

}  // namespace

bool IsEmbeddingModel(const gguf::File& file) {
  return file.Find("general.architecture") != nullptr &&
         file.GetString("general.architecture") == kArchitecture;
}

Config ReadConfig(const gguf::File& file) {
  if (!IsEmbeddingModel(file)) {
    throw std::runtime_error("not a Qwen3-VL GGUF: " +
                             (file.Find("general.architecture") != nullptr
                                  ? file.GetString("general.architecture")
                                  : std::string("no general.architecture")));
  }
  const std::string a = std::string(kArchitecture) + ".";
  Config c;
  c.layers = Int(file, a + "block_count");
  c.hidden = Int(file, a + "embedding_length");
  c.ffn = Int(file, a + "feed_forward_length");
  c.heads = Int(file, a + "attention.head_count");
  c.kv_heads = Int(file, a + "attention.head_count_kv");
  c.head_dim = IntOr(file, a + "attention.key_length",
                     c.heads > 0 ? c.hidden / c.heads : 0);
  c.rope_theta = static_cast<float>(file.GetFloat(a + "rope.freq_base"));
  c.rms_eps =
      static_cast<float>(file.GetFloat(a + "attention.layer_norm_rms_epsilon"));
  c.context_length = IntOr(file, a + "context_length", 0);

  const auto& embed = file.Tensor("token_embd.weight");
  c.vocab = static_cast<int>(embed.shape[1]);
  if (embed.RowElements() != static_cast<std::uint64_t>(c.hidden)) {
    throw std::runtime_error("qwen3vl: token_embd width != embedding_length");
  }
  if (IntOr(file, a + "attention.value_length", c.head_dim) != c.head_dim) {
    throw std::runtime_error("qwen3vl: key and value head dims differ");
  }
  if (c.layers <= 0 || c.hidden <= 0 || c.ffn <= 0 || c.heads <= 0 ||
      c.kv_heads <= 0 || c.heads % c.kv_heads != 0 || c.head_dim <= 0 ||
      c.head_dim % 2 != 0 || c.rope_theta <= 0) {
    throw std::runtime_error("qwen3vl: inconsistent shape metadata");
  }
  return c;
}

}  // namespace gufo_npu::qwen3vl_embed_2b
