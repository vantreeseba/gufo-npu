#include "models/qwen35/config.hpp"

#include <stdexcept>

namespace gufo_npu::qwen35 {
namespace {

int Int(const gguf::File& file, const std::string& key) {
  return static_cast<int>(file.GetUInt(key));
}

int IntOr(const gguf::File& file, const std::string& key, int fallback) {
  return file.Find(key) == nullptr ? fallback : Int(file, key);
}

}  // namespace

Config ReadConfig(const gguf::File& file) {
  Config c;
  c.architecture = file.GetString("general.architecture");
  if (c.architecture != "qwen35" && c.architecture != "qwen35moe") {
    throw std::runtime_error("not a Qwen3.5 GGUF: " + c.architecture);
  }
  const std::string a = c.architecture + ".";
  c.mtp_layers = IntOr(file, a + "nextn_predict_layers", 0);
  c.layers = Int(file, a + "block_count") - c.mtp_layers;
  c.hidden = Int(file, a + "embedding_length");
  c.ffn = IntOr(file, a + "feed_forward_length", 0);
  c.rms_eps =
      static_cast<float>(file.GetFloat(a + "attention.layer_norm_rms_epsilon"));
  c.full_attention_interval = Int(file, a + "full_attention_interval");

  c.heads = Int(file, a + "attention.head_count");
  c.kv_heads = Int(file, a + "attention.head_count_kv");
  c.head_dim = Int(file, a + "attention.key_length");
  c.rope_dim = Int(file, a + "rope.dimension_count");
  c.rope_theta = static_cast<float>(file.GetFloat(a + "rope.freq_base"));

  c.conv_kernel = Int(file, a + "ssm.conv_kernel");
  c.linear_head_dim = Int(file, a + "ssm.state_size");
  c.linear_key_heads = Int(file, a + "ssm.group_count");
  c.linear_value_heads = Int(file, a + "ssm.time_step_rank");

  c.experts = IntOr(file, a + "expert_count", 0);
  c.experts_used = IntOr(file, a + "expert_used_count", 0);
  c.expert_ffn = IntOr(file, a + "expert_feed_forward_length", 0);
  c.shared_expert_ffn = IntOr(file, a + "expert_shared_feed_forward_length", 0);

  const auto& embed = file.Tensor("token_embd.weight");
  c.vocab = static_cast<int>(embed.shape[1]);
  c.tied_embeddings = file.FindTensor("output.weight") == nullptr;

  if (Int(file, a + "attention.value_length") != c.head_dim) {
    throw std::runtime_error("qwen35: key and value head dims differ");
  }
  if (Int(file, a + "ssm.inner_size") != c.LinearValueDim()) {
    throw std::runtime_error("qwen35: ssm.inner_size != value heads x dim");
  }
  if (c.layers <= 0 || c.full_attention_interval <= 0 ||
      c.heads % c.kv_heads != 0 ||
      c.linear_value_heads % c.linear_key_heads != 0) {
    throw std::runtime_error("qwen35: inconsistent shape metadata");
  }
  return c;
}

}  // namespace gufo_npu::qwen35
