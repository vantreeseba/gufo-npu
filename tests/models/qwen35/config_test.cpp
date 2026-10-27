#include "models/qwen35/config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/gguf.hpp"

// ReadConfig on small in-memory GGUF headers: a dense `qwen35` model and a
// `qwen35moe` one shaped like Qwen3.6-35B-A3B (DeltaNet value heads twice the
// key heads, untied LM head, one MTP layer), and inconsistent metadata.
namespace {

namespace qwen35 = gufo_npu::qwen35;

// A metadata key: GGUF value type 4 (uint32), 6 (float32) or 8 (string).
struct Key {
  Key(std::string key, std::uint32_t value)
      : name(std::move(key)), type(4), u(value) {}
  Key(std::string key, float value) : name(std::move(key)), type(6), f(value) {}
  Key(std::string key, std::string value)
      : name(std::move(key)), type(8), s(std::move(value)) {}
  std::string name;
  std::uint32_t type;
  std::uint32_t u = 0;
  float f = 0;
  std::string s;
};

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> shape;  // F32
};

std::vector<std::uint8_t> Build(const std::vector<Key>& keys,
                                const std::vector<Tensor>& tensors) {
  std::vector<std::uint8_t> bytes;
  auto put = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    bytes.insert(bytes.end(), b, b + n);
  };
  auto put_u32 = [&](std::uint32_t v) { put(&v, sizeof(v)); };
  auto put_u64 = [&](std::uint64_t v) { put(&v, sizeof(v)); };
  auto put_string = [&](const std::string& s) {
    put_u64(s.size());
    put(s.data(), s.size());
  };
  put_u32(0x46554747);  // "GGUF"
  put_u32(3);
  put_u64(tensors.size());
  put_u64(keys.size());
  for (const auto& key : keys) {
    put_string(key.name);
    put_u32(key.type);
    if (key.type == 4) {
      put_u32(key.u);
    } else if (key.type == 6) {
      put(&key.f, sizeof(key.f));
    } else {
      put_string(key.s);
    }
  }
  std::uint64_t offset = 0;
  for (const auto& tensor : tensors) {
    put_string(tensor.name);
    put_u32(static_cast<std::uint32_t>(tensor.shape.size()));
    std::uint64_t elements = 1;
    for (const auto dim : tensor.shape) {
      put_u64(dim);
      elements *= dim;
    }
    put_u32(0);  // F32
    put_u64(offset);
    offset += (elements * sizeof(float) + 31) / 32 * 32;
  }
  bytes.resize((bytes.size() + 31) / 32 * 32 + offset, 0);
  return bytes;
}

std::vector<Key> CommonKeys(const std::string& arch) {
  const std::string a = arch + ".";
  return {
      {"general.architecture", arch},
      {a + "block_count", 9u},
      {a + "embedding_length", 64u},
      {a + "attention.layer_norm_rms_epsilon", 1e-6f},
      {a + "full_attention_interval", 4u},
      {a + "attention.head_count", 4u},
      {a + "attention.head_count_kv", 2u},
      {a + "attention.key_length", 32u},
      {a + "attention.value_length", 32u},
      {a + "rope.dimension_count", 8u},
      {a + "rope.freq_base", 1e7f},
      {a + "ssm.conv_kernel", 4u},
      {a + "ssm.state_size", 16u},
      {a + "ssm.group_count", 2u},
      {a + "nextn_predict_layers", 1u},
  };
}

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

bool Throws(const std::vector<Key>& keys, const std::vector<Tensor>& tensors) {
  const auto bytes = Build(keys, tensors);
  const auto file = gufo_npu::gguf::File::FromBytes(bytes.data(), bytes.size());
  try {
    (void)qwen35::ReadConfig(*file);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const std::vector<Tensor> tied = {{"token_embd.weight", {64, 100}}};
  const std::vector<Tensor> untied = {{"token_embd.weight", {64, 100}},
                                      {"output.weight", {64, 100}}};

  // Dense: equal DeltaNet key and value heads, tied embeddings.
  auto dense_keys = CommonKeys("qwen35");
  dense_keys.push_back({"qwen35.feed_forward_length", 192u});
  dense_keys.push_back({"qwen35.ssm.time_step_rank", 2u});
  dense_keys.push_back({"qwen35.ssm.inner_size", 32u});
  {
    const auto bytes = Build(dense_keys, tied);
    const auto file =
        gufo_npu::gguf::File::FromBytes(bytes.data(), bytes.size());
    const auto c = qwen35::ReadConfig(*file);
    Expect(c.layers == 8 && c.mtp_layers == 1, "dense: layers");
    Expect(!c.IsMoe() && c.ffn == 192 && c.FfnWidth() == 192, "dense: ffn");
    Expect(c.vocab == 100 && c.tied_embeddings, "dense: tied vocab");
    Expect(c.linear_key_heads == 2 && c.linear_value_heads == 2,
           "dense: DeltaNet heads");
  }

  // MoE: 32 value heads over 16 key heads in the real model, here 4 over 2.
  auto moe_keys = CommonKeys("qwen35moe");
  moe_keys.push_back({"qwen35moe.ssm.time_step_rank", 4u});
  moe_keys.push_back({"qwen35moe.ssm.inner_size", 64u});
  moe_keys.push_back({"qwen35moe.expert_count", 16u});
  moe_keys.push_back({"qwen35moe.expert_used_count", 4u});
  moe_keys.push_back({"qwen35moe.expert_feed_forward_length", 24u});
  moe_keys.push_back({"qwen35moe.expert_shared_feed_forward_length", 40u});
  {
    const auto bytes = Build(moe_keys, untied);
    const auto file =
        gufo_npu::gguf::File::FromBytes(bytes.data(), bytes.size());
    const auto c = qwen35::ReadConfig(*file);
    Expect(c.architecture == "qwen35moe", "moe: architecture");
    Expect(c.layers == 8 && c.mtp_layers == 1, "moe: layers");
    Expect(c.IsMoe() && c.experts == 16 && c.experts_used == 4, "moe: experts");
    Expect(c.ffn == 0 && c.expert_ffn == 24 && c.shared_expert_ffn == 40 &&
               c.FfnWidth() == 40,
           "moe: FFN widths");
    Expect(c.linear_key_heads == 2 && c.linear_value_heads == 4 &&
               c.LinearKeyDim() == 32 && c.LinearValueDim() == 64 &&
               c.ConvChannels() == 128,
           "moe: DeltaNet value heads != key heads");
    Expect(!c.tied_embeddings && c.vocab == 100, "moe: untied LM head");
    Expect(!c.IsFullAttention(2) && c.IsFullAttention(3) &&
               c.IsFullAttention(7) && c.IsFullAttention(8),
           "moe: layer pattern and MTP layer");
  }

  // Inconsistent: experts used > experts, MoE without expert widths, a
  // qwen35moe header without experts, and inner_size != value heads x dim.
  auto bad = moe_keys;
  bad[bad.size() - 3].u = 32;  // expert_used_count
  Expect(Throws(bad, untied), "throws: experts used > experts");
  bad = moe_keys;
  bad.pop_back();  // shared expert width
  Expect(Throws(bad, untied), "throws: no shared expert width");
  bad = moe_keys;
  bad.erase(bad.end() - 4);  // expert_count
  Expect(Throws(bad, untied), "throws: qwen35moe without experts");
  bad = moe_keys;
  bad[bad.size() - 5].u = 32;  // ssm.inner_size
  Expect(Throws(bad, untied), "throws: inner size");

  if (failures == 0) {
    std::puts("PASS: qwen35_config_test");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
