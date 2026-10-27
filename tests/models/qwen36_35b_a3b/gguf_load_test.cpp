#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/parallel.hpp"
#include "core/quant.hpp"
#include "models/qwen35/config.hpp"

// Opens the Qwen3.6-35B-A3B GGUF (arch qwen35moe), checks ReadConfig
// against the model's published shape (docs/models/qwen3.6-35b-a3b/README.md)
// and dequantizes every tensor row by row, checking that all values are
// finite. Needs GUFO_NPU_QWEN36_MODEL; exits 77 (skip) without.
namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_NPU_QWEN36_MODEL");
  if (path == nullptr || *path == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN36_MODEL to the Qwen3.6-35B-A3B GGUF");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(path);
  const auto c = gufo_npu::qwen35::ReadConfig(*file);
  Expect(c.architecture == "qwen35moe", "architecture");
  Expect(c.layers == 40 && c.mtp_layers == 1, "40 layers + 1 MTP layer");
  Expect(c.hidden == 2048 && c.vocab == 248320, "hidden, vocab");
  Expect(!c.tied_embeddings, "untied output.weight");
  Expect(c.full_attention_interval == 4, "full attention every 4th layer");
  Expect(c.heads == 16 && c.kv_heads == 2 && c.head_dim == 256 &&
             c.rope_dim == 64 && c.rope_theta == 1e7f,
         "attention shape");
  Expect(c.linear_key_heads == 16 && c.linear_value_heads == 32 &&
             c.linear_head_dim == 128 && c.conv_kernel == 4,
         "DeltaNet shape");
  Expect(c.ffn == 0 && c.experts == 256 && c.experts_used == 8 &&
             c.expert_ffn == 512 && c.shared_expert_ffn == 512,
         "MoE shape");
  int full = 0;
  for (int il = 0; il < c.layers; ++il) {
    full += c.IsFullAttention(il) ? 1 : 0;
  }
  Expect(full == 10, "10 full-attention layers");

  std::atomic<std::uint64_t> values{0};
  std::atomic<bool> finite{true};
  for (const auto& tensor : file->Tensors()) {
    const auto cols = static_cast<std::size_t>(tensor.RowElements());
    const auto row_bytes = gufo_npu::quant::RowBytes(tensor.type, cols);
    gufo_npu::ParallelFor(
        tensor.Rows(), [&](std::size_t begin, std::size_t end) {
          std::vector<float> row(cols);
          for (std::size_t r = begin; r < end; ++r) {
            gufo_npu::quant::Dequantize(
                tensor.type, tensor.data + r * row_bytes, row.data(), cols);
            for (const float v : row) {
              if (!std::isfinite(v)) {
                finite = false;
              }
            }
          }
          values += (end - begin) * cols;
        });
    Expect(finite, "finite values in " + tensor.name);
    finite = true;
  }
  if (failures == 0) {
    std::printf("PASS: %zu tensors, %llu values dequantized\n",
                file->Tensors().size(),
                static_cast<unsigned long long>(values.load()));
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
