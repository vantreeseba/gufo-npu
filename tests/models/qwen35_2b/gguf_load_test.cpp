#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/gguf.hpp"
#include "core/quant.hpp"

// Opens a real Qwen3.5 GGUF and dequantizes every tensor, checking that all
// values are finite. Needs GUFO_NPU_QWEN35_2B_GGUF; exits 77 (skip) without.
int main() {
  const char* path = std::getenv("GUFO_NPU_QWEN35_2B_GGUF");
  if (path == nullptr || *path == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_GGUF to a Qwen3.5-2B GGUF");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(path);
  if (file->GetString("general.architecture") != "qwen35") {
    std::fprintf(stderr, "FAIL: architecture is not qwen35\n");
    return 1;
  }
  std::uint64_t values = 0;
  std::vector<float> row;
  for (const auto& tensor : file->Tensors()) {
    row.resize(tensor.RowElements());
    const auto row_bytes =
        gufo_npu::quant::RowBytes(tensor.type, tensor.RowElements());
    for (std::uint64_t r = 0; r < tensor.Rows(); ++r) {
      gufo_npu::quant::Dequantize(tensor.type, tensor.data + r * row_bytes,
                                  row.data(), row.size());
      for (const float v : row) {
        if (!std::isfinite(v)) {
          std::fprintf(stderr, "FAIL: non-finite value in %s row %llu\n",
                       tensor.name.c_str(), static_cast<unsigned long long>(r));
          return 1;
        }
      }
      values += row.size();
    }
  }
  std::printf("PASS: %zu tensors, %llu values dequantized\n",
              file->Tensors().size(), static_cast<unsigned long long>(values));
  return 0;
}
