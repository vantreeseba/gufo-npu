#include "core/gguf.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Builds a small GGUF v3 image in memory and reads it back.
namespace {

class Writer {
public:
  template<typename T>
  void Put(T value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  void PutString(const std::string& s) {
    Put<std::uint64_t>(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
  void Align(std::size_t alignment) {
    while (bytes.size() % alignment != 0) {
      bytes.push_back(0);
    }
  }
  std::vector<std::uint8_t> bytes;
};

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  Writer w;
  w.Put<std::uint32_t>(0x46554747);  // "GGUF"
  w.Put<std::uint32_t>(3);
  w.Put<std::uint64_t>(2);  // tensors
  w.Put<std::uint64_t>(3);  // metadata
  w.PutString("general.architecture");
  w.Put<std::uint32_t>(8);
  w.PutString("qwen35");
  w.PutString("qwen35.block_count");
  w.Put<std::uint32_t>(4);
  w.Put<std::uint32_t>(25);
  w.PutString("tokenizer.ggml.tokens");
  w.Put<std::uint32_t>(9);
  w.Put<std::uint32_t>(8);
  w.Put<std::uint64_t>(2);
  w.PutString("a");
  w.PutString("bc");
  // tensor 0: F32 [3, 2]
  w.PutString("w");
  w.Put<std::uint32_t>(2);
  w.Put<std::uint64_t>(3);
  w.Put<std::uint64_t>(2);
  w.Put<std::uint32_t>(0);
  w.Put<std::uint64_t>(0);
  // tensor 1: F32 [4], after 24 bytes rounded up to 32
  w.PutString("norm");
  w.Put<std::uint32_t>(1);
  w.Put<std::uint64_t>(4);
  w.Put<std::uint32_t>(0);
  w.Put<std::uint64_t>(32);
  w.Align(32);
  for (int i = 0; i < 6; ++i) {
    w.Put<float>(static_cast<float>(i));
  }
  w.Align(32);
  for (int i = 0; i < 4; ++i) {
    w.Put<float>(10.0f + static_cast<float>(i));
  }

  const auto file =
      gufo_npu::gguf::File::FromBytes(w.bytes.data(), w.bytes.size());
  Expect(file->GetString("general.architecture") == "qwen35", "string");
  Expect(file->GetUInt("qwen35.block_count") == 25, "uint");
  Expect(file->GetStrings("tokenizer.ggml.tokens") ==
             std::vector<std::string>({"a", "bc"}),
         "string array");
  const auto& weight = file->Tensor("w");
  Expect(weight.RowElements() == 3 && weight.Rows() == 2, "shape");
  Expect(weight.size_bytes == 24, "size");
  float value = 0;
  std::memcpy(&value, weight.data + 5 * sizeof(float), sizeof(float));
  Expect(value == 5.0f, "tensor data");
  std::memcpy(&value, file->Tensor("norm").data, sizeof(float));
  Expect(value == 10.0f, "aligned offset");
  Expect(file->FindTensor("missing") == nullptr, "missing tensor");

  bool threw = false;
  try {
    (void)file->GetUInt("missing.key");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "missing key throws");

  auto truncated = w.bytes;
  truncated.resize(40);
  threw = false;
  try {
    (void)gufo_npu::gguf::File::FromBytes(truncated.data(), truncated.size());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "truncated file throws");

  if (failures == 0) {
    std::puts("PASS: gguf_test");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
