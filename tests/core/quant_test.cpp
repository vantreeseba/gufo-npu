#include "core/quant.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

// Dequantizes real GGUF blocks and compares bit for bit with gguf-py
// (fixtures from tools/gguf/make_quant_fixtures.py).
namespace {

using gufo_npu::gguf::GgmlType;

std::vector<std::uint8_t> ReadAll(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

int CheckFixture(const std::filesystem::path& dir, const char* name,
                 GgmlType type) {
  const auto raw = ReadAll(dir / (std::string(name) + ".bin"));
  const auto expected_bytes = ReadAll(dir / (std::string(name) + ".f32"));
  if (raw.empty() || expected_bytes.empty()) {
    std::fprintf(stderr, "FAIL: %s fixture missing in %s\n", name, dir.c_str());
    return 1;
  }
  const std::size_t n = expected_bytes.size() / sizeof(float);
  if (gufo_npu::quant::RowBytes(type, n) != raw.size()) {
    std::fprintf(stderr, "FAIL: %s: %zu bytes for %zu values\n", name,
                 raw.size(), n);
    return 1;
  }
  std::vector<float> expected(n);
  std::memcpy(expected.data(), expected_bytes.data(), expected_bytes.size());
  std::vector<float> actual(n);
  gufo_npu::quant::Dequantize(type, raw.data(), actual.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
      std::fprintf(stderr, "FAIL: %s[%zu]: %.9g != %.9g\n", name, i, actual[i],
                   expected[i]);
      return 1;
    }
  }
  std::printf("ok: %s (%zu values)\n", name, n);
  return 0;
}

int CheckHalf() {
  struct Case {
    std::uint16_t bits;
    float value;
  };
  const Case cases[] = {{0x0000, 0.0f},
                        {0x3C00, 1.0f},
                        {0xC000, -2.0f},
                        {0x3555, 0.333251953125f},
                        {0x0001, 5.96046448e-08f},  // smallest subnormal
                        {0x7BFF, 65504.0f}};
  for (const auto& c : cases) {
    if (gufo_npu::quant::HalfToFloat(c.bits) != c.value) {
      std::fprintf(stderr, "FAIL: half 0x%04x\n", c.bits);
      return 1;
    }
  }
  std::puts("ok: F16 conversions");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: quant_test <fixture-dir>\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  int failures = CheckHalf();
  failures += CheckFixture(dir, "Q8_0", GgmlType::kQ8_0);
  failures += CheckFixture(dir, "Q4_K", GgmlType::kQ4_K);
  failures += CheckFixture(dir, "Q5_K", GgmlType::kQ5_K);
  failures += CheckFixture(dir, "Q6_K", GgmlType::kQ6_K);
  failures += CheckFixture(dir, "BF16", GgmlType::kBF16);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
