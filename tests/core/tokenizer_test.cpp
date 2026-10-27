#include "core/tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "core/gguf.hpp"

// Encodes the HF fixtures (tools/reference/tokenizer_fixtures.py) and checks
// ids and decoded text match exactly. Needs GUFO_NPU_QWEN35_2B_GGUF for the
// vocabulary; exits 77 (skip) without it.
int main(int argc, char** argv) {
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_GGUF");
  if (model == nullptr || *model == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_GGUF to a Qwen3.5 GGUF");
    return 77;
  }
  if (argc != 2) {
    std::fprintf(stderr, "usage: tokenizer_test <fixtures.jsonl>\n");
    return 2;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  const auto load_start = std::chrono::steady_clock::now();
  const gufo_npu::Tokenizer tokenizer(*file);
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_start)
                             .count();

  std::ifstream in(argv[1]);
  int cases = 0;
  int failures = 0;
  double encode_ms = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) {
      continue;
    }
    const auto fixture = nlohmann::json::parse(line);
    const auto text = fixture["text"].get<std::string>();
    const auto want = fixture["ids"].get<std::vector<int>>();
    const auto start = std::chrono::steady_clock::now();
    const auto got = tokenizer.Encode(text);
    encode_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count();
    ++cases;
    if (got != want) {
      ++failures;
      std::fprintf(stderr, "FAIL: encode %s\n  want %s\n  got  %s\n",
                   nlohmann::json(text).dump().c_str(),
                   nlohmann::json(want).dump().c_str(),
                   nlohmann::json(got).dump().c_str());
      continue;
    }
    if (tokenizer.Decode(got) != fixture["decoded"].get<std::string>()) {
      ++failures;
      std::fprintf(stderr, "FAIL: decode %s\n",
                   nlohmann::json(text).dump().c_str());
    }
  }
  if (cases == 0) {
    std::fprintf(stderr, "FAIL: no fixtures in %s\n", argv[1]);
    return 1;
  }
  std::printf("%s: %d/%d cases match HF; load %.0f ms, encode %.2f ms total\n",
              failures == 0 ? "PASS" : "FAIL", cases - failures, cases, load_ms,
              encode_ms);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
