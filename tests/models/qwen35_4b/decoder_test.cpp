#include "models/qwen35/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/q4nx_projections.hpp"
#include "models/qwen35/reference.hpp"
#include "models/qwen35_4b/npu_decoder.hpp"

// The fast decode step against the reference, token by token, with the
// same projections (q4nx weights on the CPU), so only the non-matrix math is
// compared. The decoder keeps the attention K/V cache in fp16 (the
// reference in fp32), which accounts for most of the difference.
//   GUFO_NPU_QWEN35_4B_GGUF    UD-Q4_K_XL GGUF of Qwen3.5-4B
//   GUFO_NPU_QWEN35_4B_ORACLE  oracle output directory (for prompt ids)
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;

// fp16 K/V: 1.9e-4 measured on the chat prompt (2B: 6.2e-4).
constexpr double kLogitTolerance = 2e-3;

std::vector<int> ReadIds(const fs::path& path) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)), {});
  std::replace_if(
      text.begin(), text.end(),
      [](char ch) { return ch == '[' || ch == ']' || ch == ','; }, ' ');
  std::istringstream stream(text);
  std::vector<int> ids;
  for (int id; stream >> id;) {
    ids.push_back(id);
  }
  return ids;
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_NPU_QWEN35_4B_GGUF");
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_4B_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_4B_GGUF and GUFO_NPU_QWEN35_4B_ORACLE");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  const qwen35::Q4nxProjections projections(*file,
                                            gufo_npu::qwen35_4b::kQ4nxFit);
  const qwen35::Reference reference(*file);
  qwen35::Decoder decoder(*file, projections);
  const auto vocab = static_cast<std::size_t>(decoder.config().vocab);

  const auto ids = ReadIds(fs::path(oracle) / "chat.ids.json");
  auto ref_state = reference.NewState();
  auto state = decoder.NewState();
  double worst = 0;
  std::size_t argmax_match = 0;
  for (const int id : ids) {
    const auto want =
        reference.Forward({id}, ref_state, true, false, &projections).logits;
    const auto& got = decoder.Step(id, state);
    double diff = 0;
    double norm = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      diff += (static_cast<double>(got[i]) - want[i]) *
              (static_cast<double>(got[i]) - want[i]);
      norm += static_cast<double>(want[i]) * want[i];
    }
    worst = std::max(worst, std::sqrt(diff / norm));
    argmax_match += (std::max_element(got.begin(), got.end()) - got.begin()) ==
                    (std::max_element(want.begin(), want.end()) - want.begin());
  }
  const bool ok = worst < kLogitTolerance && argmax_match == ids.size();
  std::printf(
      "%s decoder vs reference: %zu tokens; worst logits rel err %.2e; "
      "argmax %zu/%zu\n",
      ok ? "ok  " : "FAIL", ids.size(), worst, argmax_match, ids.size());
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
