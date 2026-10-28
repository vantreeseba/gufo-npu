#include "serve/model_registry.hpp"

#include <cctype>

namespace gufo_npu::serve {

namespace {

// The shapes each package's NPU programs are built for.
constexpr int kFamilyHidden = 2048;  // Qwen3.5-2B, Qwen3.6-35B-A3B
constexpr int kQwen35_4BHidden = 2560;

}  // namespace

std::optional<ModelPackage> NpuPackageFor(const qwen35::Config& config) {
  if (config.architecture == "qwen35" && !config.IsMoe()) {
    if (config.hidden == kFamilyHidden) {
      return ModelPackage::kQwen35_2B;
    }
    if (config.hidden == kQwen35_4BHidden) {
      return ModelPackage::kQwen35_4B;
    }
  }
  if (config.architecture == "qwen35moe" && config.IsMoe() &&
      config.hidden == kFamilyHidden) {
    return ModelPackage::kQwen36_35B_A3B;
  }
  return std::nullopt;
}

const char* PackageName(ModelPackage package) {
  switch (package) {
    case ModelPackage::kQwen35_2B:
      return "Qwen3.5-2B";
    case ModelPackage::kQwen35_4B:
      return "Qwen3.5-4B";
    case ModelPackage::kQwen36_35B_A3B:
      return "Qwen3.6-35B-A3B";
  }
  return "unknown";
}

std::string NormalizeModelId(std::string_view name) {
  std::string id;
  for (const char ch : name) {
    const auto c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) != 0) {
      id.push_back(static_cast<char>(std::tolower(c)));
    } else if (ch == '.' || ch == '-') {
      id.push_back(ch);
    } else if (ch == ' ' || ch == '_') {
      id.push_back('-');
    }
  }
  return id;
}

}  // namespace gufo_npu::serve
