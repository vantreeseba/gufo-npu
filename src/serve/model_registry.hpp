#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "models/qwen35/config.hpp"

// Which model package runs a GGUF on the NPU. Each package has its own NPU
// programs and prompt processing, built for one model's shapes:
//   kQwen35_2B      Qwen3.5-2B: the family decoder (src/models/qwen35) and
//                   the family prefill.
//   kQwen35_4B      Qwen3.5-4B: src/models/qwen35_4b (hidden 2,560).
//   kQwen36_35B_A3B Qwen3.6-35B-A3B: the family decoder (MoE) and
//                   src/models/qwen36_35b_a3b's prefill.
namespace gufo_npu::serve {

enum class ModelPackage { kQwen35_2B, kQwen35_4B, kQwen36_35B_A3B };

// The package for a model's config, or nullopt when no NPU package runs
// these shapes.
std::optional<ModelPackage> NpuPackageFor(const qwen35::Config& config);

// The package's model name, e.g. "Qwen3.6-35B-A3B".
const char* PackageName(ModelPackage package);

// A model id for clients: lowercase, spaces and underscores as '-', other
// characters outside [a-z0-9.-] dropped; e.g. "Qwen3.6 35B-A3B" ->
// "qwen3.6-35b-a3b".
std::string NormalizeModelId(std::string_view name);

// The default per-request limit on prompt + generated tokens. The decoders'
// K/V caches and the prefills grow with the prompt, so the real limits are
// memory and time; this is the length the engine is checked at
// (docs/SERVING.md). Raise it with EngineOptions::context_length, up to the
// GGUF's trained context length.
inline constexpr int kDefaultContextLength = 32768;

}  // namespace gufo_npu::serve
