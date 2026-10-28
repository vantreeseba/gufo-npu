// The model registry: which NPU package runs which shapes, and model ids.
#include "serve/model_registry.hpp"

#include <cstdio>
#include <string>

namespace {

using gufo_npu::qwen35::Config;
using gufo_npu::serve::ModelPackage;
using gufo_npu::serve::NormalizeModelId;
using gufo_npu::serve::NpuPackageFor;

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

Config Dense(int hidden) {
  Config c;
  c.architecture = "qwen35";
  c.hidden = hidden;
  return c;
}

Config Moe(int hidden) {
  Config c;
  c.architecture = "qwen35moe";
  c.hidden = hidden;
  c.experts = 256;
  c.experts_used = 8;
  return c;
}

}  // namespace

int main() {
  Expect(NpuPackageFor(Dense(2048)) == ModelPackage::kQwen35_2B,
         "Qwen3.5-2B shapes run on the family package");
  Expect(NpuPackageFor(Dense(2560)) == ModelPackage::kQwen35_4B,
         "Qwen3.5-4B shapes run on its own package");
  Expect(NpuPackageFor(Moe(2048)) == ModelPackage::kQwen36_35B_A3B,
         "Qwen3.6-35B-A3B shapes run on its package");
  Expect(!NpuPackageFor(Dense(1024)), "other dense shapes have no package");
  Expect(!NpuPackageFor(Moe(4096)), "other MoE shapes have no package");
  Config wrong = Dense(2048);
  wrong.architecture = "llama";
  Expect(!NpuPackageFor(wrong), "other architectures have no package");
  Config mismatched = Moe(2048);
  mismatched.architecture = "qwen35";
  Expect(!NpuPackageFor(mismatched), "a dense arch with experts is refused");

  Expect(NormalizeModelId("Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b",
         "ids are lowercase");
  Expect(NormalizeModelId("Qwen3.5 4B_MTP") == "qwen3.5-4b-mtp",
         "spaces and underscores become dashes");
  Expect(NormalizeModelId("a/b:c") == "abc", "other characters are dropped");

  if (failures == 0) {
    std::puts("ok: model registry");
  }
  return failures == 0 ? 0 : 1;
}
