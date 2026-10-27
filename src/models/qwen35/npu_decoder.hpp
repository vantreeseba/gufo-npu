#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"

// Qwen3.5 decode on the NPU. Every matrix product, every RMS norm before one,
// the residual adds and SwiGLU run in the q4nx GEMV (its resnorm input stage
// and SwiGLU output stage); the residual stream stays on the device. Each
// GEMV writes into the input region of the next, so the calls from one
// layer's mixer output to the next layer's input projection (out, gate + up,
// down, next in) run as one chain. The CPU runs the token mixers (Gated
// DeltaNet, attention) and the DeltaNet alpha/beta rows: 25 NPU submissions
// per token.
namespace gufo_npu::qwen35 {

class NpuDecoder {
public:
  // Keeps pointers into `file`, which must outlive it.
  NpuDecoder(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuDecoder();
  NpuDecoder(const NpuDecoder&) = delete;
  NpuDecoder& operator=(const NpuDecoder&) = delete;

  const Config& config() const;
  DecodeState NewState() const;

  // Runs `token` at state.position, advances the state and returns the
  // logits (vocab values, valid until the next Step).
  const std::vector<float>& Step(int token, DecodeState& state);

  // Wall time inside NPU submissions since construction, and their count.
  double npu_seconds() const;
  long submissions() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen35
