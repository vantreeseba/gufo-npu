#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/q4nx.hpp"

// The q4nx GEMV design (kernels/designs/q4nx_gemv.py): one xclbin in one
// hardware context, one instruction sequence per variant (shape, input stage,
// output stage, 1 or 2 tokens). A Q4nxGemv keeps one matrix resident on the
// device; its calls read and write device Buffers, so the output of one call
// can be the input region of the next and a Chain can run them back to back.
// A two-token call reads the weights once for both tokens.
namespace gufo_npu::npu {

// The longest input, in values.
inline constexpr std::size_t kGemvMaxDim = 6144;
// Residual width of the resnorm input stage.
inline constexpr std::size_t kGemvHidden = 2048;
// Tokens a call can carry.
inline constexpr std::size_t kGemvTokens = 2;
// The input region (every call reads all of it): kGemvTokens slots of
// kGemvSlot fp32, then a kGemvHidden bf16 norm weight at kGemvNormOffset.
inline constexpr std::size_t kGemvSlot = 2 * kGemvHidden;  // fp32
inline constexpr std::size_t kGemvNormOffset =
    kGemvTokens * kGemvSlot * sizeof(float);  // bytes
inline constexpr std::size_t kGemvInputBytes =
    kGemvNormOffset + kGemvHidden * sizeof(std::uint16_t);
// The residual buffer: token t's kGemvHidden fp32 at t * kGemvSlot.
inline constexpr std::size_t kGemvResidualBytes =
    (kGemvSlot + kGemvHidden) * sizeof(float);

enum class Prologue {
  kPlain,    // slot t: dim fp32
  kAct,      // dim bf16 per token, back to back (a compact SwiGLU output)
  kResNorm,  // slot t: [H | D] (dim fp32 each), norm weight after the slots
             // -> rms(H + D) * w; H + D also goes to the residual buffer
};
enum class Epilogue {
  kStore,   // padded_rows fp32 per token
  kSwiGlu,  // rows 16 gate, then their 16 up, per 32 -> silu(g) * u as
            // compact bf16 (padded_rows / 2 per token)
};

// The instruction sequence name, e.g.
// "q4nx_gemv_k2048_n8192_resnorm_store_t2".
std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows,
                             Prologue prologue, Epilogue epilogue,
                             std::size_t tokens);

class GemvRuntime {
public:
  // Loads q4nx_gemv.xclbin from `kernel_dir`. Throws when it is missing or
  // the NPU cannot be opened.
  explicit GemvRuntime(const std::filesystem::path& kernel_dir);
  ~GemvRuntime();
  GemvRuntime(const GemvRuntime&) = delete;
  GemvRuntime& operator=(const GemvRuntime&) = delete;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

// Host-visible device memory, or a window of it.
class Buffer {
public:
  Buffer(GemvRuntime& runtime, std::size_t bytes);
  ~Buffer();
  Buffer(const Buffer&);
  Buffer& operator=(const Buffer&);

  // `bytes` at `offset` of this buffer; offset must be a multiple of 64.
  Buffer Sub(std::size_t offset, std::size_t bytes) const;
  void* data();
  template<typename T>
  T* as() {
    return static_cast<T*>(data());
  }
  std::size_t size() const;
  void ToDevice();
  void FromDevice();

  struct Impl;
  Impl& impl() const { return *impl_; }

private:
  explicit Buffer(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class Q4nxGemv {
public:
  // Uploads `weights`. Throws when no sequence was compiled for the variant.
  Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights,
           Prologue prologue = Prologue::kPlain,
           Epilogue epilogue = Epilogue::kStore);
  ~Q4nxGemv();
  Q4nxGemv(const Q4nxGemv&) = delete;
  Q4nxGemv& operator=(const Q4nxGemv&) = delete;

  // One call on device buffers for `tokens` (1 or 2): x (kGemvInputBytes),
  // y (y_bytes()), residual (kGemvResidualBytes, written by kResNorm).
  void Run(Buffer& x, Buffer& y, Buffer& residual, std::size_t tokens = 1);

  // Convenience for a plain/store GEMV on host memory: x has dim fp32
  // values (rounded to bf16 on the device), y gets `rows` values.
  void Run(const float* x, float* y);
  // x: dim bf16 bits (widened to fp32 on the host). y: padded_rows fp32.
  void Run(const std::uint16_t* x, float* y);

  // Mean and best wall time of start + wait, in microseconds, over
  // `iterations` warm dispatches on the buffers of the last Run.
  struct Timing {
    double mean_us = 0;
    double best_us = 0;
  };
  Timing Time(int iterations);

  std::size_t dim() const { return dim_; }
  std::size_t rows() const { return rows_; }
  std::size_t padded_rows() const { return padded_rows_; }
  std::size_t weight_bytes() const;
  // Where token t's output starts in y: t * y_token_stride() fp32 (store) or
  // fp32 words of packed bf16 (SwiGLU). Calls that output a hidden vector
  // put token t in slot t of the next input region (kGemvSlot apart).
  std::size_t y_token_stride() const { return y_token_stride_; }
  // Bytes a y buffer must span (both tokens' regions, and the SwiGLU
  // padding the design discards past them).
  std::size_t y_bytes() const;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
  std::size_t dim_ = 0;
  std::size_t rows_ = 0;
  std::size_t padded_rows_ = 0;
  std::size_t y_token_stride_ = 0;
  Epilogue epilogue_ = Epilogue::kStore;
};

// Calls submitted together and run back to back (an XRT runlist).
class Chain {
public:
  explicit Chain(GemvRuntime& runtime);
  ~Chain();
  Chain(const Chain&) = delete;
  Chain& operator=(const Chain&) = delete;

  // The buffers must outlive the chain.
  void Add(Q4nxGemv& gemv, Buffer& x, Buffer& y, Buffer& residual,
           std::size_t tokens = 1);
  // Runs every call in order and waits for the last.
  void Run();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::npu
