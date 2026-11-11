#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "npu/gemm.hpp"

// How the NPU embedder and the NPU vision tower run a GEMM over a batch of
// rows: in chunks of up to npu::kGemmMaxChunk rows, two calls in flight, so
// the CPU writes the next chunk's A and reads the previous chunk's C while
// the NPU multiplies.
namespace gufo_npu::qwen3vl_embed_2b::npu {

// One GEMM call's rows of the batch: n rows from `begin`, run as m rows (n
// rounded up to a chunk size of npu::kGemmChunks).
struct Chunk {
  int begin = 0;
  int n = 0;
  int m = 0;
};

inline std::vector<Chunk> Chunks(int rows) {
  std::vector<Chunk> chunks;
  const int max = static_cast<int>(gufo_npu::npu::kGemmMaxChunk);
  for (int begin = 0; begin < rows;) {
    const int n = std::min(max, rows - begin);
    int m = max;
    for (const std::size_t size : gufo_npu::npu::kGemmChunks) {
      if (static_cast<int>(size) >= n) {
        m = static_cast<int>(size);
        break;
      }
    }
    chunks.push_back({begin, n, m});
    begin += n;
  }
  return chunks;
}

class GemmPass {
public:
  using Bits = std::uint16_t;  // bf16

  // Device operands for every GEMM in `gemms` at the largest chunk.
  GemmPass(gufo_npu::npu::GemmRuntime& runtime,
           const std::vector<const gufo_npu::npu::Bfp16Gemm*>& gemms) {
    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    for (const auto* g : gemms) {
      a_bytes = std::max(a_bytes, g->a_bytes(gufo_npu::npu::kGemmMaxChunk));
      c_bytes = std::max(c_bytes, g->c_bytes(gufo_npu::npu::kGemmMaxChunk));
    }
    for (Slot& s : slots_) {
      s.a = std::make_unique<gufo_npu::npu::GemmBuffer>(runtime, a_bytes);
      s.c = std::make_unique<gufo_npu::npu::GemmBuffer>(runtime, c_bytes);
    }
  }

  // Runs `g` over every chunk, two calls in flight: chunk i+1 is written
  // (prepare) and started before chunk i is waited for and read (consume).
  // prepare(chunk, A) fills A's first chunk.n rows (g.dim() bf16 each);
  // consume(chunk, C) reads C's first chunk.n rows (g.rows() bf16).
  template<typename Prepare, typename Consume>
  void Run(gufo_npu::npu::Bfp16Gemm& g, const std::vector<Chunk>& chunks,
           const Prepare& prepare, const Consume& consume) {
    if (chunks.empty()) {
      return;
    }
    const auto start = [&](std::size_t i) {
      Slot& s = slots_[i % 2];
      const Chunk& c = chunks[i];
      prepare(c, s.a->template as<Bits>());
      const auto sync = Clock::now();
      s.a->ToDevice(static_cast<std::size_t>(c.n) * g.dim() * sizeof(Bits));
      sync_seconds += SecondsSince(sync);
      s.call = g.Start(*s.a, *s.c, static_cast<std::size_t>(c.m));
      s.c_sync = static_cast<std::size_t>(c.n) * g.rows() * sizeof(Bits);
      ++gemms;
    };
    start(0);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
      if (i + 1 < chunks.size()) {
        start(i + 1);
      }
      Slot& s = slots_[i % 2];
      auto t = Clock::now();
      s.call.Wait();
      gemm_seconds += SecondsSince(t);
      t = Clock::now();
      s.c->FromDevice(s.c_sync);
      sync_seconds += SecondsSince(t);
      consume(chunks[i], static_cast<const Bits*>(s.c->template as<Bits>()));
    }
  }

  // Since construction: NPU GEMM calls, time waiting for them, time syncing
  // their operand buffers.
  long gemms = 0;
  double gemm_seconds = 0;
  double sync_seconds = 0;

private:
  using Clock = std::chrono::steady_clock;

  static double SecondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  // A call's device operands; two, so one chunk's call can run while the
  // CPU fills the next one's A.
  struct Slot {
    std::unique_ptr<gufo_npu::npu::GemmBuffer> a, c;
    gufo_npu::npu::GemmCall call;
    std::size_t c_sync = 0;  // bytes of C to read back
  };
  Slot slots_[2];
};

}  // namespace gufo_npu::qwen3vl_embed_2b::npu
