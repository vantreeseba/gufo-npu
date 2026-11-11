#include "models/qwen3vl_embed_2b/npu/vit_attention.hpp"

#include <chrono>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

#include "core/bfp16.hpp"
#include "npu/gemv.hpp"
#include "npu/program.hpp"

namespace gufo_npu::qwen3vl_embed_2b::npu {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kHeads = kVitAttnHeads;
constexpr int kHd = kVitAttnHeadDim;
constexpr int kDim = kHeads * kHd;  // q, k and v each
constexpr int kDimTiles = kHd / 8;
constexpr int kKeyTiles = kVitAttnKeyBlock / 8;
constexpr int kRows = kVitAttnBucket / kVitAttnQueryBlock;  // cores per column
constexpr std::size_t kSplitBytes = 2 * kVitAttnTileBytes;
// q is scaled by 1 / sqrt(head_dim) already; the cores take base-2 scores.
constexpr float kLog2e = 1.4426950408889634f;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void SplitTileScalar(const float* tile, std::uint8_t* out) {
  for (int r = 0; r < 8; ++r) {
    float rest[8];
    float hi[8];
    bfp16::EncodeBlock(tile + r * 8, out + r * bfp16::kBlockBytes);
    bfp16::DecodeBlock(out + r * bfp16::kBlockBytes, hi);
    for (int c = 0; c < 8; ++c) {
      rest[c] = tile[r * 8 + c] - hi[c];
    }
    bfp16::EncodeBlock(rest, out + kVitAttnTileBytes + r * bfp16::kBlockBytes);
  }
}

#if defined(__x86_64__) && defined(__GNUC__)
#define GUFO_QWEN3VL_VIT_ATTN_AVX512 1

// Two blocks of 8 floats as bfp16ebs8 (2 x 9 bytes at out), returning the
// values they decode to. A block's exponent byte is its largest biased
// exponent m and its mantissas round(x 2^(133 - m)), ties to even (the
// default MXCSR mode), saturated to [-128, 127], as bfp16::EncodeBlock. A
// block with m < 7 (all |x| < 2^-120) encodes as zeros.
__attribute__((target("avx512f"))) __m512 Encode16(__m512 x,
                                                   std::uint8_t* out) {
  const __m512i lane =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512i e = _mm512_and_si512(_mm512_srli_epi32(_mm512_castps_si512(x), 23),
                               _mm512_set1_epi32(0xFF));
  for (const int step : {1, 2, 4}) {  // the max over each half's 8 lanes
    const __m512i other = _mm512_xor_si512(lane, _mm512_set1_epi32(step));
    e = _mm512_max_epi32(e, _mm512_permutexvar_epi32(other, e));
  }
  const __mmask16 live = _mm512_cmpge_epi32_mask(e, _mm512_set1_epi32(7));
  const __m512 scale = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(260), e), 23));
  const __m512 unscale = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_sub_epi32(e, _mm512_set1_epi32(6)), 23));
  __m512i man = _mm512_maskz_cvtps_epi32(live, _mm512_mul_ps(x, scale));
  man = _mm512_max_epi32(_mm512_min_epi32(man, _mm512_set1_epi32(127)),
                         _mm512_set1_epi32(-128));
  alignas(16) std::uint8_t bytes[16];
  _mm_store_si128(reinterpret_cast<__m128i*>(bytes), _mm512_cvtepi32_epi8(man));
  alignas(64) std::int32_t exps[16];
  _mm512_store_si512(exps, e);
  out[0] = static_cast<std::uint8_t>(exps[0]);
  std::memcpy(out + 1, bytes, 8);
  out[9] = static_cast<std::uint8_t>(exps[8]);
  std::memcpy(out + 10, bytes + 8, 8);
  return _mm512_maskz_mul_ps(live, _mm512_cvtepi32_ps(man), unscale);
}

__attribute__((target("avx512f"))) void SplitTileAvx512(const float* tile,
                                                        std::uint8_t* out) {
  for (int pair = 0; pair < 4; ++pair) {  // tile rows 2 pair, 2 pair + 1
    const __m512 x = _mm512_loadu_ps(tile + pair * 16);
    const __m512 hi = Encode16(x, out + pair * 2 * bfp16::kBlockBytes);
    Encode16(_mm512_sub_ps(x, hi),
             out + kVitAttnTileBytes + pair * 2 * bfp16::kBlockBytes);
  }
}
#endif

}  // namespace

int VitAttnLength(int patches) {
  if (patches < 1 || patches > kVitAttnMaxPatches) {
    throw std::invalid_argument(
        "qwen3vl vit attention: " + std::to_string(patches) +
        " patches, the program takes 1 to " +
        std::to_string(kVitAttnMaxPatches));
  }
  return (patches + kVitAttnBucket - 1) / kVitAttnBucket * kVitAttnBucket;
}

std::string VitAttnSequence(int length) {
  return std::string(kVitAttnName) + "_n" + std::to_string(length);
}

namespace detail {

bool HasAvx512() {
#ifdef GUFO_QWEN3VL_VIT_ATTN_AVX512
  static const bool has = __builtin_cpu_supports("avx512f");
  return has;
#else
  return false;
#endif
}

void SplitTile(const float* tile, std::uint8_t* out, bool avx512) {
  if (!avx512) {
    SplitTileScalar(tile, out);
    return;
  }
#ifdef GUFO_QWEN3VL_VIT_ATTN_AVX512
  if (HasAvx512()) {
    SplitTileAvx512(tile, out);
    return;
  }
#endif
  throw std::invalid_argument("qwen3vl vit attention: the CPU has no AVX-512");
}

}  // namespace detail

struct VitAttention::Impl {
  gufo_npu::npu::Program program;
  // Operands for the longest image; each call uses windows of the first
  // bytes its length needs.
  gufo_npu::npu::Buffer q, kv, o;
  std::optional<gufo_npu::npu::Buffer> q_win, kv_win, o_win;
  std::optional<gufo_npu::npu::Batch> batch;
  // Their host memory.
  std::uint8_t* q_host;
  std::uint8_t* kv_host;
  const std::uint16_t* o_host;
  int patches = 0;
  int length = 0;
  bool avx512 = detail::HasAvx512();
  Profile profile;

  static std::size_t QBytes(int length) {
    return static_cast<std::size_t>(kHeads) * (length / kVitAttnQueryBlock) *
           kVitAttnQBytes;
  }
  static std::size_t KVBytes(int length) {
    return static_cast<std::size_t>(kHeads) * (length / kVitAttnKeyBlock) *
           kVitAttnKVBytes;
  }
  static std::size_t OBytes(int length) {
    return static_cast<std::size_t>(kHeads) * (length / kVitAttnQueryBlock) *
           kRows * kVitAttnOBytes;
  }

  explicit Impl(const std::filesystem::path& dir)
      : program(dir, kVitAttnName),
        q(program, QBytes(VitAttnLength(kVitAttnMaxPatches))),
        kv(program, KVBytes(VitAttnLength(kVitAttnMaxPatches))),
        o(program, OBytes(VitAttnLength(kVitAttnMaxPatches))),
        q_host(q.as<std::uint8_t>()),
        kv_host(kv.as<std::uint8_t>()),
        o_host(o.as<std::uint16_t>()) {}

  void Reset(int n) {
    const int len = VitAttnLength(n);
    if (len != length) {
      const std::string sequence = VitAttnSequence(len);
      if (!program.Has(sequence)) {
        throw std::runtime_error("qwen3vl vit attention: " + sequence +
                                 " was not built");
      }
      q_win.emplace(q.Sub(0, QBytes(len)));
      kv_win.emplace(kv.Sub(0, KVBytes(len)));
      o_win.emplace(o.Sub(0, OBytes(len)));
      batch.emplace(program);
      batch->Add(sequence, {&*q_win, &*kv_win, &*o_win});
      length = len;
    }
    patches = n;
    // The key masks: 0 for the image's patches, -inf for the padding.
    const int blocks = len / kVitAttnKeyBlock;
    for (int h = 0; h < kHeads; ++h) {
      for (int b = 0; b < blocks; ++b) {
        float bias[kVitAttnKeyBlock];
        for (int j = 0; j < kVitAttnKeyBlock; ++j) {
          bias[j] = b * kVitAttnKeyBlock + j < n
                        ? 0.0f
                        : -std::numeric_limits<float>::infinity();
        }
        std::memcpy(
            kv_host +
                (static_cast<std::size_t>(h) * blocks + b) * kVitAttnKVBytes +
                kVitAttnKSplitBytes,
            bias, sizeof(bias));
      }
    }
  }

  // Head h's 8-row tile t (rows 8t .. 8t + 7): its q, k and v tiles, split.
  void PackTile(const float* qkv, int h, int t) {
    const int groups = length / kVitAttnBucket;
    const int blocks = length / kVitAttnKeyBlock;
    const int qblock = t / 8;  // 8 query tiles per block
    std::uint8_t* qdst =
        q_host +
        (static_cast<std::size_t>(h) * groups * kRows + qblock) *
            kVitAttnQBytes +
        static_cast<std::size_t>(t % 8) * kDimTiles * kSplitBytes;
    std::uint8_t* obj =
        kv_host + (static_cast<std::size_t>(h) * blocks + t / kKeyTiles) *
                      kVitAttnKVBytes;
    const int kt = t % kKeyTiles;
    // The tile's rows, zero past the image.
    float rows[8][3 * kHd];
    for (int r = 0; r < 8; ++r) {
      const int row = t * 8 + r;
      if (row >= patches) {
        std::memset(rows[r], 0, sizeof(rows[r]));
        continue;
      }
      const float* src =
          qkv + static_cast<std::size_t>(row) * 3 * kDim + h * kHd;
      for (int part = 0; part < 3; ++part) {
        std::memcpy(rows[r] + part * kHd, src + part * kDim,
                    kHd * sizeof(float));
      }
    }
    alignas(64) float tile[64];
    for (int dt = 0; dt < kDimTiles; ++dt) {
      for (int r = 0; r < 8; ++r) {  // q: row = query, column = dim
        for (int c = 0; c < 8; ++c) {
          tile[r * 8 + c] = rows[r][dt * 8 + c] * kLog2e;
        }
      }
      detail::SplitTile(tile, qdst + dt * kSplitBytes, avx512);
      for (int r = 0; r < 8; ++r) {  // k: row = key, column = dim
        std::memcpy(tile + r * 8, rows[r] + kHd + dt * 8, 8 * sizeof(float));
      }
      detail::SplitTile(tile, obj + (dt * kKeyTiles + kt) * kSplitBytes,
                        avx512);
      for (int r = 0; r < 8; ++r) {  // v^T: row = dim, column = key
        for (int c = 0; c < 8; ++c) {
          tile[r * 8 + c] = rows[c][2 * kHd + dt * 8 + r];
        }
      }
      detail::SplitTile(
          tile,
          obj + kVitAttnKBytes +
              (((dt / 4) * kKeyTiles + kt) * 4 + dt % 4) * kSplitBytes,
          avx512);
    }
  }

  void Run(const float* qkv, const ParallelRun& run) {
    if (length == 0) {
      throw std::logic_error("qwen3vl vit attention: Run before Reset");
    }
    const auto start = Clock::now();
    const int tiles = length / 8;
    run(static_cast<std::size_t>(kHeads) * tiles, [&](std::size_t i) {
      PackTile(qkv, static_cast<int>(i / tiles), static_cast<int>(i % tiles));
    });
    const auto npu_start = Clock::now();
    profile.pack_seconds +=
        std::chrono::duration<double>(npu_start - start).count();
    q_win->ToDevice();
    kv_win->ToDevice();
    batch->Run();
    o_win->FromDevice();
    profile.npu_seconds += SecondsSince(npu_start);
    ++profile.calls;
  }

  void Output(int row, std::uint16_t* out) const {
    if (row < 0 || row >= patches) {
      throw std::out_of_range("qwen3vl vit attention: row out of range");
    }
    const int groups = length / kVitAttnBucket;
    const int block = row / kVitAttnQueryBlock;
    const int qt = row % kVitAttnQueryBlock / 8;
    for (int h = 0; h < kHeads; ++h) {
      // O[h][group][quarter][core]: query block 4 group + core, query tiles
      // 2 quarter and 2 quarter + 1.
      const std::size_t obj =
          ((static_cast<std::size_t>(h) * groups + block / kRows) * kRows +
           qt / 2) *
              kRows +
          block % kRows;
      const std::uint16_t* tiles = o_host + obj * (kVitAttnOBytes / 2) +
                                   (qt % 2) * kDimTiles * 64 + row % 8 * 8;
      for (int dt = 0; dt < kDimTiles; ++dt) {
        std::memcpy(out + h * kHd + dt * 8, tiles + dt * 64,
                    8 * sizeof(std::uint16_t));
      }
    }
  }
};

VitAttention::VitAttention(const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(kernel_dir)) {}

VitAttention::~VitAttention() = default;

void VitAttention::Reset(int patches) {
  impl_->Reset(patches);
}

void VitAttention::Run(const float* qkv, const ParallelRun& run) {
  impl_->Run(qkv, run);
}

void VitAttention::Output(int row, std::uint16_t* out) const {
  impl_->Output(row, out);
}

const VitAttention::Profile& VitAttention::profile() const {
  return impl_->profile;
}

}  // namespace gufo_npu::qwen3vl_embed_2b::npu
