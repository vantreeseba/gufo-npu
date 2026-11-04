#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "core/q4nx.hpp"
#include "core/simd.hpp"

// What the CPU's q4nx loops share (cpu_experts.cpp, cpu_gemv.cpp): a q4nx
// block's layout (src/core/q4nx.hpp), where a row tile's blocks sit, and the
// AVX-512 code permutations.
namespace gufo_npu::qwen36_35b_a3b::q4nx_tile {

inline constexpr std::size_t kTile = q4nx::kMTile;  // rows per tile
inline constexpr std::size_t kHalf = kTile / 2;
inline constexpr std::size_t kKTile = q4nx::kKTile;
inline constexpr std::size_t kGroups = kKTile / q4nx::kGroup;
// Byte offsets in a block: per row bf16 d and dmin; per group and row u8 sc
// and i8 mn; then the codes, [m / 16][g][l][m % 16], two per byte.
inline constexpr std::size_t kD = 0;
inline constexpr std::size_t kDmin = kD + kTile * 2;
inline constexpr std::size_t kSc = kDmin + kTile * 2;
inline constexpr std::size_t kMn = kSc + kGroups * kTile;
inline constexpr std::size_t kCodes = kMn + kGroups * kTile;
static_assert(kCodes + kTile * kKTile / 2 == q4nx::kBlockBytes);

// The block of row tile `tile` and k block `kb` (q4nx's GEMV order).
inline const std::uint8_t* BlockAt(const std::uint8_t* matrix,
                                   std::size_t k_blocks, std::size_t tile,
                                   std::size_t kb) {
  const std::size_t index =
      ((tile / q4nx::kRows) * k_blocks + kb) * q4nx::kRows + tile % q4nx::kRows;
  return matrix + index * q4nx::kBlockBytes;
}

inline float Bf16At(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return simd::FromBf16(v);
}

#if defined(__x86_64__)
// Whether the AVX-512 BF16 and VBMI paths can run on this CPU.
inline bool HasFast() {
  static const bool has = __builtin_cpu_supports("avx512f") &&
                          __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vbmi") &&
                          __builtin_cpu_supports("avx512bf16");
  return has;
}

// Byte permutations from a 64-byte run of codes (8 inputs l of one half and
// group: low nibbles in `lo`, high in `hi`; lo[l * 8 + b] is row 2b's code,
// hi[l * 8 + b] row 2b + 1's) to pairs (l, l + 1) per row: byte
// q * 32 + r * 2 + s is row r's code of input 2q + s, for pairs q = 0, 1
// (low) and 2, 3 (high).
struct Perms {
  alignas(64) std::uint8_t low[64];
  alignas(64) std::uint8_t high[64];
  Perms() {
    for (int q = 0; q < 4; ++q) {
      for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 2; ++s) {
          const int value = ((r & 1) != 0 ? 64 : 0) + (2 * q + s) * 8 + r / 2;
          const int byte = (q % 2) * 32 + r * 2 + s;
          (q < 2 ? low : high)[byte] = static_cast<std::uint8_t>(value);
        }
      }
    }
  }
};
#else
inline bool HasFast() {
  return false;
}
#endif

}  // namespace gufo_npu::qwen36_35b_a3b::q4nx_tile
