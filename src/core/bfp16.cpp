#include "core/bfp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/parallel.hpp"

namespace gufo_npu::bfp16 {
namespace {

constexpr std::size_t kColA = kCtK / kBlock;  // 8 x 8 blocks per k slice

void CheckShape(std::size_t rows, std::size_t dim) {
  if (rows == 0 || dim == 0 || rows % kNTile != 0 || dim % kKTile != 0) {
    throw std::invalid_argument(
        "bfp16 GEMM weights need rows a multiple of " + std::to_string(kNTile) +
        " and dim of " + std::to_string(kKTile) + ", got " +
        std::to_string(rows) + " x " + std::to_string(dim));
  }
}

// Where the block holding W[row][k0 .. k0 + 7] (k0 a multiple of 8) goes, in
// blocks. pack_b's axes: k = ((kb * 4 + ks) * 16 + i) * 8 + s and
// n = (cb * 8 + tb) * 8 + t, ordered (cb, kb, ks, tb, i, t) with s inside a
// block.
std::size_t BlockIndex(std::size_t row, std::size_t k0, std::size_t dim) {
  const std::size_t kb = k0 / kKTile;
  const std::size_t ks = k0 % kKTile / kCtK;
  const std::size_t i = k0 % kCtK / kBlock;
  const std::size_t cb = row / kNTile;
  const std::size_t tb = row % kNTile / kBlock;
  const std::size_t t = row % kBlock;
  const std::size_t k_blocks = dim / kKTile;
  return ((((cb * k_blocks + kb) * (kKTile / kCtK) + ks) * (kNTile / kBlock) +
           tb) *
              kColA +
          i) *
             kBlock +
         t;
}

// round(mag / 2^shift), ties to even, for |mag| < 2^24 and shift >= 1.
std::int64_t RoundShift(std::int64_t mag, int shift) {
  if (shift >= 25) {
    return 0;  // |mag| / 2^shift < 1/2
  }
  const std::int64_t q = mag >> shift;  // floor
  const std::int64_t r = mag - (q << shift);
  const std::int64_t half = std::int64_t{1} << (shift - 1);
  return (r > half || (r == half && (q & 1) != 0)) ? q + 1 : q;
}

}  // namespace

void EncodeBlock(const float* in, std::uint8_t* out) {
  std::uint32_t bits[kBlock];
  std::memcpy(bits, in, sizeof(bits));
  int exps[kBlock];
  int max_exp = 0;
  for (std::size_t i = 0; i < kBlock; ++i) {
    exps[i] = static_cast<int>((bits[i] >> 23) & 0xFF);
    max_exp = std::max(max_exp, exps[i]);
  }
  out[0] = static_cast<std::uint8_t>(max_exp);
  for (std::size_t i = 0; i < kBlock; ++i) {
    const bool negative = (bits[i] >> 31) != 0;
    std::int64_t man = bits[i] & 0x7FFFFF;
    if (exps[i] != 0) {
      man |= 0x800000;
    }
    const int shift = max_exp - exps[i];
    std::int64_t v;
    if (shift >= 32) {
      v = negative ? -1 : 0;
    } else {
      // 17 keeps 7 magnitude bits plus the sign; shift aligns to the block.
      v = RoundShift(negative ? -man : man, 17 + shift);
    }
    v = std::clamp<std::int64_t>(v, -128, 127);
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(v));
  }
}

void DecodeBlock(const std::uint8_t* in, float* out) {
  const double scale = std::ldexp(1.0, static_cast<int>(in[0]) - 127 - 6);
  for (std::size_t i = 0; i < kBlock; ++i) {
    out[i] = static_cast<float>(static_cast<std::int8_t>(in[1 + i]) * scale);
  }
}

std::size_t PackedBytes(std::size_t rows, std::size_t dim) {
  return rows * dim / kBlock * kBlockBytes;
}

std::vector<std::uint8_t> PackGemmB(const float* w, std::size_t rows,
                                    std::size_t dim) {
  CheckShape(rows, dim);
  std::vector<std::uint8_t> out(PackedBytes(rows, dim));
  ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      for (std::size_t k0 = 0; k0 < dim; k0 += kBlock) {
        EncodeBlock(w + r * dim + k0,
                    out.data() + BlockIndex(r, k0, dim) * kBlockBytes);
      }
    }
  });
  return out;
}

std::vector<float> UnpackGemmB(const std::vector<std::uint8_t>& packed,
                               std::size_t rows, std::size_t dim) {
  CheckShape(rows, dim);
  if (packed.size() != PackedBytes(rows, dim)) {
    throw std::invalid_argument("bfp16: packed size does not match the shape");
  }
  std::vector<float> w(rows * dim);
  ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      for (std::size_t k0 = 0; k0 < dim; k0 += kBlock) {
        DecodeBlock(packed.data() + BlockIndex(r, k0, dim) * kBlockBytes,
                    w.data() + r * dim + k0);
      }
    }
  });
  return w;
}

}  // namespace gufo_npu::bfp16
