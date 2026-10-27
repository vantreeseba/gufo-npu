#include "core/q4nx.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/parallel.hpp"
#include "core/quant.hpp"

namespace gufo_npu::q4nx {
namespace {

constexpr std::size_t kGroups = kKTile / kGroup;  // per block row
// Byte offsets in a block (see q4nx.hpp).
constexpr std::size_t kD = 0;
constexpr std::size_t kDmin = kD + kMTile * 2;
constexpr std::size_t kSc = kDmin + kMTile * 2;
constexpr std::size_t kMn = kSc + kGroups * kMTile;
constexpr std::size_t kCodes = kMn + kGroups * kMTile;
static_assert(kCodes + kMTile * kKTile / 2 == kBlockBytes);
static_assert(kCodes % 64 == 0, "the kernel loads codes 64 bytes at a time");
constexpr std::size_t kQ4KBytes = 144;  // block_q4_K

void CheckShape(std::size_t rows, std::size_t dim) {
  if (rows == 0 || dim == 0 || dim % kKTile != 0) {
    throw std::invalid_argument("q4nx needs dim a multiple of " +
                                std::to_string(kKTile) + ", got " +
                                std::to_string(dim));
  }
}

Matrix Allocate(std::size_t rows, std::size_t dim) {
  CheckShape(rows, dim);
  Matrix m;
  m.rows = rows;
  m.padded_rows = PaddedRows(rows);
  m.dim = dim;
  m.bytes.assign(m.padded_rows / kMTile * (dim / kKTile) * kBlockBytes, 0);
  return m;
}

// The block of row tile `tile` (kMTile out-features) and k block `kb`.
std::uint8_t* Block(Matrix& m, std::size_t tile, std::size_t kb) {
  const std::size_t k_blocks = m.dim / kKTile;
  const std::size_t index =
      ((tile / kRows) * k_blocks + kb) * kRows + tile % kRows;
  return m.bytes.data() + index * kBlockBytes;
}

const std::uint8_t* Block(const Matrix& m, std::size_t tile, std::size_t kb) {
  return Block(const_cast<Matrix&>(m), tile, kb);
}

void StoreU16(std::uint8_t* p, std::uint16_t v) {
  std::memcpy(p, &v, sizeof(v));
}

std::uint16_t LoadU16(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::size_t CodeIndex(std::size_t m, std::size_t k) {
  return (((m / 16) * kGroups + k / 32) * 32 + k % 32) * 16 + m % 16;
}

// Row m's super-scale and super-minimum (bf16 bits).
void SetSuper(std::uint8_t* block, std::size_t m, std::uint16_t d,
              std::uint16_t dmin) {
  StoreU16(block + kD + m * 2, d);
  StoreU16(block + kDmin + m * 2, dmin);
}

void SetSub(std::uint8_t* block, std::size_t m, std::size_t g, std::uint8_t sc,
            std::int8_t mn) {
  block[kSc + g * kMTile + m] = sc;
  block[kMn + g * kMTile + m] = static_cast<std::uint8_t>(mn);
}

// Group g's step (d * sc) and minimum (dmin * mn) for row m, exact in fp32.
void Params(const std::uint8_t* block, std::size_t m, std::size_t g,
            float* step, float* min) {
  const float d = quant::Bf16ToFloat(LoadU16(block + kD + m * 2));
  const float dmin = quant::Bf16ToFloat(LoadU16(block + kDmin + m * 2));
  *step = d * static_cast<float>(block[kSc + g * kMTile + m]);
  *min = dmin * static_cast<float>(
                    static_cast<std::int8_t>(block[kMn + g * kMTile + m]));
}

void SetCode(std::uint8_t* block, std::size_t m, std::size_t k,
             std::uint8_t code) {
  const std::size_t i = CodeIndex(m, k);
  std::uint8_t& byte = block[kCodes + i / 2];
  byte = (i % 2 == 0) ? static_cast<std::uint8_t>((byte & 0xF0) | code)
                      : static_cast<std::uint8_t>((byte & 0x0F) | (code << 4));
}

// The smallest bf16 at or above x (x >= 0).
std::uint16_t Bf16AtLeast(float x) {
  std::uint16_t bits = quant::FloatToBf16(x);
  if (quant::Bf16ToFloat(bits) < x) {
    ++bits;
  }
  return bits;
}

// One Q4_K super-block (256 in-features of out-feature m) into a q4nx block.
void PackQ4KSuper(const std::uint8_t* src, std::uint8_t* block, std::size_t m) {
  const float d = quant::HalfToFloat(LoadU16(src));
  const float dmin = quant::HalfToFloat(LoadU16(src + 2));
  const std::uint8_t* scales = src + 4;
  const std::uint8_t* qs = src + 16;
  SetSuper(block, m, quant::FloatToBf16(d), quant::FloatToBf16(dmin));
  for (std::size_t g = 0; g < kGroups; ++g) {
    std::uint8_t sc;
    std::uint8_t mn;
    quant::ScaleMinK4(g, scales, &sc, &mn);  // 6 bits each
    SetSub(block, m, g, sc, static_cast<std::int8_t>(-mn));
    // Sub-blocks pair up: 2p in the low nibbles of qs[32p..], 2p+1 high.
    const std::uint8_t* q = qs + (g / 2) * 32;
    const int shift = (g % 2) * 4;
    for (std::size_t l = 0; l < 32; ++l) {
      SetCode(block, m, g * 32 + l, (q[l] >> shift) & 0xF);
    }
  }
}

// Row m of a block from its kKTile fp32 weights: per group a min/max fit,
// with the minimum rounded down onto dmin's grid (int8) and the step rounded
// up onto d's (uint8), so every weight is within the group's 16 levels.
void PackRowBlock(const float* w, std::uint8_t* block, std::size_t m) {
  float lo[kGroups];
  float hi[kGroups];
  float max_abs_min = 0;
  for (std::size_t g = 0; g < kGroups; ++g) {
    const auto [lo_it, hi_it] =
        std::minmax_element(w + g * kGroup, w + (g + 1) * kGroup);
    lo[g] = *lo_it;
    hi[g] = *hi_it;
    max_abs_min = std::max(max_abs_min, std::abs(lo[g]));
  }
  const std::uint16_t dmin_bits = Bf16AtLeast(max_abs_min / 127.0f);
  const float dmin = quant::Bf16ToFloat(dmin_bits);
  std::int8_t mn[kGroups];
  float need[kGroups];  // the step each group needs from its minimum
  float max_need = 0;
  for (std::size_t g = 0; g < kGroups; ++g) {
    const float q = dmin > 0 ? std::floor(lo[g] / dmin) : 0.0f;
    mn[g] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
    need[g] = (hi[g] - dmin * static_cast<float>(mn[g])) / 15.0f;
    max_need = std::max(max_need, need[g]);
  }
  const std::uint16_t d_bits = Bf16AtLeast(max_need / 255.0f);
  const float d = quant::Bf16ToFloat(d_bits);
  SetSuper(block, m, d_bits, dmin_bits);
  for (std::size_t g = 0; g < kGroups; ++g) {
    const float sc = d > 0 ? std::ceil(need[g] / d) : 0.0f;
    SetSub(block, m, g, static_cast<std::uint8_t>(std::clamp(sc, 0.0f, 255.0f)),
           mn[g]);
    float step;
    float min;
    Params(block, m, g, &step, &min);
    for (std::size_t l = 0; l < kGroup; ++l) {
      float code =
          step > 0 ? std::nearbyint((w[g * kGroup + l] - min) / step) : 0.0f;
      code = std::clamp(code, 0.0f, 15.0f);
      SetCode(block, m, g * kGroup + l, static_cast<std::uint8_t>(code));
    }
  }
}

// Packs every row of `out`. row_fn(r, buffer) returns row r's Q4_K data when
// it can be copied code for code, or fills `buffer` with its fp32 weights and
// returns nullptr. Tiles run in parallel; the rows of a tile share code bytes,
// so one thread packs a whole tile.
template<typename RowFn>
void PackTiles(Matrix& out, RowFn&& row_fn) {
  const std::size_t tiles = (out.rows + kMTile - 1) / kMTile;
  ParallelFor(tiles, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(out.dim);
    for (std::size_t tile = begin; tile < end; ++tile) {
      for (std::size_t m = 0; m < kMTile; ++m) {
        const std::size_t r = tile * kMTile + m;
        if (r >= out.rows) {
          break;
        }
        const std::uint8_t* q4k = row_fn(r, row.data());
        for (std::size_t kb = 0; kb < out.dim / kKTile; ++kb) {
          std::uint8_t* block = Block(out, tile, kb);
          if (q4k != nullptr) {
            PackQ4KSuper(q4k + kb * kQ4KBytes, block, m);
            continue;
          }
          PackRowBlock(row.data() + kb * kKTile, block, m);
        }
      }
    }
  });
}

}  // namespace

std::size_t PaddedRows(std::size_t rows) {
  return (rows + kOutPerRound - 1) / kOutPerRound * kOutPerRound;
}

Matrix PackRows(const std::vector<Segment>& segments, std::size_t dim,
                const std::vector<std::size_t>& order) {
  std::size_t rows = 0;
  bool exact = true;
  for (const auto& seg : segments) {
    rows += seg.rows;
    exact = exact && seg.type == gguf::GgmlType::kQ4_K;
  }
  Matrix out = Allocate(rows, dim);
  out.exact = exact;
  // First row of each segment.
  std::vector<std::size_t> starts;
  std::size_t start = 0;
  for (const auto& seg : segments) {
    starts.push_back(start);
    start += seg.rows;
  }
  if (!order.empty() && order.size() != rows) {
    throw std::invalid_argument("q4nx: row order does not cover every row");
  }
  PackTiles(out, [&](std::size_t packed, float* row) -> const std::uint8_t* {
    const std::size_t r = order.empty() ? packed : order[packed];
    const auto it = std::upper_bound(starts.begin(), starts.end(), r) - 1;
    const auto& seg = segments[static_cast<std::size_t>(it - starts.begin())];
    const std::uint8_t* src =
        seg.data + (r - *it) * quant::RowBytes(seg.type, dim);
    if (seg.type == gguf::GgmlType::kQ4_K) {
      return src;
    }
    quant::Dequantize(seg.type, src, row, dim);
    return nullptr;
  });
  return out;
}

std::vector<std::size_t> SwiGluOrder(std::size_t ffn) {
  constexpr std::size_t kHalf = kMTile / 2;
  if (ffn % kHalf != 0) {
    throw std::invalid_argument("q4nx: SwiGLU width must be a multiple of 16");
  }
  std::vector<std::size_t> order(2 * ffn);
  for (std::size_t n = 0; n < order.size(); ++n) {
    const std::size_t tile = n / kMTile;
    const std::size_t m = n % kMTile;
    order[n] = (m < kHalf ? 0 : ffn) + tile * kHalf + m % kHalf;
  }
  return order;
}

Matrix Pack(gguf::GgmlType type, const std::uint8_t* data, std::size_t rows,
            std::size_t dim) {
  return PackRows({{type, data, rows}}, dim);
}

Matrix PackFloat(const float* weights, std::size_t rows, std::size_t dim) {
  Matrix out = Allocate(rows, dim);
  PackTiles(out, [&](std::size_t r, float* row) -> const std::uint8_t* {
    std::memcpy(row, weights + r * dim, dim * sizeof(float));
    return nullptr;
  });
  return out;
}

std::vector<float> Unpack(const Matrix& matrix) {
  std::vector<float> w(matrix.padded_rows * matrix.dim);
  const std::size_t tiles = matrix.padded_rows / kMTile;
  ParallelFor(tiles, [&](std::size_t begin, std::size_t end) {
    for (std::size_t tile = begin; tile < end; ++tile) {
      for (std::size_t kb = 0; kb < matrix.dim / kKTile; ++kb) {
        const std::uint8_t* block = Block(matrix, tile, kb);
        for (std::size_t m = 0; m < kMTile; ++m) {
          float* row =
              w.data() + (tile * kMTile + m) * matrix.dim + kb * kKTile;
          for (std::size_t k = 0; k < kKTile; ++k) {
            float step;
            float min;
            Params(block, m, k / kGroup, &step, &min);
            const std::size_t i = CodeIndex(m, k);
            const std::uint8_t byte = block[kCodes + i / 2];
            const int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            row[k] = min + step * static_cast<float>(code);
          }
        }
      }
    }
  });
  return w;
}

void Gemv(const Matrix& matrix, const float* x, float* y) {
  const std::size_t tiles = (matrix.rows + kMTile - 1) / kMTile;
  const std::size_t k_blocks = matrix.dim / kKTile;
  // Per-32 sums of x fold each group's minimum into one multiply.
  std::vector<double> sums(matrix.dim / kGroup);
  for (std::size_t g = 0; g < sums.size(); ++g) {
    for (std::size_t l = 0; l < kGroup; ++l) {
      sums[g] += x[g * kGroup + l];
    }
  }
  ParallelFor(tiles, [&](std::size_t begin, std::size_t end) {
    for (std::size_t tile = begin; tile < end; ++tile) {
      double acc[kMTile] = {};
      for (std::size_t kb = 0; kb < k_blocks; ++kb) {
        const std::uint8_t* block = Block(matrix, tile, kb);
        const std::uint8_t* codes = block + kCodes;
        const float* xk = x + kb * kKTile;
        for (std::size_t g = 0; g < kGroups; ++g) {
          // Codes of group g: [m / 16][g][l][m % 16], two per byte.
          for (std::size_t half = 0; half < kMTile / 16; ++half) {
            double dot[16] = {};
            const std::uint8_t* q = codes + ((half * kGroups + g) * 32) * 8;
            for (std::size_t l = 0; l < 32; ++l) {
              const double xv = xk[g * 32 + l];
              for (std::size_t i = 0; i < 8; ++i) {
                const std::uint8_t byte = q[l * 8 + i];
                dot[2 * i] += (byte & 0xF) * xv;
                dot[2 * i + 1] += (byte >> 4) * xv;
              }
            }
            for (std::size_t i = 0; i < 16; ++i) {
              const std::size_t m = half * 16 + i;
              float step;
              float min;
              Params(block, m, g, &step, &min);
              acc[m] += step * dot[i] + min * sums[kb * kGroups + g];
            }
          }
        }
      }
      for (std::size_t m = 0; m < kMTile; ++m) {
        const std::size_t r = tile * kMTile + m;
        if (r < matrix.rows) {
          y[r] = static_cast<float>(acc[m]);
        }
      }
    }
  });
}

std::vector<double> Reference(const std::vector<float>& unpacked,
                              const Matrix& matrix, const float* x) {
  std::vector<double> y(matrix.padded_rows);
  ParallelFor(matrix.padded_rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      const float* row = unpacked.data() + r * matrix.dim;
      double acc = 0;
      for (std::size_t k = 0; k < matrix.dim; ++k) {
        acc += static_cast<double>(row[k]) * x[k];
      }
      y[r] = acc;
    }
  });
  return y;
}

}  // namespace gufo_npu::q4nx
