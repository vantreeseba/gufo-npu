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

constexpr std::size_t kGroups = kKTile / kGroup;           // per block row
constexpr std::size_t kScaleBytes = kGroups * kMTile * 2;  // scales or minima
constexpr std::size_t kQ4KBytes = 144;                     // block_q4_K

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

void SetParams(std::uint8_t* block, std::size_t m, std::size_t g,
               std::uint16_t scale, std::uint16_t min) {
  StoreU16(block + (g * kMTile + m) * 2, scale);
  StoreU16(block + kScaleBytes + (g * kMTile + m) * 2, min);
}

void SetCode(std::uint8_t* block, std::size_t m, std::size_t k,
             std::uint8_t code) {
  const std::size_t i = CodeIndex(m, k);
  std::uint8_t& byte = block[2 * kScaleBytes + i / 2];
  byte = (i % 2 == 0) ? static_cast<std::uint8_t>((byte & 0xF0) | code)
                      : static_cast<std::uint8_t>((byte & 0x0F) | (code << 4));
}

// One Q4_K super-block (256 in-features of out-feature m) into a q4nx block.
void PackQ4KSuper(const std::uint8_t* src, std::uint8_t* block, std::size_t m) {
  const float d = quant::HalfToFloat(LoadU16(src));
  const float dmin = quant::HalfToFloat(LoadU16(src + 2));
  const std::uint8_t* scales = src + 4;
  const std::uint8_t* qs = src + 16;
  for (std::size_t g = 0; g < kGroups; ++g) {
    std::uint8_t sc;
    std::uint8_t mn;
    quant::ScaleMinK4(g, scales, &sc, &mn);
    SetParams(block, m, g, quant::FloatToBf16(d * sc),
              quant::FloatToBf16(-dmin * mn));
    // Sub-blocks pair up: 2p in the low nibbles of qs[32p..], 2p+1 high.
    const std::uint8_t* q = qs + (g / 2) * 32;
    const int shift = (g % 2) * 4;
    for (std::size_t l = 0; l < 32; ++l) {
      SetCode(block, m, g * 32 + l, (q[l] >> shift) & 0xF);
    }
  }
}

// Min/max fit of 32 weights.
void PackGroup(const float* w, std::uint8_t* block, std::size_t m,
               std::size_t g) {
  const auto [lo_it, hi_it] = std::minmax_element(w, w + kGroup);
  const std::uint16_t min_bits = quant::FloatToBf16(*lo_it);
  const std::uint16_t scale_bits =
      quant::FloatToBf16((*hi_it - *lo_it) / 15.0f);
  SetParams(block, m, g, scale_bits, min_bits);
  const float min = quant::Bf16ToFloat(min_bits);
  const float scale = quant::Bf16ToFloat(scale_bits);
  for (std::size_t l = 0; l < kGroup; ++l) {
    float code = scale > 0 ? std::nearbyint((w[l] - min) / scale) : 0.0f;
    code = std::clamp(code, 0.0f, 15.0f);
    SetCode(block, m, g * kGroup + l, static_cast<std::uint8_t>(code));
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
          for (std::size_t g = 0; g < kGroups; ++g) {
            PackGroup(row.data() + kb * kKTile + g * kGroup, block, m, g);
          }
        }
      }
    }
  });
}

}  // namespace

std::size_t PaddedRows(std::size_t rows) {
  return (rows + kOutPerRound - 1) / kOutPerRound * kOutPerRound;
}

Matrix PackRows(const std::vector<Segment>& segments, std::size_t dim) {
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
  PackTiles(out, [&](std::size_t r, float* row) -> const std::uint8_t* {
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
            const std::size_t g = k / kGroup;
            const float scale =
                quant::Bf16ToFloat(LoadU16(block + (g * kMTile + m) * 2));
            const float min = quant::Bf16ToFloat(
                LoadU16(block + kScaleBytes + (g * kMTile + m) * 2));
            const std::size_t i = CodeIndex(m, k);
            const std::uint8_t byte = block[2 * kScaleBytes + i / 2];
            const int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            row[k] = min + scale * static_cast<float>(code);
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
        const std::uint8_t* codes = block + 2 * kScaleBytes;
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
              const float scale =
                  quant::Bf16ToFloat(LoadU16(block + (g * kMTile + m) * 2));
              const float min = quant::Bf16ToFloat(
                  LoadU16(block + kScaleBytes + (g * kMTile + m) * 2));
              acc[m] += scale * dot[i] + min * sums[kb * kGroups + g];
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
