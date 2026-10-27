#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/gguf.hpp"

// q4nx: the 4-bit weight format of the NPU GEMV (kernels/designs/q4nx_gemv.py).
//
// A block holds kMTile out-features by kKTile in-features (4736 bytes, 4.625
// bits per weight):
//   d     bf16 [m]               super-scale
//   dmin  bf16 [m]               super-minimum
//   sc    uint8 [k / 32][m]      sub-scale
//   mn    int8 [k / 32][m]       sub-minimum
//   codes 4 bits [m / 16][k / 32][k % 32][m % 16], low nibble first
// A weight is dmin * mn + d * sc * code. Q4_K maps onto it code for code
// (its 6-bit scales and minima become sc and -mn; d and dmin round to bf16).
//
// Blocks are ordered for the design's DMA: each of the array's kCols columns
// reads one contiguous slice per round, and each of its kRows cores owns
// kMTile out-features of it.
namespace gufo_npu::q4nx {

inline constexpr std::size_t kMTile = 32;
inline constexpr std::size_t kKTile = 256;
inline constexpr std::size_t kGroup = 32;
inline constexpr std::size_t kCols = 8;
inline constexpr std::size_t kRows = 4;
inline constexpr std::size_t kBlockBytes =
    kMTile * kKTile / 2 + kMTile * kKTile / kGroup * 2 + kMTile * 4;
// Out-features the whole array produces per round.
inline constexpr std::size_t kOutPerRound = kCols * kRows * kMTile;

struct Matrix {
  std::size_t rows = 0;         // out-features with weights
  std::size_t padded_rows = 0;  // rows rounded up to kOutPerRound
  std::size_t dim = 0;          // in-features, a multiple of kKTile
  // True when the codes, sub-scales and sub-minima came straight from Q4_K
  // (only d and dmin were rounded to bf16); false when the weights were
  // requantized.
  bool exact = false;
  std::vector<std::uint8_t> bytes;
};

std::size_t PaddedRows(std::size_t rows);

// Packs a GGUF weight of `rows` x `dim` (row-major, ggml block types).
// Q4_K keeps its codes; every other type is dequantized and requantized: per
// 32 in-features a min/max fit whose minimum and step round onto the block's
// d and dmin grid (step up, minimum down), so a weight is off by at most half
// a step (its group's range / 15 plus d) / 2. Padding rows are zero.
Matrix Pack(gguf::GgmlType type, const std::uint8_t* data, std::size_t rows,
            std::size_t dim);
// One GGUF weight in a fused matrix.
struct Segment {
  gguf::GgmlType type;
  const std::uint8_t* data;
  std::size_t rows;
};
// Stacks several weights with the same `dim` into one matrix, rows in order,
// so projections that share an input run as one GEMV. Exact when every
// segment is Q4_K.
// `order`, when given, picks the stacked row for each packed row.
Matrix PackRows(const std::vector<Segment>& segments, std::size_t dim,
                const std::vector<std::size_t>& order = {});

// The packed-row order for a [gate (ffn rows) | up (ffn rows)] stack that the
// GEMV's SwiGLU output stage expects: each kMTile tile holds 16 gate rows,
// then the same 16 up rows. The output's first 16 of every 32 values are
// then silu(gate) * up in gate-row order.
std::vector<std::size_t> SwiGluOrder(std::size_t ffn);
// Requantizes fp32 weights (row-major `rows` x `dim`).
Matrix PackFloat(const float* weights, std::size_t rows, std::size_t dim);

// The weights the NPU computes with, row-major padded_rows x dim.
std::vector<float> Unpack(const Matrix& matrix);

// y = W x on the CPU from the packed matrix, `rows` outputs: the weights the
// NPU uses, accumulated in double.
void Gemv(const Matrix& matrix, const float* x, float* y);

// y = W x in double, over the padded rows, from Unpack(matrix).
std::vector<double> Reference(const std::vector<float>& unpacked,
                              const Matrix& matrix, const float* x);

}  // namespace gufo_npu::q4nx
