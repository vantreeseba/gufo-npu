#pragma once

#include <cstddef>
#include <cstdint>

#include "core/gguf.hpp"

// CPU dequantization of ggml block formats, bit-exact with gguf-py.
namespace gufo_npu::quant {

// Elements per block (1 for plain float types).
std::size_t BlockElements(gguf::GgmlType type);
// Bytes per block.
std::size_t BlockBytes(gguf::GgmlType type);
// Bytes for `n` elements; `n` must be a multiple of BlockElements.
std::size_t RowBytes(gguf::GgmlType type, std::size_t n);

// Converts `n` elements starting at `src` (a block boundary) to fp32.
void Dequantize(gguf::GgmlType type, const std::uint8_t* src, float* dst,
                std::size_t n);

// 6-bit scale and min of sub-block j (0..7) of a Q4_K/Q5_K super-block, from
// its 12 packed scale bytes.
void ScaleMinK4(std::size_t j, const std::uint8_t* scales, std::uint8_t* scale,
                std::uint8_t* min);

float HalfToFloat(std::uint16_t h);
float Bf16ToFloat(std::uint16_t h);
// Rounds to nearest, ties to even.
std::uint16_t FloatToBf16(float f);

}  // namespace gufo_npu::quant
