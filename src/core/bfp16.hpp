#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// bfp16ebs8, the AIE2P block-floating-point type the prefill GEMM's mmul
// multiplies (kernels/designs/bfp16_gemm.py): 8 values share an 8-bit
// exponent byte and carry 8-bit two's-complement mantissas, 9 bytes per
// block, value = mantissa * 2^(exponent - 133).
//
// PackGemmB lays a weight matrix out as the GEMM's B operand in the order its
// cores read it (IRON's iron.operators.flm.packing.pack_b with k_tile 512,
// n_tile 64, ct_k 128, s = t = 8), so its transfers are linear.
namespace gufo_npu::bfp16 {

inline constexpr std::size_t kBlock = 8;       // values per block
inline constexpr std::size_t kBlockBytes = 9;  // exponent, then 8 mantissas
// The GEMM's B tiling: K tiles of kKTile, held kCtK at a time; N tiles of
// kNTile; 8 x 8 mmul blocks.
inline constexpr std::size_t kKTile = 512;
inline constexpr std::size_t kNTile = 64;
inline constexpr std::size_t kCtK = 128;

// 8 floats into one block: the block's largest exponent, each mantissa
// rounded to nearest (ties to even) as a core in conv_even mode converts,
// saturating at +127 (IRON's f32_to_bfp16ebs8, byte for byte).
void EncodeBlock(const float* in, std::uint8_t* out);
void DecodeBlock(const std::uint8_t* in, float* out);

// Bytes PackGemmB returns for `rows` x `dim`.
std::size_t PackedBytes(std::size_t rows, std::size_t dim);

// W (`rows` out-features x `dim` in-features, row-major) as the B operand
// (dim x rows) of the prefill GEMM. Each block is 8 consecutive in-features
// of one out-feature. `rows` must be a multiple of kNTile and `dim` of
// kKTile.
std::vector<std::uint8_t> PackGemmB(const float* w, std::size_t rows,
                                    std::size_t dim);

// The weights the GEMM multiplies, row-major rows x dim (the inverse of
// PackGemmB's layout, decoded).
std::vector<float> UnpackGemmB(const std::vector<std::uint8_t>& packed,
                               std::size_t rows, std::size_t dim);

}  // namespace gufo_npu::bfp16
