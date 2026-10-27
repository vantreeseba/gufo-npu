#include "core/quant.hpp"

#include <cstring>
#include <stdexcept>

// Block layouts follow ggml (ggml-common.h) and gguf-py's quants.py.
namespace gufo_npu::quant {
namespace {

using gguf::GgmlType;

constexpr std::size_t kQK = 256;  // K-quant super-block
constexpr std::size_t kQ8 = 32;   // Q8_0 block
constexpr std::size_t kScaleBytes = 12;

std::uint16_t LoadU16(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// 6-bit scale and min for sub-block j of a Q4_K/Q5_K super-block.
void ScaleMinK4(std::size_t j, const std::uint8_t* q, std::uint8_t* scale,
                std::uint8_t* min) {
  if (j < 4) {
    *scale = q[j] & 63;
    *min = q[j + 4] & 63;
  } else {
    *scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void DequantQ8_0(const std::uint8_t* src, float* dst, std::size_t blocks) {
  for (std::size_t b = 0; b < blocks; ++b, src += 2 + kQ8, dst += kQ8) {
    const float d = HalfToFloat(LoadU16(src));
    const auto* qs = reinterpret_cast<const std::int8_t*>(src + 2);
    for (std::size_t l = 0; l < kQ8; ++l) {
      dst[l] = d * static_cast<float>(qs[l]);
    }
  }
}

// block_q4_K: d, dmin (f16), scales[12], qs[128]
void DequantQ4_K(const std::uint8_t* src, float* dst, std::size_t blocks) {
  for (std::size_t b = 0; b < blocks; ++b, src += 144) {
    const float d = HalfToFloat(LoadU16(src));
    const float dmin = HalfToFloat(LoadU16(src + 2));
    const std::uint8_t* scales = src + 4;
    const std::uint8_t* q = scales + kScaleBytes;
    std::size_t is = 0;
    for (std::size_t j = 0; j < kQK; j += 64) {
      std::uint8_t sc;
      std::uint8_t m;
      ScaleMinK4(is, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      ScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (std::size_t l = 0; l < 32; ++l) {
        *dst++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
      }
      for (std::size_t l = 0; l < 32; ++l) {
        *dst++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      }
      q += 32;
      is += 2;
    }
  }
}

// block_q5_K: d, dmin (f16), scales[12], qh[32], qs[128]
void DequantQ5_K(const std::uint8_t* src, float* dst, std::size_t blocks) {
  for (std::size_t b = 0; b < blocks; ++b, src += 176) {
    const float d = HalfToFloat(LoadU16(src));
    const float dmin = HalfToFloat(LoadU16(src + 2));
    const std::uint8_t* scales = src + 4;
    const std::uint8_t* qh = scales + kScaleBytes;
    const std::uint8_t* ql = qh + 32;
    std::size_t is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (std::size_t j = 0; j < kQK; j += 64) {
      std::uint8_t sc;
      std::uint8_t m;
      ScaleMinK4(is, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      ScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (std::size_t l = 0; l < 32; ++l) {
        const int v = (ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0);
        *dst++ = d1 * static_cast<float>(v) - m1;
      }
      for (std::size_t l = 0; l < 32; ++l) {
        const int v = (ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
        *dst++ = d2 * static_cast<float>(v) - m2;
      }
      ql += 32;
      is += 2;
      u1 = static_cast<std::uint8_t>(u1 << 2);
      u2 = static_cast<std::uint8_t>(u2 << 2);
    }
  }
}

// block_q6_K: ql[128], qh[64], scales[16] (int8), d (f16)
void DequantQ6_K(const std::uint8_t* src, float* dst, std::size_t blocks) {
  for (std::size_t b = 0; b < blocks; ++b, src += 210) {
    const std::uint8_t* ql = src;
    const std::uint8_t* qh = src + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(src + 192);
    const float d = HalfToFloat(LoadU16(src + 208));
    for (std::size_t n = 0; n < kQK; n += 128) {
      for (std::size_t l = 0; l < 32; ++l) {
        const std::size_t is = l / 16;
        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        dst[l] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        dst[l + 32] =
            d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        dst[l + 64] =
            d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        dst[l + 96] =
            d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      dst += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

}  // namespace

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  std::uint32_t exponent = (h >> 10) & 0x1FU;
  std::uint32_t mantissa = h & 0x3FFU;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFU;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::size_t BlockElements(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return 1;
    case GgmlType::kQ8_0:
      return kQ8;
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
      return kQK;
  }
  throw std::invalid_argument("unknown ggml type");
}

std::size_t BlockBytes(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return 2;
    case GgmlType::kQ8_0:
      return 34;
    case GgmlType::kQ4_K:
      return 144;
    case GgmlType::kQ5_K:
      return 176;
    case GgmlType::kQ6_K:
      return 210;
  }
  throw std::invalid_argument("unknown ggml type");
}

std::size_t RowBytes(GgmlType type, std::size_t n) {
  const auto block = BlockElements(type);
  if (n % block != 0) {
    throw std::invalid_argument("row is not a whole number of blocks");
  }
  return n / block * BlockBytes(type);
}

void Dequantize(GgmlType type, const std::uint8_t* src, float* dst,
                std::size_t n) {
  const std::size_t blocks = n / BlockElements(type);
  if (blocks * BlockElements(type) != n) {
    throw std::invalid_argument("dequantize: partial block");
  }
  switch (type) {
    case GgmlType::kF32:
      std::memcpy(dst, src, n * sizeof(float));
      return;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < n; ++i) {
        dst[i] = HalfToFloat(LoadU16(src + 2 * i));
      }
      return;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < n; ++i) {
        dst[i] = Bf16ToFloat(LoadU16(src + 2 * i));
      }
      return;
    case GgmlType::kQ8_0:
      DequantQ8_0(src, dst, blocks);
      return;
    case GgmlType::kQ4_K:
      DequantQ4_K(src, dst, blocks);
      return;
    case GgmlType::kQ5_K:
      DequantQ5_K(src, dst, blocks);
      return;
    case GgmlType::kQ6_K:
      DequantQ6_K(src, dst, blocks);
      return;
  }
  throw std::invalid_argument("unknown ggml type");
}

}  // namespace gufo_npu::quant
