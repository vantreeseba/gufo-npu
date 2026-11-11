#pragma once

#include <cstdint>
#include <span>
#include <vector>

// PNG and JPEG decoding for image inputs, to the pixels Pillow gives
// (Image.open(f).convert("RGB"), and the alpha channel of the images Pillow
// opens as RGBA), since model references preprocess with Pillow. The
// decoders are libpng and libjpeg-turbo, Pillow's own JPEG library.
//
// Per PNG type, as Pillow converts it: 1-, 2- and 4-bit gray scaled to
// 0..255; 16-bit samples cut to their high byte, except 16-bit gray, which
// Pillow clips to 255 (its I;16 mode); palettes looked up; gray and alpha
// at 8 bits loses the alpha (Pillow's LA), at 16 bits keeps it (Pillow
// opens it as RGBA). tRNS, gAMA and other colour chunks are ignored, as
// Pillow's convert("RGB") ignores them. JPEGs: 8-bit grayscale, YCbCr or
// RGB, baseline or progressive; EXIF orientation is not applied.
namespace gufo_npu::image {

// The default cap on width x height (64 Mi pixels), under Pillow's
// decompression-bomb limit of 178,956,970.
inline constexpr long kMaxPixels = 64L << 20;

struct Decoded {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;    // width x height RGB8, row-major
  std::vector<std::uint8_t> alpha;  // width x height, or empty: no alpha

  bool HasAlpha() const { return !alpha.empty(); }
};

// Decodes a whole PNG or JPEG file. Throws std::invalid_argument for
// another format, a truncated or corrupt file, an unsupported kind (CMYK
// or 12-bit JPEG) or more than max_pixels pixels.
Decoded Decode(std::span<const std::uint8_t> bytes,
               long max_pixels = kMaxPixels);

}  // namespace gufo_npu::image
