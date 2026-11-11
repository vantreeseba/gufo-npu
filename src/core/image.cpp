#include "core/image.hpp"

#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// jpeglib.h needs <cstdio> first.
#include <jpeglib.h>
#include <png.h>

// libpng and libjpeg leave errors through longjmp. The functions that call
// setjmp (ReadPng, ReadJpeg) keep only plain locals, fill vectors their
// caller owns, and report failure by returning false with the message in
// their source's buffer; the caller then frees the decoder and throws.
namespace gufo_npu::image {
namespace {

constexpr std::uint8_t kPngSignature[8] = {0x89, 'P',  'N',  'G',
                                           '\r', '\n', 0x1a, '\n'};
// Progressive JPEGs with more scans are refused: each scan reprocesses the
// whole image, which a crafted file can repeat thousands of times.
constexpr int kMaxJpegScans = 500;

constexpr std::size_t kMessageSize = 200;

// Writes the "too many pixels" message; false when within the limit.
bool OverLimit(long width, long height, long max_pixels, char* message) {
  if (width * height <= max_pixels) {
    return false;
  }
  std::snprintf(message, kMessageSize,
                "the image is %ld x %ld pixels, above the limit of %ld", width,
                height, max_pixels);
  return true;
}

// ---- PNG ----

struct PngSource {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::size_t offset = 0;
  std::jmp_buf jump;
  char error[kMessageSize] = {};
};

void OnPngError(png_structp png, png_const_charp message) {
  auto* source = static_cast<PngSource*>(png_get_error_ptr(png));
  std::snprintf(source->error, sizeof source->error, "%s", message);
  std::longjmp(source->jump, 1);
}

void OnPngWarning(png_structp, png_const_charp) {}

void OnPngRead(png_structp png, png_bytep out, png_size_t count) {
  auto* source = static_cast<PngSource*>(png_get_io_ptr(png));
  if (count > source->size - source->offset) {
    png_error(png, "the file is truncated");
  }
  std::memcpy(out, source->data + source->offset, count);
  source->offset += count;
}

struct PngHeader {
  long width = 0;
  long height = 0;
  int bit_depth = 0;
  int color_type = 0;
};

// The header, then the samples, a byte each below 8 bits and big-endian
// pairs at 16, rows de-interlaced.
bool ReadPng(png_structp png, png_infop info, PngSource& source,
             long max_pixels, PngHeader& header,
             std::vector<std::uint8_t>& samples, std::vector<png_bytep>& rows) {
  if (setjmp(source.jump) != 0) {
    return false;
  }
  png_set_read_fn(png, &source, OnPngRead);
  png_read_info(png, info);
  header.width = png_get_image_width(png, info);
  header.height = png_get_image_height(png, info);
  header.bit_depth = png_get_bit_depth(png, info);
  header.color_type = png_get_color_type(png, info);
  if (OverLimit(header.width, header.height, max_pixels, source.error)) {
    return false;
  }
  png_set_packing(png);
  png_set_interlace_handling(png);
  png_read_update_info(png, info);
  const std::size_t row_bytes = png_get_rowbytes(png, info);
  samples.resize(row_bytes * static_cast<std::size_t>(header.height));
  rows.resize(static_cast<std::size_t>(header.height));
  for (std::size_t y = 0; y < rows.size(); ++y) {
    rows[y] = samples.data() + y * row_bytes;
  }
  png_read_image(png, rows.data());
  return true;
}

int PngChannels(int color_type) {
  switch (color_type) {
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      return 2;
    case PNG_COLOR_TYPE_RGB:
      return 3;
    case PNG_COLOR_TYPE_RGB_ALPHA:
      return 4;
    default:  // gray, palette
      return 1;
  }
}

// The samples as Pillow converts them (see image.hpp).
Decoded ConvertPng(const PngHeader& header,
                   const std::vector<std::uint8_t>& samples,
                   png_const_colorp palette, int palette_size) {
  const int depth = header.bit_depth;
  const int channels = PngChannels(header.color_type);
  const int sample_bytes = depth == 16 ? 2 : 1;
  const bool keep_alpha =
      header.color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
      (header.color_type == PNG_COLOR_TYPE_GRAY_ALPHA && depth == 16);
  const std::size_t pixels = static_cast<std::size_t>(header.width) *
                             static_cast<std::size_t>(header.height);
  Decoded out;
  out.width = static_cast<int>(header.width);
  out.height = static_cast<int>(header.height);
  out.rgb.resize(pixels * 3);
  if (keep_alpha) {
    out.alpha.resize(pixels);
  }
  // A sample cut to 8 bits: the high byte at 16 bits.
  const auto sample = [&](const std::uint8_t* px, int c) -> std::uint8_t {
    return px[c * sample_bytes];
  };
  const int gray_scale = depth < 8 ? 255 / ((1 << depth) - 1) : 1;
  for (std::size_t i = 0; i < pixels; ++i) {
    const std::uint8_t* px =
        samples.data() + i * static_cast<std::size_t>(channels * sample_bytes);
    std::uint8_t* rgb = out.rgb.data() + i * 3;
    switch (header.color_type) {
      case PNG_COLOR_TYPE_GRAY:
      case PNG_COLOR_TYPE_GRAY_ALPHA: {
        std::uint8_t v = 0;
        if (header.color_type == PNG_COLOR_TYPE_GRAY && depth == 16) {
          v = static_cast<std::uint8_t>(std::min(px[0] << 8 | px[1], 255));
        } else {
          v = static_cast<std::uint8_t>(sample(px, 0) * gray_scale);
        }
        rgb[0] = rgb[1] = rgb[2] = v;
        if (keep_alpha) {
          out.alpha[i] = sample(px, 1);
        }
        break;
      }
      case PNG_COLOR_TYPE_PALETTE: {
        const int index = px[0];
        if (index < palette_size) {
          rgb[0] = palette[index].red;
          rgb[1] = palette[index].green;
          rgb[2] = palette[index].blue;
        } else {
          rgb[0] = rgb[1] = rgb[2] = 0;
        }
        break;
      }
      default:  // RGB, RGB + alpha
        for (int c = 0; c < 3; ++c) {
          rgb[c] = sample(px, c);
        }
        if (keep_alpha) {
          out.alpha[i] = sample(px, 3);
        }
        break;
    }
  }
  return out;
}

Decoded DecodePng(std::span<const std::uint8_t> bytes, long max_pixels) {
  PngSource source;
  source.data = bytes.data();
  source.size = bytes.size();
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &source,
                                           OnPngError, OnPngWarning);
  if (png == nullptr) {
    throw std::bad_alloc();
  }
  png_infop info = png_create_info_struct(png);
  if (info == nullptr) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    throw std::bad_alloc();
  }
  struct Guard {
    png_structp* png;
    png_infop* info;
    ~Guard() { png_destroy_read_struct(png, info, nullptr); }
  } guard{&png, &info};
  PngHeader header;
  std::vector<std::uint8_t> samples;
  std::vector<png_bytep> rows;
  if (!ReadPng(png, info, source, max_pixels, header, samples, rows)) {
    throw std::invalid_argument(std::string("invalid PNG: ") + source.error);
  }
  png_colorp palette = nullptr;
  int palette_size = 0;
  if (header.color_type == PNG_COLOR_TYPE_PALETTE) {
    png_get_PLTE(png, info, &palette, &palette_size);
  }
  return ConvertPng(header, samples, palette, palette_size);
}

// ---- JPEG ----

struct JpegErrors {
  jpeg_error_mgr manager;  // first: libjpeg's err pointer is to it
  std::jmp_buf jump;
  char error[std::max<std::size_t>(JMSG_LENGTH_MAX, kMessageSize)] = {};
};

void OnJpegError(j_common_ptr cinfo) {
  auto* errors = reinterpret_cast<JpegErrors*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, errors->error);
  std::longjmp(errors->jump, 1);
}

// Warnings pass, as Pillow lets them pass, except running out of data,
// which Pillow reports as a truncated file.
void OnJpegMessage(j_common_ptr cinfo, int level) {
  if (level < 0 && cinfo->err->msg_code == JWRN_JPEG_EOF) {
    (*cinfo->err->error_exit)(cinfo);
  }
}

void OnJpegProgress(j_common_ptr cinfo) {
  const auto* decompress = reinterpret_cast<j_decompress_ptr>(cinfo);
  if (decompress->input_scan_number > kMaxJpegScans) {
    auto* errors = reinterpret_cast<JpegErrors*>(cinfo->err);
    std::snprintf(errors->error, sizeof errors->error,
                  "more than %d progressive scans", kMaxJpegScans);
    std::longjmp(errors->jump, 1);
  }
}

bool ReadJpeg(jpeg_decompress_struct& cinfo, JpegErrors& errors,
              jpeg_progress_mgr& progress, std::span<const std::uint8_t> bytes,
              long max_pixels, Decoded& out, std::vector<std::uint8_t>& row) {
  if (setjmp(errors.jump) != 0) {
    return false;
  }
  jpeg_create_decompress(&cinfo);
  cinfo.progress = &progress;
  // Older jpeglib.h declare the buffer non-const; it is only read.
  jpeg_mem_src(&cinfo, const_cast<std::uint8_t*>(bytes.data()),
               static_cast<unsigned long>(bytes.size()));
  jpeg_read_header(&cinfo, TRUE);
  if (cinfo.data_precision != 8) {
    std::snprintf(errors.error, sizeof errors.error,
                  "%d-bit samples are not supported", cinfo.data_precision);
    return false;
  }
  switch (cinfo.jpeg_color_space) {
    case JCS_GRAYSCALE:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
    case JCS_YCbCr:
    case JCS_RGB:
      cinfo.out_color_space = JCS_RGB;
      break;
    default:
      std::snprintf(errors.error, sizeof errors.error,
                    "CMYK and YCCK images are not supported");
      return false;
  }
  if (OverLimit(cinfo.image_width, cinfo.image_height, max_pixels,
                errors.error)) {
    return false;
  }
  jpeg_start_decompress(&cinfo);
  const std::size_t width = cinfo.output_width;
  out.width = static_cast<int>(cinfo.output_width);
  out.height = static_cast<int>(cinfo.output_height);
  out.rgb.resize(width * cinfo.output_height * 3);
  row.resize(width);
  while (cinfo.output_scanline < cinfo.output_height) {
    std::uint8_t* line = out.rgb.data() + cinfo.output_scanline * width * 3;
    if (cinfo.out_color_space == JCS_GRAYSCALE) {
      JSAMPROW target = row.data();
      jpeg_read_scanlines(&cinfo, &target, 1);
      for (std::size_t x = 0; x < width; ++x) {
        line[x * 3] = line[x * 3 + 1] = line[x * 3 + 2] = row[x];
      }
    } else {
      JSAMPROW target = line;
      jpeg_read_scanlines(&cinfo, &target, 1);
    }
  }
  jpeg_finish_decompress(&cinfo);
  return true;
}

Decoded DecodeJpeg(std::span<const std::uint8_t> bytes, long max_pixels) {
  jpeg_decompress_struct cinfo{};
  JpegErrors errors{};
  cinfo.err = jpeg_std_error(&errors.manager);
  errors.manager.error_exit = OnJpegError;
  errors.manager.emit_message = OnJpegMessage;
  jpeg_progress_mgr progress{};
  progress.progress_monitor = OnJpegProgress;
  struct Guard {
    jpeg_decompress_struct* cinfo;
    ~Guard() { jpeg_destroy_decompress(cinfo); }
  } guard{&cinfo};
  Decoded out;
  std::vector<std::uint8_t> row;
  if (!ReadJpeg(cinfo, errors, progress, bytes, max_pixels, out, row)) {
    throw std::invalid_argument(std::string("invalid JPEG: ") + errors.error);
  }
  return out;
}

}  // namespace

Decoded Decode(std::span<const std::uint8_t> bytes, long max_pixels) {
  if (bytes.size() >= sizeof kPngSignature &&
      std::memcmp(bytes.data(), kPngSignature, sizeof kPngSignature) == 0) {
    return DecodePng(bytes, max_pixels);
  }
  if (bytes.size() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 &&
      bytes[2] == 0xFF) {
    return DecodeJpeg(bytes, max_pixels);
  }
  throw std::invalid_argument("unsupported image format: expected PNG or JPEG");
}

}  // namespace gufo_npu::image
