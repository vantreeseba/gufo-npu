#include "core/image.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Decodes the PNG and JPEG fixtures and compares byte for byte with Pillow's
// decode of the same files (tests/fixtures/image, from
// tools/reference/image_fixtures.py; expected.txt lists the cases).
namespace {

namespace image = gufo_npu::image;

std::vector<std::uint8_t> ReadAll(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

int Compare(const std::string& name, const char* what,
            const std::vector<std::uint8_t>& actual,
            const std::vector<std::uint8_t>& expected) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr, "FAIL: %s %s: %zu bytes, Pillow %zu\n", name.c_str(),
                 what, actual.size(), expected.size());
    return 1;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL: %s %s[%zu]: %d, Pillow %d\n", name.c_str(),
                   what, i, actual[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

int CheckCase(const std::filesystem::path& dir, const std::string& line) {
  std::istringstream fields(line);
  std::string name;
  std::string first;
  fields >> name >> first;
  const auto bytes = ReadAll(dir / name);
  if (bytes.empty()) {
    std::fprintf(stderr, "FAIL: %s missing in %s\n", name.c_str(), dir.c_str());
    return 1;
  }
  if (first == "error") {
    try {
      image::Decode(bytes);
    } catch (const std::invalid_argument& e) {
      std::printf("ok: %s rejected (%s)\n", name.c_str(), e.what());
      return 0;
    }
    std::fprintf(stderr, "FAIL: %s decoded; Pillow rejects it\n", name.c_str());
    return 1;
  }
  const int width = std::stoi(first);
  int height = 0;
  fields >> height;
  std::vector<std::string> outputs;
  for (std::string output; fields >> output;) {
    outputs.push_back(output);
  }
  const bool has_alpha = outputs.size() > 1;
  image::Decoded decoded;
  try {
    decoded = image::Decode(bytes);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s: %s\n", name.c_str(), e.what());
    return 1;
  }
  if (decoded.width != width || decoded.height != height ||
      decoded.HasAlpha() != has_alpha) {
    std::fprintf(stderr, "FAIL: %s: %d x %d%s, Pillow %d x %d%s\n",
                 name.c_str(), decoded.width, decoded.height,
                 decoded.HasAlpha() ? " with alpha" : "", width, height,
                 has_alpha ? " with alpha" : "");
    return 1;
  }
  const std::string stem = name.substr(0, name.rfind('.'));
  int failures =
      Compare(name, "rgb", decoded.rgb, ReadAll(dir / (stem + ".rgb")));
  if (has_alpha) {
    failures +=
        Compare(name, "alpha", decoded.alpha, ReadAll(dir / (stem + ".alpha")));
  }
  if (failures == 0) {
    std::printf("ok: %s\n", name.c_str());
  }
  return failures;
}

int CheckErrors(const std::filesystem::path& dir) {
  const auto rgb8 = ReadAll(dir / "rgb8.png");
  const auto jpeg = ReadAll(dir / "jpeg420.jpg");
  int failures = 0;
  const auto rejects = [&](const char* what, auto&& decode) {
    try {
      decode();
    } catch (const std::invalid_argument& e) {
      std::printf("ok: %s rejected (%s)\n", what, e.what());
      return;
    }
    std::fprintf(stderr, "FAIL: %s decoded\n", what);
    ++failures;
  };
  rejects("empty input", [] { image::Decode({}); });
  rejects("PNG over the pixel limit",
          [&] { image::Decode(rgb8, 37L * 29 - 1); });
  rejects("JPEG over the pixel limit",
          [&] { image::Decode(jpeg, 37L * 29 - 1); });
  try {
    image::Decode(rgb8, 37L * 29);
    std::puts("ok: PNG at the pixel limit");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: PNG at the pixel limit: %s\n", e.what());
    ++failures;
  }
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: image_test <fixture-dir>\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  std::ifstream expected(dir / "expected.txt");
  if (!expected) {
    std::fprintf(stderr, "FAIL: no expected.txt in %s\n", dir.c_str());
    return 1;
  }
  int failures = 0;
  int cases = 0;
  for (std::string line; std::getline(expected, line);) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    failures += CheckCase(dir, line);
    ++cases;
  }
  failures += CheckErrors(dir);
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("image_test: %d cases passed\n", cases);
  return 0;
}
