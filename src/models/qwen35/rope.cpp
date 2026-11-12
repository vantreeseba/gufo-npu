#include "models/qwen35/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gufo_npu::qwen35 {

std::vector<float> RopeInvFreq(const Config& c) {
  std::vector<float> inv_freq(c.rope_dim / 2);
  for (int i = 0; i < c.rope_dim / 2; ++i) {
    inv_freq[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(c.rope_theta),
                       2.0 * i / static_cast<double>(c.rope_dim)));
  }
  return inv_freq;
}

int MropeAxis(const std::array<int, 3>& sections, int i) {
  if (i % 3 == 1 && i < 3 * sections[1]) {
    return 1;
  }
  if (i % 3 == 2 && i < 3 * sections[2]) {
    return 2;
  }
  return 0;
}

void RopeAngles(const Config& c, const std::vector<float>& inv_freq,
                RopePosition p, float* cos, float* sin) {
  const int axes[3] = {p.t, p.h, p.w};
  for (int i = 0; i < c.rope_dim / 2; ++i) {
    const int pos = axes[MropeAxis(c.rope_sections, i)];
    const double angle = static_cast<double>(pos) * inv_freq[i];
    cos[i] = static_cast<float>(std::cos(angle));
    sin[i] = static_cast<float>(std::sin(angle));
  }
}

int RopeDeltaAfter(const std::vector<RopePosition>& positions, int end) {
  if (positions.empty()) {
    return 0;
  }
  int top = positions.front().t;
  for (const auto& p : positions) {
    top = std::max({top, p.t, p.h, p.w});
  }
  return top + 1 - end;
}

PromptPositions ImagePromptPositions(const std::vector<int>& tokens,
                                     int image_token,
                                     const std::vector<ImageGrid>& grids,
                                     int merge) {
  PromptPositions out;
  out.positions.reserve(tokens.size());
  std::size_t image = 0;
  int pos = 0;
  for (std::size_t i = 0; i < tokens.size();) {
    if (tokens[i] != image_token) {
      out.positions.push_back(TextPosition(pos++));
      ++i;
      continue;
    }
    std::size_t run = 0;
    while (i + run < tokens.size() && tokens[i + run] == image_token) {
      ++run;
    }
    if (image >= grids.size()) {
      throw std::invalid_argument("qwen35: more image token runs than grids");
    }
    const ImageGrid& g = grids[image++];
    if (merge <= 0 || g.t <= 0 || g.h <= 0 || g.w <= 0 || g.h % merge != 0 ||
        g.w % merge != 0) {
      throw std::invalid_argument("qwen35: bad image grid");
    }
    const int gh = g.h / merge;
    const int gw = g.w / merge;
    if (run != static_cast<std::size_t>(g.t) * gh * gw) {
      throw std::invalid_argument(
          "qwen35: an image token run does not match its grid");
    }
    for (int f = 0; f < g.t; ++f) {
      for (int r = 0; r < gh; ++r) {
        for (int col = 0; col < gw; ++col) {
          out.positions.push_back({pos + f, pos + r, pos + col});
        }
      }
    }
    pos += std::max(gh, gw);
    i += run;
  }
  if (image != grids.size()) {
    throw std::invalid_argument("qwen35: fewer image token runs than grids");
  }
  out.rope_delta =
      RopeDeltaAfter(out.positions, static_cast<int>(tokens.size()));
  return out;
}

}  // namespace gufo_npu::qwen35
