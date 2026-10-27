// Records Qwen3.6-35B-A3B's expert routing over a text prompt, for the MoE
// prefill spike (docs/research/NPU-MOE-PREFILL.md; tools/qwen36_35b_a3b/
// prefill_spike.cpp): the prompt's tokens run
// one at a time through NpuDecoder, and each layer's chosen experts are
// written as int16 [token][layer][experts_used] to the output file.
//
//   qwen36_route_dump <model.gguf> <kernel-dir> <prompt.txt> <out.bin> [max]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/npu_decoder.hpp"

namespace qwen35 = gufo_npu::qwen35;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: qwen36_route_dump <model.gguf> <kernel-dir> "
                 "<prompt.txt> <out.bin> [max-tokens]\n");
    return 2;
  }
  const auto file = gufo_npu::gguf::File::Open(argv[1]);
  std::ifstream in(argv[3]);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  const gufo_npu::Tokenizer tokenizer(*file);
  std::vector<int> ids = tokenizer.Encode(text);
  if (argc > 5) {
    ids.resize(std::min<std::size_t>(ids.size(), std::atoi(argv[5])));
  }
  qwen35::NpuDecoder npu(*file, argv[2]);
  const auto& c = npu.config();
  const int layers = c.layers;
  const int used = c.experts_used;
  auto state = npu.NewState();
  std::vector<std::int16_t> out;
  out.reserve(ids.size() * layers * used);
  for (std::size_t t = 0; t < ids.size(); ++t) {
    npu.Step(ids[t], state);
    for (int l = 0; l < layers; ++l) {
      const int* e = npu.experts(l);
      for (int i = 0; i < used; ++i) {
        out.push_back(static_cast<std::int16_t>(e[i]));
      }
    }
  }
  std::ofstream f(argv[4], std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()),
          static_cast<std::streamsize>(out.size() * sizeof(std::int16_t)));
  std::printf("%zu tokens x %d layers x %d experts -> %s\n", ids.size(),
              layers, used, argv[4]);
  return 0;
}
