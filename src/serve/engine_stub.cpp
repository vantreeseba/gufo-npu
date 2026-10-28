// TEMPORARY (remove at integration): LoadEngine until the engine part
// (src/serve/engine.cpp) lands, so `gufo-npu serve` links on this branch.
#include <stdexcept>

#include "serve/engine.hpp"

namespace gufo_npu::serve {

std::unique_ptr<Engine> LoadEngine(const EngineOptions& /*options*/) {
  throw std::runtime_error(
      "no engine in this build (engine_stub.cpp placeholder)");
}

}  // namespace gufo_npu::serve
