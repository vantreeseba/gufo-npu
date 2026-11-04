#pragma once

// npu::Buffer's XRT side, for the runtime sources that pass buffers to
// calls (gemv_xrt.cpp, program_xrt.cpp). Only compiled with XRT.

#include <cstddef>
#include <optional>

#include "npu/gemv.hpp"
#include "xrt/xrt_bo.h"

namespace gufo_npu::npu {

struct Buffer::Impl {
  xrt::bo bo;
  std::size_t bytes;
  // A sub-buffer's root buffer and its offset in it: Sub always makes the
  // XRT sub-buffer from the root. Syncing an XRT sub-buffer of a sub-buffer
  // left the CPU and the NPU with different data (stale decoder inputs).
  std::optional<xrt::bo> root;
  std::size_t offset = 0;
};

}  // namespace gufo_npu::npu
