#pragma once

// npu::Buffer's XRT side, for the runtime sources that pass buffers to
// calls (gemv_xrt.cpp, program_xrt.cpp). Only compiled with XRT.

#include <cstddef>

#include "npu/gemv.hpp"
#include "xrt/xrt_bo.h"

namespace gufo_npu::npu {

struct Buffer::Impl {
  xrt::bo bo;
  std::size_t bytes;
};

}  // namespace gufo_npu::npu
