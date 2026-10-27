#pragma once

// XRT helpers shared by the NPU runtime sources. Only compiled with XRT.

#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace gufo_npu::npu {

using Clock = std::chrono::steady_clock;

inline double MicrosSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

// One compiled design (<dir>/<name>.xclbin + .insts.elf) loaded into its own
// hardware context.
struct LoadedKernel {
  // Kernel names in mlir-aie xclbins start with this prefix.
  static constexpr char kKernelPrefix[] = "MLIR_AIE";
  // Opcode for "run the instruction stream" in mlir-aie's NPU kernels.
  static constexpr unsigned int kOpcodeRun = 3;

  xrt::device device{0U};
  xrt::xclbin xclbin;
  std::optional<xrt::hw_context> context;
  std::optional<xrt::elf> elf;
  std::optional<xrt::module> module;
  xrt::kernel kernel;

  LoadedKernel(const std::filesystem::path& dir, const std::string& design)
      : xclbin((dir / (design + ".xclbin")).string()) {
    std::string name;
    for (const auto& k : xclbin.get_kernels()) {
      if (k.get_name().rfind(kKernelPrefix, 0) == 0) {
        name = k.get_name();
        break;
      }
    }
    if (name.empty()) {
      throw std::runtime_error("no MLIR_AIE kernel in " + design + ".xclbin");
    }
    device.register_xclbin(xclbin);
    context.emplace(device, xclbin.get_uuid());
    elf.emplace((dir / (design + ".insts.elf")).string());
    module.emplace(*elf);
    kernel = xrt::ext::kernel(*context, *module, name);
  }

  xrt::bo Buffer(std::size_t bytes) { return xrt::ext::bo{device, bytes}; }

  // A run with the design's runtime-sequence arguments in order.
  xrt::run Run(std::initializer_list<const xrt::bo*> args) {
    xrt::run run(kernel);
    run.set_arg(0, kOpcodeRun);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    int index = 3;
    for (const xrt::bo* bo : args) {
      run.set_arg(index++, *bo);
    }
    return run;
  }
};

}  // namespace gufo_npu::npu
