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

// Opcode for "run the instruction stream" in mlir-aie's NPU kernels.
inline constexpr unsigned int kOpcodeRun = 3;

// An xclbin (<dir>/<name>.xclbin) loaded into its own hardware context.
// Instruction sequences built for it run in that context.
struct HwContext {
  // Kernel names in mlir-aie xclbins start with this prefix.
  static constexpr char kKernelPrefix[] = "MLIR_AIE";

  xrt::device device{0U};
  xrt::xclbin xclbin;
  std::optional<xrt::hw_context> context;
  std::string kernel_name;

  HwContext(const std::filesystem::path& dir, const std::string& name)
      : xclbin((dir / (name + ".xclbin")).string()) {
    for (const auto& k : xclbin.get_kernels()) {
      if (k.get_name().rfind(kKernelPrefix, 0) == 0) {
        kernel_name = k.get_name();
        break;
      }
    }
    if (kernel_name.empty()) {
      throw std::runtime_error("no MLIR_AIE kernel in " + name + ".xclbin");
    }
    device.register_xclbin(xclbin);
    context.emplace(device, xclbin.get_uuid());
  }

  xrt::bo Buffer(std::size_t bytes) { return xrt::ext::bo{device, bytes}; }
};

// One instruction sequence (<dir>/<name>.insts.elf) of a loaded xclbin.
struct Sequence {
  xrt::elf elf;
  xrt::module module;
  xrt::kernel kernel;

  Sequence(HwContext& hw, const std::filesystem::path& dir,
           const std::string& name)
      : elf((dir / (name + ".insts.elf")).string()),
        module(elf),
        kernel(xrt::ext::kernel(*hw.context, module, hw.kernel_name)) {}

  // A run with the sequence's arguments in order.
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

// A design whose xclbin and sequence share a name, in its own context.
struct LoadedKernel {
  HwContext hw;
  Sequence sequence;

  LoadedKernel(const std::filesystem::path& dir, const std::string& design)
      : hw(dir, design), sequence(hw, dir, design) {}

  xrt::bo Buffer(std::size_t bytes) { return hw.Buffer(bytes); }
  xrt::runlist Runlist() { return xrt::runlist(*hw.context); }
  xrt::run Run(std::initializer_list<const xrt::bo*> args) {
    return sequence.Run(args);
  }
};

}  // namespace gufo_npu::npu
