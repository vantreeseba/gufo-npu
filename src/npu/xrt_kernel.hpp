#pragma once

// XRT helpers shared by the NPU runtime sources. Only compiled with XRT.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

// One instruction sequence (<dir>/<name>.insts.bin) run through the kernel's
// instruction-buffer argument (the "legacy" flow): the host owns the words
// and may rewrite them between calls. Runs of this flow and of the ELF flow
// cannot share a runlist (XRT leaves a mixed one in ERT_CMD_STATE_NEW).
struct InstrSequence {
  std::vector<std::uint32_t> words;
  xrt::bo bo;

  // `kernel` is xrt::kernel(*hw.context, hw.kernel_name).
  InstrSequence(HwContext& hw, const xrt::kernel& kernel,
                const std::filesystem::path& path)
      : words(ReadWords(path)) {
    if (words.empty()) {
      throw std::runtime_error("empty instruction sequence " + path.string());
    }
    // The firmware refuses instruction buffers that are not cacheable.
    bo = xrt::bo(hw.device, words.size() * sizeof(std::uint32_t),
                 XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    Sync();
  }

  // Copies `words` to the device buffer.
  void Sync() {
    std::memcpy(bo.map<void*>(), words.data(),
                words.size() * sizeof(std::uint32_t));
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run Run(const xrt::kernel& kernel,
               std::initializer_list<const xrt::bo*> args) {
    xrt::run run(kernel);
    run.set_arg(0, kOpcodeRun);
    run.set_arg(1, bo);
    run.set_arg(2, static_cast<std::uint32_t>(words.size()));
    int index = 3;
    for (const xrt::bo* arg : args) {
      run.set_arg(index++, *arg);
    }
    return run;
  }

  static std::vector<std::uint32_t> ReadWords(
      const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("cannot read " + path.string());
    }
    const std::string bytes((std::istreambuf_iterator<char>(in)), {});
    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(),
                words.size() * sizeof(std::uint32_t));
    return words;
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
