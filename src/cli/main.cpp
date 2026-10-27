#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "build_info.hpp"
#include "core/diagnose.hpp"
#include "npu/device.hpp"

namespace {

using gufo_npu::diagnose::Check;
using gufo_npu::diagnose::Status;

constexpr char kUsage[] =
    "usage: gufo-npu <command> [options]\n"
    "\n"
    "commands:\n"
    "  diagnose                 check driver, device, limits and XRT\n"
    "  smoke [--iters N] [--kernels DIR]\n"
    "                           run the smoke kernel and time dispatches\n"
    "  bench dispatch [--rounds N] [--depths 1,8,32] [--kernels DIR]\n"
    "                           per-dispatch cost: sync, async and runlist\n"
    "  version                  print the version\n";

void PrintChecks(const std::vector<Check>& checks) {
  for (const auto& check : checks) {
    std::printf(
        "  [%-4s] %-16s %s\n",
        std::string(gufo_npu::diagnose::StatusLabel(check.status)).c_str(),
        check.name.c_str(), check.detail.c_str());
  }
}

int Diagnose() {
  std::printf("gufo-npu %s\n", gufo_npu::build_info::kVersion);
  auto checks = gufo_npu::diagnose::HostChecks();
  const auto runtime = gufo_npu::npu::RuntimeChecks();
  checks.insert(checks.end(), runtime.begin(), runtime.end());
  PrintChecks(checks);
  for (const auto& check : checks) {
    if (check.status == Status::kFail) {
      return 1;
    }
  }
  return 0;
}

std::filesystem::path ExecutableDir() {
  std::error_code error;
  const auto exe = std::filesystem::read_symlink("/proc/self/exe", error);
  return error ? std::filesystem::path{} : exe.parent_path();
}

// --kernels, then $GUFO_NPU_KERNEL_DIR, then <prefix>/share/..., then the
// build tree this binary came from.
std::filesystem::path KernelDir(const std::optional<std::string>& flag) {
  if (flag) {
    return *flag;
  }
  if (const char* env = std::getenv("GUFO_NPU_KERNEL_DIR")) {
    return env;
  }
  const auto installed = ExecutableDir().parent_path() /
                         gufo_npu::build_info::kInstallKernelSubdir;
  if (std::filesystem::is_directory(installed)) {
    return installed;
  }
  return gufo_npu::build_info::kBuildKernelDir;
}

int Smoke(const std::vector<std::string_view>& args) {
  int iterations = 100;
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--iters" && i + 1 < args.size()) {
      iterations = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--kernels" && i + 1 < args.size()) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  const auto dir = KernelDir(kernels);
  std::printf("kernels: %s\n", dir.c_str());
  const auto result = gufo_npu::npu::RunSmoke(dir, iterations);
  if (!result.ok) {
    std::fprintf(stderr, "smoke: FAIL: %s\n", result.error.c_str());
    return 1;
  }
  std::printf(
      "smoke: PASS, %d dispatches; first %.1f us, mean %.1f us, min %.1f us\n",
      result.iterations, result.first_us, result.mean_us, result.min_us);
  return 0;
}

std::vector<int> ParseInts(std::string_view text) {
  std::vector<int> values;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const auto comma = text.find(',', pos);
    const auto item = text.substr(pos, comma == std::string_view::npos
                                           ? std::string_view::npos
                                           : comma - pos);
    if (!item.empty()) {
      values.push_back(std::atoi(std::string(item).c_str()));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    pos = comma + 1;
  }
  return values;
}

int BenchDispatch(const std::vector<std::string_view>& args) {
  int rounds = 200;
  std::vector<int> depths = {1, 4, 8, 16, 32, 81};
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--rounds" && i + 1 < args.size()) {
      rounds = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--depths" && i + 1 < args.size()) {
      depths = ParseInts(args[++i]);
    } else if (args[i] == "--kernels" && i + 1 < args.size()) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  const auto result =
      gufo_npu::npu::BenchDispatch(KernelDir(kernels), rounds, depths);
  std::printf("%-8s %6s %11s %14s\n", "mode", "depth", "dispatches",
              "us/dispatch");
  for (const auto& t : result.timings) {
    std::printf("%-8s %6d %11d %14.1f\n", t.mode.c_str(), t.depth, t.dispatches,
                t.us_per_dispatch);
  }
  if (!result.ok) {
    std::fprintf(stderr, "bench dispatch: FAIL: %s\n", result.error.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  if (args.empty() || args[0] == "help" || args[0] == "--help") {
    std::cout << kUsage;
    return args.empty() ? 2 : 0;
  }
  const std::vector<std::string_view> rest(args.begin() + 1, args.end());
  if (args[0] == "version" || args[0] == "--version") {
    std::printf("gufo-npu %s\n", gufo_npu::build_info::kVersion);
    return 0;
  }
  if (args[0] == "diagnose") {
    return Diagnose();
  }
  if (args[0] == "smoke") {
    return Smoke(rest);
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "dispatch") {
    return BenchDispatch({rest.begin() + 1, rest.end()});
  }
  std::cerr << "unknown command: " << args[0] << "\n" << kUsage;
  return 2;
}
