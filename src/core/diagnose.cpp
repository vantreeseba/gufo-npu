#include "core/diagnose.hpp"

#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace gufo_npu::diagnose {
namespace {

constexpr char kAccelNode[] = "/dev/accel/accel0";
constexpr char kAccelSysfs[] = "/sys/class/accel/accel0/device";
constexpr char kDriverModule[] = "/sys/module/amdxdna";
// Qwen3.6-35B-A3B at ~4.5 bits per weight plus headroom.
constexpr std::uint64_t kModelPinnedBytes = 24ULL << 30;

std::optional<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string Trim(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

}  // namespace

std::string_view StatusLabel(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInfo:
      return "info";
    case Status::kWarn:
      return "warn";
    case Status::kFail:
      return "FAIL";
  }
  return "?";
}

std::optional<std::uint64_t> MemTotalKib(std::string_view meminfo) {
  constexpr std::string_view kKey = "MemTotal:";
  const auto at = meminfo.find(kKey);
  if (at == std::string_view::npos) {
    return std::nullopt;
  }
  auto rest = meminfo.substr(at + kKey.size());
  const auto digits = rest.find_first_of("0123456789");
  if (digits == std::string_view::npos) {
    return std::nullopt;
  }
  rest = rest.substr(digits);
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(rest.data(), rest.data() + rest.size(), value);
  if (error != std::errc{} || end == rest.data()) {
    return std::nullopt;
  }
  return value;
}

bool CmdlineDisablesIommu(std::string_view cmdline) {
  std::size_t pos = 0;
  while (pos < cmdline.size()) {
    const auto end = cmdline.find_first_of(" \t\n", pos);
    const auto token = cmdline.substr(pos, end == std::string_view::npos
                                               ? std::string_view::npos
                                               : end - pos);
    if (token == "amd_iommu=off" || token == "iommu=off") {
      return true;
    }
    if (end == std::string_view::npos) {
      break;
    }
    pos = end + 1;
  }
  return false;
}

std::string FormatBytes(std::uint64_t bytes) {
  constexpr std::array<const char*, 5> kUnits = {"B", "KiB", "MiB", "GiB",
                                                 "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < kUnits.size()) {
    value /= 1024.0;
    ++unit;
  }
  std::array<char, 32> buffer{};
  if (unit == 0) {
    std::snprintf(buffer.data(), buffer.size(), "%llu B",
                  static_cast<unsigned long long>(bytes));
  } else {
    std::snprintf(buffer.data(), buffer.size(), "%.1f %s", value, kUnits[unit]);
  }
  return buffer.data();
}

Check MemlockCheck(std::optional<std::uint64_t> limit,
                   std::uint64_t needed_bytes) {
  if (!limit) {
    return {"memlock", Status::kOk, "unlimited"};
  }
  const auto detail = FormatBytes(*limit) + "; the model pins about " +
                      FormatBytes(needed_bytes) +
                      " (run with --ulimit memlock=-1:-1 or raise it in "
                      "/etc/security/limits.conf)";
  return {"memlock", *limit >= needed_bytes ? Status::kOk : Status::kFail,
          detail};
}

Check RuntimePmCheck(std::optional<std::string> control) {
  if (!control) {
    return {"runtime power", Status::kInfo, "unknown (no sysfs entry)"};
  }
  const auto value = Trim(*control);
  if (value == "on") {
    return {"runtime power", Status::kOk, "on (NPU stays awake)"};
  }
  return {"runtime power", Status::kWarn,
          value +
              ": the NPU suspends when idle and the next request pays a wake "
              "delay; to keep it awake: echo on | sudo tee " +
              std::string(kAccelSysfs) + "/power/control"};
}

std::vector<Check> HostChecks() {
  std::vector<Check> checks;

  utsname uts{};
  if (uname(&uts) == 0) {
    checks.push_back({"kernel", Status::kInfo, uts.release});
  }

  if (std::filesystem::exists(kDriverModule)) {
    const auto version = ReadFile(std::string(kDriverModule) + "/version");
    checks.push_back(
        {"amdxdna driver", Status::kOk,
         version ? "loaded, version " + Trim(*version) : "loaded (in-tree)"});
  } else {
    checks.push_back(
        {"amdxdna driver", Status::kFail,
         "not loaded; needs Linux 7.0+ (in-tree) or amdxdna-dkms"});
  }

  if (!std::filesystem::exists(kAccelNode)) {
    checks.push_back({"NPU device", Status::kFail,
                      std::string(kAccelNode) +
                          " missing (in a container, pass --device " +
                          kAccelNode + ")"});
  } else if (access(kAccelNode, R_OK | W_OK) != 0) {
    checks.push_back({"NPU device", Status::kFail,
                      std::string(kAccelNode) +
                          " exists but is not read/write for this user"});
  } else {
    checks.push_back({"NPU device", Status::kOk, kAccelNode});
  }

  if (const auto cmdline = ReadFile("/proc/cmdline")) {
    const bool off = CmdlineDisablesIommu(*cmdline);
    checks.push_back({"iommu", off ? Status::kFail : Status::kOk,
                      off ? "disabled on the kernel command line; the NPU "
                            "needs it (amd_iommu=pt is fine)"
                          : "not disabled"});
  }

  rlimit memlock{};
  if (getrlimit(RLIMIT_MEMLOCK, &memlock) == 0) {
    const std::optional<std::uint64_t> limit =
        memlock.rlim_cur == RLIM_INFINITY
            ? std::nullopt
            : std::optional<std::uint64_t>(memlock.rlim_cur);
    checks.push_back(MemlockCheck(limit, kModelPinnedBytes));
  }

  checks.push_back(
      RuntimePmCheck(ReadFile(std::string(kAccelSysfs) + "/power/control")));

  if (const auto meminfo = ReadFile("/proc/meminfo")) {
    if (const auto kib = MemTotalKib(*meminfo)) {
      checks.push_back({"memory", Status::kInfo, FormatBytes(*kib * 1024)});
    }
  }
  return checks;
}

}  // namespace gufo_npu::diagnose
