#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Host checks for running gufo-npu on a Strix Halo NPU. The parsing helpers
// are pure so tests can feed them fixture text.
namespace gufo_npu::diagnose {

enum class Status { kOk, kInfo, kWarn, kFail };

struct Check {
  std::string name;
  Status status;
  std::string detail;
};

std::string_view StatusLabel(Status status);

// MemTotal from /proc/meminfo text, in KiB.
std::optional<std::uint64_t> MemTotalKib(std::string_view meminfo);

// True when the kernel command line turns the AMD IOMMU off, which hides
// the NPU from amdxdna.
bool CmdlineDisablesIommu(std::string_view cmdline);

// "21.3 GiB" style.
std::string FormatBytes(std::uint64_t bytes);

// Weights live in pinned host buffers, so the memlock limit must cover the
// model. `limit` is nullopt when RLIMIT_MEMLOCK is unlimited.
Check MemlockCheck(std::optional<std::uint64_t> limit,
                   std::uint64_t needed_bytes);

// The NPU's PCI runtime power control: "on" keeps it awake, "auto" lets it
// suspend between requests and pay a wake-up delay.
Check RuntimePmCheck(std::optional<std::string> control);

// Reads /dev, /sys and /proc on this host.
std::vector<Check> HostChecks();

}  // namespace gufo_npu::diagnose
