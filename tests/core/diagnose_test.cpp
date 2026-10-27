#include "core/diagnose.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  using namespace gufo_npu::diagnose;

  Expect(MemTotalKib("MemTotal:       64566252 kB\nMemFree: 1 kB\n") ==
             64566252ULL,
         "MemTotal parses");
  Expect(!MemTotalKib("MemFree: 1 kB\n"), "missing MemTotal is nullopt");

  Expect(CmdlineDisablesIommu("BOOT_IMAGE=/vmlinuz ro amd_iommu=off quiet"),
         "amd_iommu=off detected");
  Expect(CmdlineDisablesIommu("iommu=off"), "iommu=off detected");
  Expect(!CmdlineDisablesIommu("ro amd_iommu=pt quiet"),
         "amd_iommu=pt is allowed");
  Expect(!CmdlineDisablesIommu("ro xamd_iommu=offx"), "only whole tokens");

  Expect(FormatBytes(512) == "512 B", "bytes");
  Expect(FormatBytes(8ULL << 20) == "8.0 MiB", "MiB");
  Expect(FormatBytes(24ULL << 30) == "24.0 GiB", "GiB");

  Expect(MemlockCheck(std::nullopt, 1).status == Status::kOk,
         "unlimited memlock passes");
  Expect(MemlockCheck(8ULL << 20, 24ULL << 30).status == Status::kFail,
         "8 MiB memlock fails");

  Expect(RuntimePmCheck(std::string("on\n")).status == Status::kOk, "pm on");
  Expect(RuntimePmCheck(std::string("auto\n")).status == Status::kWarn,
         "pm auto warns");
  Expect(RuntimePmCheck(std::nullopt).status == Status::kInfo, "pm unknown");

  if (failures == 0) {
    std::puts("PASS: diagnose_test");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
