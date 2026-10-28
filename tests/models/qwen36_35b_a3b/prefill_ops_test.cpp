#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "core/bfp16.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/program.hpp"

// Qwen3.6-35B-A3B's prefill programs (src/models/qwen36_35b_a3b/kernels) on
// random q4nx weights of every converted shape, against the CPU over the
// same q4nx weights:
//   - qwen36_35b_a3b_dequant: q4nx (the GEMV's block order, as the decoder
//     keeps it) to bfp16 in the GEMM's B order. It rounds each weight once,
//     so its weights must be as close to the q4nx ones as the host's packing
//     (bfp16::PackGemmB of the unpacked q4nx weights): relative error within
//     5% of the host's. (About 0.3% of bytes still differ: q4nx weights have
//     few significant bits, so exact ties to the 8-bit block mantissa are
//     common, and the core breaks some of them the other way.)
//   - qwen36_35b_a3b_prefill: C = A W^T at M = 256, 512 and 1024 on the
//     converted weights, relative error < 1e-2 against a double-precision
//     product over the q4nx weights (A rounded to bf16 as it is passed).
// Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace bfp16 = gufo_npu::bfp16;
namespace quant = gufo_npu::quant;

constexpr double kMaxGemmError = 1e-2;
// The converted weights' error against q4nx, relative to the host packing's
// (a second rounding, as the spike's kernel had, gave 8.4e-3 against 6.2e-3:
// 1.35).
constexpr double kMaxWeightErrorRatio = 1.05;

double RelativeError(const std::vector<float>& got,
                     const std::vector<float>& want) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    diff += d * d;
    norm += static_cast<double>(want[i]) * want[i];
  }
  return std::sqrt(diff / norm);
}

constexpr std::size_t kShapes[][2] = {
    {2048, 12288}, {2048, 9216}, {4096, 2048}, {2048, 1024}, {512, 2048}};
constexpr std::size_t kChunks[] = {256, 512, 1024};

std::string Name(const char* program, std::size_t k, std::size_t n) {
  return std::string(program) + "_k" + std::to_string(k) + "_n" +
         std::to_string(n);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen36_prefill_ops_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  npu::Program dequant(argv[1], "qwen36_35b_a3b_dequant");
  npu::Program gemm(argv[1], "qwen36_35b_a3b_prefill");
  std::mt19937 rng(11);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  int failures = 0;

  for (const auto& shape : kShapes) {
    const std::size_t k = shape[0];
    const std::size_t n = shape[1];
    std::vector<float> w(n * k);
    for (auto& v : w) {
      v = weight(rng);
    }
    const auto packed = q4nx::PackFloat(w.data(), n, k, q4nx::Fit::kSearch);
    const auto wq = q4nx::Unpack(packed);  // the weights decode multiplies

    npu::Buffer qw(dequant, packed.bytes.size());
    std::memcpy(qw.data(), packed.bytes.data(), packed.bytes.size());
    qw.ToDevice();
    const std::size_t b_bytes = bfp16::PackedBytes(n, k);
    npu::Buffer b(gemm, b_bytes);
    npu::Batch convert(dequant);
    convert.Add(Name("qwen36_35b_a3b_dequant", k, n), {&qw, &b});
    convert.Run();
    b.FromDevice();
    const auto host = bfp16::PackGemmB(wq.data(), n, k);
    const auto* got = b.as<std::uint8_t>();
    const std::vector<std::uint8_t> npu_bytes(got, got + host.size());
    std::size_t different = 0;
    for (std::size_t i = 0; i < host.size(); ++i) {
      different += npu_bytes[i] != host[i];
    }
    const double npu_err =
        RelativeError(bfp16::UnpackGemmB(npu_bytes, n, k), wq);
    const double host_err = RelativeError(bfp16::UnpackGemmB(host, n, k), wq);
    const bool convert_ok = npu_err <= host_err * kMaxWeightErrorRatio;
    failures += convert_ok ? 0 : 1;
    std::printf(
        "%s dequant k%zu n%zu: weights vs q4nx %.3e (host packing "
        "%.3e, limit x%.2f); %zu of %zu bytes differ from it\n",
        convert_ok ? "ok  " : "FAIL", k, n, npu_err, host_err,
        kMaxWeightErrorRatio, different, host.size());

    for (const std::size_t m : kChunks) {
      std::vector<std::uint16_t> a(m * k);
      for (auto& v : a) {
        v = quant::FloatToBf16(act(rng));
      }
      npu::Buffer ab(gemm, a.size() * 2);
      npu::Buffer cb(gemm, m * n * 2);
      std::memcpy(ab.data(), a.data(), a.size() * 2);
      ab.ToDevice();
      npu::Batch call(gemm);
      call.Add("qwen36_35b_a3b_prefill_m" + std::to_string(m) + "_k" +
                   std::to_string(k) + "_n" + std::to_string(n),
               {&ab, &b, &cb});
      call.Run();
      cb.FromDevice();
      const auto* c = cb.as<std::uint16_t>();
      // A sample of rows (every 37th) against double precision.
      double diff = 0;
      double norm = 0;
      for (std::size_t r = 0; r < m; r += 37) {
        for (std::size_t j = 0; j < n; ++j) {
          double want = 0;
          for (std::size_t i = 0; i < k; ++i) {
            want += static_cast<double>(quant::Bf16ToFloat(a[r * k + i])) *
                    wq[j * k + i];
          }
          const double d = quant::Bf16ToFloat(c[r * n + j]) - want;
          diff += d * d;
          norm += want * want;
        }
      }
      const double err = std::sqrt(diff / norm);
      const bool ok = err < kMaxGemmError;
      failures += ok ? 0 : 1;
      std::printf("%s GEMM m%zu k%zu n%zu: rel err %.2e (limit %.0e)\n",
                  ok ? "ok  " : "FAIL", m, k, n, err, kMaxGemmError);
    }
  }
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
