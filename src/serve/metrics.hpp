#pragma once

#include <array>
#include <map>
#include <mutex>
#include <string>

// Server counters in the Prometheus text format (GET /metrics).
namespace gufo_npu::serve {

class Metrics {
public:
  void Request(int status);
  void Generation(int prompt_tokens, int cached_prompt_tokens,
                  int completion_tokens, double prefill_seconds,
                  double decode_seconds, int steps, int accepted);
  void TimeToFirstToken(double seconds);  // from arrival, queue included
  void RequestDuration(double seconds);

  std::string Render(int waiting, bool running) const;

private:
  static constexpr std::array<double, 12> kBuckets = {
      0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120};
  struct Histogram {
    std::array<long, kBuckets.size()> counts{};
    long count = 0;
    double sum = 0;
    void Add(double x);
  };
  static void RenderHistogram(std::string& out, const char* name,
                              const char* help, const Histogram& h);

  mutable std::mutex mutex_;
  std::map<int, long> requests_;  // by HTTP status
  long prompt_tokens_ = 0;
  long cached_prompt_tokens_ = 0;
  long completion_tokens_ = 0;
  double prefill_seconds_ = 0;
  double decode_seconds_ = 0;
  long steps_ = 0;
  long accepted_ = 0;
  Histogram ttft_;
  Histogram duration_;
};

}  // namespace gufo_npu::serve
