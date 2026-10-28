#include "serve/metrics.hpp"

#include <cstdio>
#include <type_traits>

namespace gufo_npu::serve {
namespace {

void Line(std::string& out, const char* fmt, auto... args) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), fmt, args...);
  out += buf;
}

void Header(std::string& out, const char* name, const char* type,
            const char* help) {
  Line(out, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, type);
}

}  // namespace

void Metrics::Histogram::Add(double x) {
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    if (x <= kBuckets[i]) {
      ++counts[i];
    }
  }
  ++count;
  sum += x;
}

void Metrics::Request(int status) {
  std::lock_guard lock(mutex_);
  ++requests_[status];
}

void Metrics::Generation(int prompt_tokens, int cached_prompt_tokens,
                         int completion_tokens, double prefill_seconds,
                         double decode_seconds, int steps, int accepted) {
  std::lock_guard lock(mutex_);
  prompt_tokens_ += prompt_tokens;
  cached_prompt_tokens_ += cached_prompt_tokens;
  completion_tokens_ += completion_tokens;
  prefill_seconds_ += prefill_seconds;
  decode_seconds_ += decode_seconds;
  steps_ += steps;
  accepted_ += accepted;
}

void Metrics::TimeToFirstToken(double seconds) {
  std::lock_guard lock(mutex_);
  ttft_.Add(seconds);
}

void Metrics::RequestDuration(double seconds) {
  std::lock_guard lock(mutex_);
  duration_.Add(seconds);
}

void Metrics::RenderHistogram(std::string& out, const char* name,
                              const char* help, const Histogram& h) {
  Header(out, name, "histogram", help);
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    Line(out, "%s_bucket{le=\"%g\"} %ld\n", name, kBuckets[i], h.counts[i]);
  }
  Line(out, "%s_bucket{le=\"+Inf\"} %ld\n", name, h.count);
  Line(out, "%s_sum %.6f\n%s_count %ld\n", name, h.sum, name, h.count);
}

std::string Metrics::Render(int waiting, bool running) const {
  std::lock_guard lock(mutex_);
  std::string out;
  Header(out, "gufo_npu_requests_total", "counter",
         "Chat completion requests by HTTP status.");
  for (const auto& [status, n] : requests_) {
    Line(out, "gufo_npu_requests_total{status=\"%d\"} %ld\n", status, n);
  }
  const auto counter = [&](const char* name, const char* help, auto value) {
    Header(out, name, "counter", help);
    if constexpr (std::is_floating_point_v<decltype(value)>) {
      Line(out, "%s %.6f\n", name, value);
    } else {
      Line(out, "%s %ld\n", name, value);
    }
  };
  counter("gufo_npu_prompt_tokens_total", "Prompt tokens.", prompt_tokens_);
  counter("gufo_npu_prompt_tokens_cached_total",
          "Prompt tokens reused from the prompt cache.", cached_prompt_tokens_);
  counter("gufo_npu_completion_tokens_total", "Generated tokens.",
          completion_tokens_);
  counter("gufo_npu_prefill_seconds_total",
          "Engine time to the first generated token.", prefill_seconds_);
  counter("gufo_npu_decode_seconds_total", "Engine time generating.",
          decode_seconds_);
  counter("gufo_npu_decode_steps_total", "Decode passes.", steps_);
  counter("gufo_npu_drafts_accepted_total", "MTP drafts kept.", accepted_);
  Header(out, "gufo_npu_queue_waiting", "gauge",
         "Requests waiting for the engine.");
  Line(out, "gufo_npu_queue_waiting %d\n", waiting);
  Header(out, "gufo_npu_generation_running", "gauge",
         "1 while a generation runs.");
  Line(out, "gufo_npu_generation_running %d\n", running ? 1 : 0);
  RenderHistogram(out, "gufo_npu_time_to_first_token_seconds",
                  "Arrival to first generated token, queue included.", ttft_);
  RenderHistogram(out, "gufo_npu_request_duration_seconds",
                  "Arrival to the end of the reply.", duration_);
  return out;
}

}  // namespace gufo_npu::serve
