#include "models/qwen35/moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace gufo_npu::qwen35 {
namespace {

// The smallest positive normal fp16.
constexpr double kWeightSumFloor = 6.103515625e-5;

}  // namespace

void Route(const float* logits, int experts, int k, int* chosen,
           float* weights) {
  thread_local std::vector<double> prob;
  thread_local std::vector<int> order;
  prob.resize(experts);
  order.resize(experts);
  const double max_logit = *std::max_element(logits, logits + experts);
  double total = 0;
  for (int e = 0; e < experts; ++e) {
    prob[e] = std::exp(static_cast<double>(logits[e]) - max_logit);
    total += prob[e];
  }
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) {
                      return prob[a] > prob[b] || (prob[a] == prob[b] && a < b);
                    });
  double top = 0;
  for (int slot = 0; slot < k; ++slot) {
    top += prob[order[slot]] / total;
  }
  top = std::max(top, kWeightSumFloor);
  for (int slot = 0; slot < k; ++slot) {
    chosen[slot] = order[slot];
    weights[slot] = static_cast<float>(prob[order[slot]] / total / top);
  }
}

float SharedExpertScale(float logit) {
  return static_cast<float>(1.0 /
                            (1.0 + std::exp(-static_cast<double>(logit))));
}

}  // namespace gufo_npu::qwen35
