#pragma once

// Expert routing of a Qwen3.5/3.6 MoE layer (GGUF arch `qwen35moe`), shared
// by the reference and the decoders. docs/models/qwen3.6-35b-a3b/README.md.
namespace gufo_npu::qwen35 {

// From `experts` router logits: softmax over all of them, the `k` most
// probable experts (ties to the lower index) in order of probability, and
// their probabilities renormalized to sum 1 (llama.cpp floors the sum at the
// smallest normal fp16 first). Writes `k` ids to `chosen` and `k` weights.
void Route(const float* logits, int experts, int k, int* chosen,
           float* weights);

// The shared expert's scale from its gate logit: sigmoid(logit).
float SharedExpertScale(float logit);

}  // namespace gufo_npu::qwen35
