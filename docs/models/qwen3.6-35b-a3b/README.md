# Qwen3.6-35B-A3B

Hybrid MoE: 40 layers (30 Gated DeltaNet + 10 gated full-attention, GQA 16/2,
head dim 256), 256 experts with 8 active plus 1 shared expert (FFN 512),
hidden 2048, vocab 248,320, one MTP layer. About 3 B active parameters per
token, ~1.7 GB of weights read per decoded token at ~4.5 bits per weight.

Weights, in order of support:
1. FastFlowLM's Q4NX pack (`qwen3.6-moe:35b-a3b`), which is already laid out
   for the NPU.
2. Unsloth UD GGUF (arch `qwen35moe`), repacked for the NPU (also carries the
   MTP head).

Not implemented yet; see [the plan](../../PLAN.md).
