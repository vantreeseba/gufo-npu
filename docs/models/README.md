# Models

| Model | Role | Weights | State |
|---|---|---|---|
| [Qwen3.5-2B](qwen3.5-2b/README.md) | MVP; fast side tasks | Unsloth GGUF (UD-Q4_K_XL) | NPU decode with MTP, NPU prefill |
| [Qwen3.5-4B](qwen3.5-4b/README.md) | dense, bigger sibling of the 2B; its own NPU programs | Unsloth GGUF (UD-Q4_K_XL) | NPU decode with MTP, NPU prefill |
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/README.md) | second pass (MoE) | Unsloth GGUF (UD-Q4_K_XL) | CPU reference and decoder, NPU decode |
| [Qwen3-VL-Embedding-2B](qwen3-vl-embedding-2b/README.md) | text embeddings for search (`/v1/embeddings`) | mradermacher F16 GGUF | NPU GEMMs, matches HF (cosine ≥ 0.998); ~4k tok/s batched |

Each model folder holds:

- `README.md`: the model card, weights (Hugging Face repo and revision) and
  usage.
- `BENCHMARKS.md`: measurements against FastFlowLM, with dates and settings.
- `QUALITY.md`: checks against independent implementations, tolerances and
  open gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.

A model's code, kernels and tests live in its own package and folders; see
"Code layout" in [AGENTS.md](../../AGENTS.md).

