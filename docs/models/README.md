# Models

| Model | Role | Weights | State |
|---|---|---|---|
| [Qwen3.5-2B](qwen3.5-2b/README.md) | MVP; fast side tasks | Unsloth GGUF (UD-Q4_K_XL) | NPU decode with MTP, NPU prefill |
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/README.md) | second pass (MoE) | Unsloth GGUF (UD-Q4_K_XL) | CPU reference and decoder, NPU decode |
