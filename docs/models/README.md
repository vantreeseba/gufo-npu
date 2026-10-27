# Models

| Model | Role | Weights | State |
|---|---|---|---|
| [Qwen3.5-2B](qwen3.5-2b/README.md) | MVP; fast side tasks | FLM Q4NX pack, Unsloth GGUF | planned |
| [Qwen3.5-4B](qwen3.5-4b/README.md) | dense, bigger sibling of the 2B; its own NPU programs | Unsloth GGUF (UD-Q4_K_XL) | NPU decode with MTP, NPU prefill |
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/README.md) | second pass (MoE) | FLM Q4NX pack, then Unsloth GGUF | CPU reference |
