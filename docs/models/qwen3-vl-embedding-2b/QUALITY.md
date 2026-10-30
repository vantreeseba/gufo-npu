# Qwen3-VL-Embedding-2B: quality

The oracle is HF transformers in float32 on the model's own weights
(`tools/reference/qwen3vl_embedding_oracle.py`): 3 queries with a retrieval
instruction, 6 documents (one German), and inputs for the default and custom
instructions, Unicode, code and a ~1,500-token text that crosses a GEMM
chunk. The script also checks that the chat template renders exactly what
`input.cpp` builds and that the tokenizer appends `<|endoftext|>`.

| Check | Tolerance | Result |
|---|---|---|
| Token ids (`InputIds`) = HF's | exact | pending (first run on the NPU box) |
| Tokenizer fixtures (18 texts) | exact | pending |
| CPU reference vs HF fp32 | cosine ≥ 0.999 per input | pending |
| NPU vs HF fp32 | cosine ≥ 0.99 per input | pending |
| Retrieval: each query's first document | = HF's | pending |
| Each input alone vs batched (NPU) | cosine ≥ 0.9999 | pending |
| GEMM shapes vs double reference | relative L2 < 1e-2 | pending |

## Known differences from the reference embedder

- Inputs over the context limit are refused rather than truncated.
- Text only; images and video are not supported.
