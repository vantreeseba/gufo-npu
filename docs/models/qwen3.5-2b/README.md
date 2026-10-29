# Qwen3.5-2B

Fast side-task model and bring-up target for the shared Qwen3.5/3.6 kernels.
Dense hybrid: 24 layers (18 Gated DeltaNet + 6 gated full-attention, GQA 8/2,
head dim 256; DeltaNet 16 key / 16 value heads), hidden 2048, FFN 6144, vocab
248,320, tied embeddings, one MTP layer
([config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)).
Est. ~1.1 GB of weights read per decoded token at ~4.5 bits per weight.

Weights: Unsloth `Qwen3.5-2B-MTP-GGUF`, UD-Q4_K_XL, revision
`e05864f8066d874d5f85aaff007ae57a2a7d1efe` (the BF16 file of the same
revision for the HF oracle tests). FastFlowLM's own pack is
`FastFlowLM/Qwen3.5-2B-NPU2` (`qwen3.5:2b`).

NPU decode with MTP speculation and NPU prefill: see
[benchmarks](BENCHMARKS.md) and [quality](QUALITY.md).

## How a token runs

Orange runs on the NPU, blue on the CPU.

**Decode, one layer** (1 or 2 tokens per pass):

```mermaid
flowchart LR
  inp["Norm + input projection"]:::npu --> mix["Token mixer:<br/>Gated DeltaNet (18 layers)<br/>or gated attention (6 layers)"]:::cpu
  mix --> sub
  subgraph sub ["One chained NPU submission"]
    out["Output projection"]:::npu --> ffn["Residual + norm,<br/>gate + up, SwiGLU"]:::npu --> down["Down projection"]:::npu --> nxt["Residual + norm,<br/>next layer's input projection"]:::npu
  end
  nxt -.-> mix
  lm["LM head"]:::npu --> pick["Sample / check the MTP draft"]:::cpu
  classDef npu fill:#ffa94d,stroke:#d9480f,color:#000
  classDef cpu fill:#74c0fc,stroke:#1864ab,color:#000
```

*Qwen3.5-2B: 24 layers, hidden 2,048, FFN 6,144, GQA 8 query / 2 KV heads (head dim 256), DeltaNet 16 key / 16 value heads. Every weight matrix is a q4nx GEMV on the NPU; the CPU runs only the mixers (and DeltaNet's small alpha/beta rows), so a token is 25 NPU submissions (one per layer, plus one).*

**Prompt pass** (prompts of 7 tokens or more):

```mermaid
flowchart LR
  chunk["Prompt chunk<br/>(up to 1,024 tokens)"] --> proj["Projections as<br/>bfp16 GEMMs"]:::npu
  proj --> mixp["Norms, residuals, SwiGLU,<br/>DeltaNet + attention, batched"]:::cpu
  mixp --> proj
  mixp --> state["Decoder state filled<br/>(K/V, DeltaNet, MTP cache)"]
  state --> first["First token: LM head<br/>on the last row"]:::npu
  classDef npu fill:#ffa94d,stroke:#d9480f,color:#000
  classDef cpu fill:#74c0fc,stroke:#1864ab,color:#000
```

*With two chunks in flight, one chunk's GEMMs on the NPU overlap the CPU mixing the other.*
