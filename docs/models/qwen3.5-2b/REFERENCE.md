# Qwen3.5 forward pass (GGUF arch `qwen35`)

The math `src/models/qwen35/reference.cpp` implements, with GGUF tensor
layouts. Cross-checked against gufo's CPU reference, llama.cpp
(`conversion/qwen.py`, `src/models/qwen35.cpp`, `delta-net-base.cpp`) and HF
transformers `modeling_qwen3_5.py` (5.21.0). The MoE FFN and untied LM head
of arch `qwen35moe` are in
[../qwen3.6-35b-a3b/README.md](../qwen3.6-35b-a3b/README.md).

GGUF matrices `[ne0, ne1]` are ne1 rows of ne0 contiguous elements:
`y[r] = Σ_c W[r][c] x[c]`. `RMS(x) = x / sqrt(mean(x²) + eps)`, eps = 1e-6.

## Layers

- Layer `il` is full attention when `(il + 1) % full_attention_interval == 0`
  (2B: layers 3, 7, 11, 15, 19, 23); the rest are Gated DeltaNet.
- Blocks after `block_count - nextn_predict_layers` are MTP layers, not run in
  the main pass.
- Each layer: `h += Mixer(RMS(h) * attn_norm)`, then
  `h += ffn_down(silu(ffn_gate n) ⊙ ffn_up n)` with
  `n = RMS(h) * post_attention_norm`.
- Output: `logits = token_embd · (RMS(h) * output_norm)` (tied embeddings).

## Norm weights

The converter stores `1 + w` for every `*norm.weight` except `ssm_norm`
(attn_norm, post_attention_norm, output_norm, attn_q_norm, attn_k_norm,
nextn.enorm/hnorm/shared_head_norm). So every norm is `RMS(x) * w_gguf`;
`ssm_norm` is also `RMS(x) * w` with its raw weight.

## Gated DeltaNet

1. `qkv = attn_qkv · x` → q `[0, Kd)`, k `[Kd, 2Kd)`, v `[2Kd, 2Kd + Vd)`;
   head h is `[h·128, (h+1)·128)`. `z = attn_gate · x`,
   `b = ssm_beta · x`, `a = ssm_alpha · x` (one value per value head).
2. Causal depthwise conv over `qkv`, kernel 4, weight `ssm_conv1d[c][t]`,
   t = 3 is the current token, zero history at sequence start; then SiLU.
3. `β = sigmoid(b)`, `g = ssm_a · softplus(a + ssm_dt.bias)` where
   **`ssm_a` already holds `-exp(A_log)`**; decay `exp(g)`. Softplus returns x
   for x > 20.
4. `k̂ = k / sqrt(Σk² + 1e-6)`, `q̂ = q / sqrt(Σq² + 1e-6) / sqrt(128)`.
5. Value head h uses key head `h % Hk` (GGUF tiled order; the converter
   reorders V-side tensors, so stay in GGUF order end to end). The 2B has
   Hv = Hk; Qwen3.6-35B-A3B (Hv = 2 Hk) checks this order against llama.cpp.
6. State `S[i][j]` (key i, value j), per value head:
   `S ← exp(g) S`; `u = Sᵀ k̂`; `S += k̂ ⊗ β (v − u)`; `o = Sᵀ q̂`.
7. `y_h = RMS(o_h) * ssm_norm * silu(z_h)`; `out = ssm_out · y`.

Chunked (prefill) forms are algebraically equal to the per-token loop.

## Gated full attention

- `attn_q · x` gives `2 · heads · 256` values; head h is
  `[query 256, gate 256]` at offset `h · 512`.
- q and k heads: `RMS(·) * q_norm / k_norm` (256 wide), then RoPE.
- RoPE on dims `[0, 64)` of each head, NeoX half split, θ_i =
  pos · 1e7^(−2i/64): `x'[i] = x[i] cos − x[i+32] sin`,
  `x'[i+32] = x[i] sin + x[i+32] cos`. For text, the MRoPE sections all see
  the same position, so this is plain RoPE.
- GQA: query head h uses kv head `h / (heads / kv_heads)`. Scores
  `q·k / sqrt(256)`, causal softmax, `ctx = Σ p v`, then
  `ctx ⊙ sigmoid(gate)`, then `attn_output`.

## MTP layer (blk.24 on the 2B)

`x = eh_proj · [RMS(embed(next)) * enorm ; RMS(h) * hnorm]`, where h is the
main model's hidden state **after output_norm**; then one full-attention
block with its own KV cache; then `RMS(x) * shared_head_norm` and the tied
LM head. HF transformers drops MTP weights, so it has no oracle; the draft
position index is still to be confirmed.
