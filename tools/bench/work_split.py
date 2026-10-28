#!/usr/bin/env python3
"""How much of the work runs on the NPU and how much on the CPU.

Computed, not measured: FLOPs and bytes per op group from each model's
shapes, placed where gufo-npu runs the op today (src/models/qwen35,
src/models/qwen35_4b, src/models/qwen36_35b_a3b). Prints the per-model
tables in docs/models/*/BENCHMARKS.md ("Where the work runs") and the
FastFlowLM comparison.

Conventions:
- FLOPs: 2 per multiply-add; elementwise ops (norms, SiLU, gates, softmax,
  adds, argmax) 1 per element, transcendentals included.
- Decode bytes: weights, recurrent state and K/V cache read per step.
- Prefill bytes: each weight read once per prompt (the GEMM's own tiling
  re-reads are not counted).
- Formats: q4nx 4,736 bytes per 8,192 weights; bfp16 9 bits per weight; fp32
  4 bytes; the K/V cache fp16.
- MTP: a step verifies 2 tokens and drafts 1.8 (80% of drafts accepted,
  measured 72-93%).
- MoE prefill: the experts touched are the expected distinct experts under
  uniform routing (all 256 from ~200 tokens up).

  work_split.py
"""

Q4NX = 4736 / 8192
BFP16 = 9 / 8
F32 = 4
DRAFT_VOCAB = 32768
ACCEPT = 0.8

MODELS = {
    "Qwen3.5-2B": dict(d=2048, layers=24, ffn=6144, heads=8, kv=2, hd=256,
                       hk=16, hv=16, dk=128, vocab=248320, moe=False,
                       decode_ctx=(128, 2048, 9000),
                       prompts=(35, 1024, 4096)),
    "Qwen3.5-4B": dict(d=2560, layers=32, ffn=9216, heads=16, kv=4, hd=256,
                       hk=16, hv=32, dk=128, vocab=248320, moe=False,
                       decode_ctx=(128, 1024, 9000),
                       prompts=(35, 1024, 4096, 9000)),
    "Qwen3.6-35B-A3B": dict(d=2048, layers=40, ffn=None, heads=16, kv=2,
                            hd=256, hk=16, hv=32, dk=128, vocab=248320,
                            moe=True, experts=256, used=8, effn=512,
                            decode_ctx=(128, 1024, 9000),
                            prompts=(35, 125, 1055, 4246, 8499)),
}

GROUPS = ["dense projections", "experts", "LM head", "router", "DeltaNet",
          "attention", "norms/activations", "MTP draft", "sampling"]


def shapes(m):
    c = dict(m)
    c["key_dim"] = c["hk"] * c["dk"]
    c["value_dim"] = c["hv"] * c["dk"]
    c["conv_ch"] = 2 * c["key_dim"] + c["value_dim"]
    c["dn_in"] = c["conv_ch"] + c["value_dim"]  # q, k, v and z
    c["attn_q"] = c["heads"] * c["hd"]
    c["attn_in"] = 2 * c["attn_q"] + 2 * c["kv"] * c["hd"]  # q+gate, k, v
    c["mix_out"] = max(c["value_dim"], c["attn_q"])
    return c


def is_attention(il):
    return il % 4 == 3  # full_attention_interval 4


def add(work, group, device, flops=0.0, nbytes=0.0):
    f, b = work.get((group, device), (0.0, 0.0))
    work[(group, device)] = (f + flops, b + nbytes)


def decode_block(work, c, tokens, ctx, attention, mtp=False):
    """One block for `tokens` tokens at context `ctx`. Decode placement:
    projections with their fused norms, SwiGLU and residuals on the NPU;
    the mixer (with alpha/beta) and the MoE router on the CPU."""
    d = c["d"]

    def g(name):
        return "MTP draft" if mtp else name

    rows = c["attn_in"] if attention else c["dn_in"]
    for r, k in ((rows, d), (d, c["mix_out"])):
        add(work, g("dense projections"), "NPU", 2 * r * k * tokens, r * k * Q4NX)
    if c["moe"]:
        w = (c["used"] + 1) * 3 * c["effn"] * d
        width = (c["used"] + 1) * c["effn"]
        add(work, g("experts"), "NPU", 2 * w * tokens, w * Q4NX)
        rw = (c["experts"] + 1) * d  # router and shared-expert gate, fp32
        add(work, g("router"), "CPU", (2 * rw + 3 * d + 4 * c["experts"]) * tokens,
            rw * (2 if mtp else F32))  # the MTP layer's is bf16
    else:
        w = 3 * c["ffn"] * d
        width = c["ffn"]
        add(work, g("dense projections"), "NPU", 2 * w * tokens, w * Q4NX)
    add(work, g("norms/activations"), "NPU", (8 * d + 4 * width) * tokens)
    if attention:
        h, kv, hd = c["heads"], c["kv"], c["hd"]
        f = 4 * h * hd * ctx + 3 * h * ctx + 3 * (h + kv) * hd + 2 * h * hd
        add(work, g("attention"), "CPU", f * tokens, 2 * kv * hd * ctx * 2)
    else:
        hv, dk = c["hv"], c["dk"]
        f = (7 * hv * dk * dk  # decay, k.S, rank-1 update, q.S
             + 8 * c["conv_ch"]  # causal conv, kernel 4
             + 6 * c["key_dim"] + 5 * c["value_dim"]  # q/k L2 norms, gated norm
             + 4 * hv * d + 3 * d)  # alpha/beta from the normed input
        add(work, g("DeltaNet"), "CPU", f * tokens,
            hv * dk * dk * F32 + 4 * c["conv_ch"] * F32 + 2 * hv * d * F32)


def decode_step(c, ctx, mtp):
    """One decode step: a token, or with MTP a 2-token verify pass plus a
    draft over 1 + ACCEPT tokens (eh_proj, the MTP block and the 32k draft
    LM head on the NPU; its attention, router and argmax on the CPU)."""
    work = {}
    tokens = 2 if mtp else 1
    d = c["d"]
    for il in range(c["layers"]):
        decode_block(work, c, tokens, ctx, is_attention(il))
    add(work, "LM head", "NPU", 2 * c["vocab"] * d * tokens, c["vocab"] * d * Q4NX)
    add(work, "sampling", "CPU", c["vocab"] * tokens, c["vocab"] * F32 * tokens)
    if mtp:
        n = 1 + ACCEPT
        add(work, "MTP draft", "NPU", 2 * 2 * d * d * n, 2 * d * d * Q4NX)
        decode_block(work, c, n, ctx, attention=True, mtp=True)
        add(work, "MTP draft", "NPU", 2 * DRAFT_VOCAB * d * n, DRAFT_VOCAB * d * Q4NX)
        add(work, "MTP draft", "CPU", DRAFT_VOCAB, DRAFT_VOCAB * F32)
    return work


def experts_touched(c, tokens):
    e, k = c["experts"], c["used"]
    return e * (1 - (1 - k / e) ** tokens)


def prefill(c, n):
    """An n-token prompt through the NPU prefill. Prefill placement: the
    dense GEMMs on the NPU (bfp16); norms, SwiGLU, adds, mixers, alpha/beta,
    the router and the 35B's experts (q4nx, read in place) on the CPU. Then
    the first token's LM head on the NPU, the MTP cache over the prompt
    (eh_proj and the MTP block's input projection on the NPU) and one
    draft."""
    work = {}
    d = c["d"]
    for il in range(c["layers"]):
        attention = is_attention(il)
        rows = c["attn_in"] if attention else c["dn_in"]
        for r, k in ((rows, d), (d, c["mix_out"])):
            add(work, "dense projections", "NPU", 2 * r * k * n, r * k * BFP16)
        add(work, "norms/activations", "CPU", 8 * d * n)
        if c["moe"]:
            w = 3 * c["effn"] * d
            add(work, "experts", "CPU", 2 * (c["used"] + 1) * w * n,
                (experts_touched(c, n) + 1) * w * Q4NX)
            add(work, "norms/activations", "CPU", 4 * (c["used"] + 1) * c["effn"] * n)
            rw = (c["experts"] + 1) * d
            add(work, "router", "CPU", (2 * rw + 4 * c["experts"]) * n, rw * F32)
        else:
            w = 3 * c["ffn"] * d
            add(work, "dense projections", "NPU", 2 * w * n, w * BFP16)
            add(work, "norms/activations", "CPU", 4 * c["ffn"] * n)
        if attention:
            h, kv, hd = c["heads"], c["kv"], c["hd"]
            pairs = n * (n + 1) / 2  # causal, empty cache
            f = (4 * h * hd + 3 * h) * pairs + (3 * (h + kv) * hd + 2 * h * hd) * n
            add(work, "attention", "CPU", f)
        else:
            hv, dk = c["hv"], c["dk"]
            f = (7 * hv * dk * dk + 8 * c["conv_ch"] + 6 * c["key_dim"]
                 + 5 * c["value_dim"] + 4 * hv * d + 3 * d) * n
            add(work, "DeltaNet", "CPU", f, 2 * hv * d * F32)
    add(work, "LM head", "NPU", 2 * c["vocab"] * d, c["vocab"] * d * Q4NX)
    add(work, "sampling", "CPU", c["vocab"], c["vocab"] * F32)
    mtp_w = 2 * d * d + c["attn_in"] * d
    add(work, "MTP draft", "NPU", 2 * mtp_w * n, mtp_w * BFP16)
    add(work, "MTP draft", "CPU", 3 * c["kv"] * c["hd"] * 2 * n)
    for (group, device), (f, b) in decode_step(c, n, mtp=True).items():
        if group == "MTP draft":
            add(work, group, device, f / (1 + ACCEPT), b)
    return work


def flm_cpu_prefill_flops(c, n):
    """FastFlowLM v1.0.10's CPU work in prefill: the cpu_func calls of its
    prefill contexts (SwiGLU, expert weighting, top-k, sigmoid gates,
    alpha/beta activations, gated norm), per the engine binaries' symbols
    and call graph; the rest runs in its NPU programs."""
    d = c["d"]
    f = 0.0
    for il in range(c["layers"]):
        if c["moe"]:
            f += 4 * (c["used"] + 1) * c["effn"] + 2 * (c["used"] + 1) * d
            f += 4 * c["experts"] + 4
        else:
            f += 4 * c["ffn"]
        if is_attention(il):
            f += 4 * c["heads"] * c["hd"]
        else:
            f += 8 * c["hv"] + 5 * c["value_dim"]
    return f * n + c["vocab"]


def totals(work, i):
    npu = sum(v[i] for (_, dev), v in work.items() if dev == "NPU")
    cpu = sum(v[i] for (_, dev), v in work.items() if dev == "CPU")
    return npu, cpu


def pct(x):
    if x == 0:
        return "–"
    return "<0.1%" if x < 0.05 else f"{x:.1f}%"


def share(work, group, device, i):
    total = sum(v[i] for v in work.values())
    return 100 * work.get((group, device), (0, 0))[i] / total


def rows_for(columns, cell):
    for group in GROUPS:
        for device in ("NPU", "CPU"):
            if any((group, device) in w for _, w in columns):
                yield f"| {group} | {device} | " + " | ".join(cell(w, group, device) for _, w in columns)


def print_model(name, m):
    c = shapes(m)
    cols = [(f"{ctx:,} context" if ctx != 128 else "Short (~128 context)",
             decode_step(c, ctx, False)) for ctx in m["decode_ctx"]]
    cols.append(("Short, MTP step", decode_step(c, 128, True)))
    print(f"\n#### {name}: decode, per step (FLOPs / bytes read)\n")
    print("| Op group | Runs on | " + " | ".join(n for n, _ in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for row in rows_for(cols, lambda w, g, d: f"{pct(share(w, g, d, 0))} / {pct(share(w, g, d, 1))}"):
        print(row + " |")
    cells = []
    for _, w in cols:
        nf, cf = totals(w, 0)
        nb, cb = totals(w, 1)
        cells.append(f"**{100 * nf / (nf + cf):.1f}% / {100 * nb / (nb + cb):.1f}%**")
    print("| **NPU total** | | " + " | ".join(cells) + " |")
    cells = []
    for _, w in cols:
        cells.append(f"{sum(totals(w, 0)) / 1e9:.1f} GFLOP / {sum(totals(w, 1)) / 1e9:.2f} GB")
    print("| Work per step | | " + " | ".join(cells) + " |")

    cols = [(f"{n:,} tokens", prefill(c, n)) for n in m["prompts"]]
    longest = cols[-1][1]
    print(f"\n#### {name}: prefill, per prompt (FLOPs; weight bytes at {m['prompts'][-1]:,} tokens)\n")
    print("| Op group | Runs on | " + " | ".join(n for n, _ in cols) + " | Weight bytes |")
    print("|---|---|" + "---|" * len(cols) + "---|")
    for group in GROUPS:
        for device in ("NPU", "CPU"):
            if any((group, device) in w for _, w in cols):
                cells = [pct(share(w, group, device, 0)) for _, w in cols]
                print(f"| {group} | {device} | " + " | ".join(cells)
                      + f" | {pct(share(longest, group, device, 1))} |")
    cells = []
    for _, w in cols:
        nf, cf = totals(w, 0)
        cells.append(f"**{100 * nf / (nf + cf):.1f}%**")
    nb, cb = totals(longest, 1)
    print("| **NPU total** | | " + " | ".join(cells) + f" | **{100 * nb / (nb + cb):.1f}%** |")
    cells = [f"{sum(totals(w, 0)) / 1e12:.2f} TFLOP" for _, w in cols]
    print("| Work per prompt | | " + " | ".join(cells) + f" | {(nb + cb) / 1e9:.2f} GB |")

    flm = []
    for n, (_, w) in zip(m["prompts"], cols):
        flm.append(100 * flm_cpu_prefill_flops(c, n) / sum(totals(w, 0)))
    dec = decode_step(c, 128, False)
    print(f"\nFastFlowLM, same shapes: CPU at most {max(flm):.2f}% of prefill FLOPs, "
          f"{100 * c['vocab'] / sum(totals(dec, 0)):.3f}% of decode FLOPs (sampling).")


if __name__ == "__main__":
    for model_name, model in MODELS.items():
        print_model(model_name, model)
