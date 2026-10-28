# Qwen3.5-2B experiments

Short retained/rejected decisions; the runs behind them are in
[BENCHMARKS.md](BENCHMARKS.md) and [QUALITY.md](QUALITY.md).

| Experiment | Decision / qualification |
| --- | --- |
| Compact q4nx scales (4736-byte blocks, 4.625 bits per weight) | Retained: 7.5% fewer weight bytes. Sky without drafting 36.9–37.2 → 38.6–39.1 tok/s, Fibonacci with MTP 60.6–61.1 → 63.0–63.7. Rounding d instead of d · sc moves the weights slightly, so greedy replies differ a little. |
| fp16 K/V cache | Retained: halves the cache read per token. 9k context without drafting 28.6–28.7 → 31.6–31.9 tok/s, decoder logits within the 2e-3 tolerance (6.2e-4). |
| 32k draft vocabulary for the MTP head | Retained: the draft's LM head scores 32,768 frequent ids instead of 248,320. 2k prompt then the sky question, with MTP: 41.4–41.7 → 46.5–47.3 tok/s. |
| GEMV residual drain awaited first (DMA wait order) | Fixed: a wait took another task's completion token, about once in 30,000 chained calls (24 mismatching buffers in 200,000 runs → 0). `q4nx_gemv_chain_test` guards it. |
| Instruction-buffer flow (the 35B's patchable calls) | Rejected for this model: same output hashes, but sky 39.0–39.1 → 38.1–38.3 and Fibonacci 63.1–63.5 → 61.8–62.1 tok/s. The 2B keeps the ELF flow. |
| Prompt processing on bfp16 GEMMs | Retained: time to first token 0.11 s at 35 tokens and 1.23 s at 4k (the decode path took 33 s at 2,120 tokens), for ~1.56 GB more pinned memory. Logits 1.0–3.3e-2 from the decode path; first generated token always matches. |
| Search fit for requantized weights (`q4nx::Fit::kSearch`, which the 35B and 4B use) | Rejected for this model; it keeps `kMinMax` (`qwen35::Q4nxFit`). Mixed on the 2B: on 1,515 held-out positions KL against the GGUF falls 22–23%, top-1 rises 0.33 points without the LM head but falls 0.40 points (6 positions) with every matrix in q4nx, the decode set. NPU decode vs CPU 73/73, vs HF 61/73 (63/73 before); hashes change, speed unchanged (sky 38.9–39.3, Fibonacci MTP 62.7–62.8 tok/s); load 2.6 s instead of 1.3 s. A clear win on every measure was the bar, and the decode set's top-1 is not. Importance weighting (below in the 35B's EXPERIMENTS.md) is the next candidate for this model too. | See [the 35B's measurements](../qwen3.6-35b-a3b/QUALITY.md#choosing-the-4-bit-values). |
