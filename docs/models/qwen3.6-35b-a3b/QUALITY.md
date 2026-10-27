# Qwen3.6-35B-A3B quality

No checks yet. Planned contracts (see [the plan](../../PLAN.md)):

| Check | Tolerance |
|---|---|
| CPU reference vs HF transformers, teacher-forced logits | greedy argmax match |
| Each NPU op vs CPU reference | relative error < 1e-2 |
| NPU model vs CPU reference, teacher-forced | greedy argmax match |
| Prefill vs one-token decode | identical next token |
| MTP head vs reference | cosine > 0.999 |
