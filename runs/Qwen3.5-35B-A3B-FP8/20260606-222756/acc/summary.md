# acc summary — teacher-forced prefill logprobs

## sglang: lora vs no-lora (per-token |diff|)

| n | mean abs | max abs | p50 | p95 | half-MSE | tol | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 31999 | 0.07241 | 5.53974 | 0.00055 | 0.45016 | 0.03328 | 0.05 | EXCEEDS tol — LoRA-path numerical regression OR an intentionally divergent adapter |

## vs vLLM/trainer reference (KL = 0.5·mean((a−b)²), from the .pt)

| pair | KL | meaning |
|---|---:|---|
| orig_sampler (vLLM) vs trainer | 0.000584 | inherent noise floor |
| sglang-lora vs trainer | 0.518013 | **above the vLLM noise floor — inspect** |
| sglang-lora vs orig_sampler (vLLM) | 0.518686 | direct sglang↔vLLM gap |

> Prefill-only: decode health is gated separately (coherence check per cell).
