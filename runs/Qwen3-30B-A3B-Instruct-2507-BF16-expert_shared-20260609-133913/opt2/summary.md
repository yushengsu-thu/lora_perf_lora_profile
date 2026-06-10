# opt2 A/B — SGLANG_OPT_LORA_FUSED_TOPK_PACK off vs on (LoRA cell, shared_outer; opt1 align held on)

| variant | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|
| off | 16 | 34160.6 | 2422.0 | 14.5 |
| off | 32 | 34359.2 | 4453.1 | 16.6 |
| off | 64 | 34845.8 | 7689.8 | 20.8 |
| on | 16 | 34560.5 | 2558.6 | 13.8 |
| on | 32 | 33628.7 | 4612.0 | 16.2 |
| on | 64 | 36080.1 | 7924.8 | 20.2 |

on/off ratio (decode >100% = opt2 faster; e2e <100% = faster)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 101.2% | 105.6% | 94.9% |
| 32 | 97.9% | 103.6% | 97.2% |
| 64 | 103.5% | 103.1% | 97.0% |
