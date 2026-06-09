# opt1 A/B — SGLANG_OPT_LORA_FUSED_MERGED_ALIGN off vs on (LoRA cell, shared_outer)

| variant | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| off | 16 | 35925.4 | 2315.2 | — | 15.1 |
| off | 32 | 35381.7 | 4228.4 | — | 17.4 |
| off | 64 | 35456.1 | 7290.2 | — | 21.7 |
| on | 16 | 35680.6 | 2568.8 | — | 13.7 |
| on | 32 | 35788.8 | 4646.9 | — | 15.9 |
| on | 64 | 36388.7 | 7929.1 | — | 20.1 |

on/off ratio (prefill & decode: >100%=opt1 faster; e2e: <100%=opt1 faster)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 99.3% | 111.0% | 90.8% |
| 32 | 101.2% | 109.9% | 91.8% |
| 64 | 102.6% | 108.8% | 92.9% |
