# opt8 step0 probe — --enforce-piecewise-cuda-graph off vs on (LoRA cell)

| variant | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| off | 16 | 39121.9 | 2596.4 | 6.2 | 13.5 |
| off | 32 | 39684.9 | 4693.9 | 6.8 | 15.6 |
| off | 64 | 39435.6 | 8100.5 | 7.9 | 19.5 |
| on | 16 | 24199.6 | 2595.8 | 6.2 | 14.0 |
| on | 32 | 24406.0 | 4710.7 | 6.8 | 16.6 |
| on | 64 | 24444.4 | 8077.2 | 7.9 | 21.6 |

on/off ratio (prefill & decode: higher=faster; e2e: lower=faster)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 61.9% | 100.0% | 103.9% |
| 32 | 61.5% | 100.4% | 106.3% |
| 64 | 62.0% | 99.7% | 110.7% |
