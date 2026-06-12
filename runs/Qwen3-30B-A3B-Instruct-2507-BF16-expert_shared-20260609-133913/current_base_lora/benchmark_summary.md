# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 188952.2 | 3926.2 | 4.08 | 8.52 |
| no-lora | 32 | 190476.1 | 6901.9 | 4.64 | 9.84 |
| no-lora | 64 | 192230.1 | 11708.4 | 5.47 | 11.88 |
| lora | 16 | 33239.6 | 2537.3 | 6.31 | 13.90 |
| lora | 32 | 38679.5 | 4663.8 | 6.86 | 15.75 |
| lora | 64 | 39090.1 | 8043.7 | 7.96 | 19.65 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 17.6% | 64.6% | 163.2% |
| 32 | 20.3% | 67.6% | 160.0% |
| 64 | 20.3% | 68.7% | 165.4% |
