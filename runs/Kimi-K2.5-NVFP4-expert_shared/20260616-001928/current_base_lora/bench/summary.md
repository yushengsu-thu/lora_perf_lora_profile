# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 54733.7 | 1246.5 | 12.84 | 26.89 |
| lora | 16 | 20530.0 | 879.5 | 18.19 | 38.85 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 37.5% | 70.6% | 144.5% |
