# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 55971.2 | 1246.8 | 12.83 | 26.87 |
| lora | 16 | 21172.5 | 887.7 | 18.02 | 38.46 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 37.8% | 71.2% | 143.2% |
