# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 269123.5 | 3965.7 | 4.03 | 8.38 |
| lora | 16 | 81839.4 | 2564.1 | 6.24 | 13.18 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 30.4% | 64.7% | 157.2% |
