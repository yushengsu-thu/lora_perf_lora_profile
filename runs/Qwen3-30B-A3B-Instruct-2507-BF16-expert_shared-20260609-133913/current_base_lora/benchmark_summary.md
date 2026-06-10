# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 188141.8 | 3932.9 | 4.07 | 8.51 |
| no-lora | 32 | 192148.2 | 6919.6 | 4.62 | 9.81 |
| no-lora | 64 | 193851.7 | 11713.6 | 5.46 | 11.87 |
| lora | 16 | 35750.7 | 2574.1 | 6.22 | 13.65 |
| lora | 32 | 36330.5 | 4685.9 | 6.83 | 15.79 |
| lora | 64 | 36646.8 | 8088.2 | 7.91 | 19.78 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 19.0% | 65.5% | 160.4% |
| 32 | 18.9% | 67.7% | 160.9% |
| 64 | 18.9% | 69.0% | 166.7% |
