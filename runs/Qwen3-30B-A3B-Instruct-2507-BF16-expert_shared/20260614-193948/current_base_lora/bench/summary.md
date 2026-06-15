# bench summary — LoRA vs no-LoRA

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| no-lora | 16 | 186496.9 | 3938.7 | 4.06 | 8.50 |
| no-lora | 32 | 192319.0 | 6936.7 | 4.61 | 9.79 |
| no-lora | 64 | 193240.8 | 11770.5 | 5.44 | 11.81 |
| lora | 16 | 38850.9 | 2578.4 | 6.21 | 13.55 |
| lora | 32 | 39820.0 | 4680.6 | 6.84 | 15.65 |
| lora | 64 | 39952.6 | 8093.9 | 7.91 | 19.47 |

lora / no-lora ratio  (prefill & decode tok/s: higher=faster; e2e latency: higher=slower)

| bs | prefill | decode | e2e |
|---|---|---|---|
| 16 | 20.8% | 65.5% | 159.5% |
| 32 | 20.7% | 67.5% | 159.9% |
| 64 | 20.7% | 68.8% | 164.8% |
