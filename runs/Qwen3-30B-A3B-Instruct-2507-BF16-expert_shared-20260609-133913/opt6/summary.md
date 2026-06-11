# single×two matrix

- OFF = `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE=0`
- ON  = `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE=1`

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 38942.8 | 2121.1 | 16.3 |
| off | single | 32 | 39060.8 | 3962.1 | 18.2 |
| off | single | 64 | 39550.7 | 7028.3 | 22.0 |
| off | two | 16 | 39167.0 | 2571.1 | 13.6 |
| off | two | 32 | 39691.5 | 4701.3 | 15.6 |
| off | two | 64 | 39547.9 | 8103.2 | 19.5 |
| on | single | 16 | 37889.6 | 2117.1 | 16.3 |
| on | single | 32 | 39174.2 | 3963.5 | 18.2 |
| on | single | 64 | 38844.8 | 7048.6 | 22.0 |
| on | two | 16 | 38373.7 | 2581.6 | 13.5 |
| on | two | 32 | 39865.2 | 4722.7 | 15.5 |
| on | two | 64 | 40167.5 | 8071.4 | 19.5 |

## ON/OFF ratio @bs16 (prefill & decode >100% = faster)

| stream | prefill | decode | e2e |
|---|---|---|---|
| single | 97.3% | 99.8% | 100.3% |
| two | 98.0% | 100.4% | 99.7% |
