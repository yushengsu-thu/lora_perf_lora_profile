# single×two matrix — opt5 prefill routing reuse

- OFF = `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE=0`
- ON  = `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE=1`

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 35268.9 | 2124.5 | 16.35 |
| off | single | 32 | 36086.4 | 3962.0 | 18.36 |
| off | single | 64 | 36421.0 | 7053.9 | 22.18 |
| off | two | 16 | 36070.4 | 2583.0 | 13.59 |
| off | two | 32 | 36262.6 | 4704.1 | 15.74 |
| off | two | 64 | 35995.4 | 8073.9 | 19.88 |
| on | single | 16 | 38175.9 | 2123.5 | 16.29 |
| on | single | 32 | 38947.5 | 3960.2 | 18.23 |
| on | single | 64 | 39134.2 | 7049.1 | 21.94 |
| on | two | 16 | 39470.6 | 2573.2 | 13.56 |
| on | two | 32 | 40013.9 | 4685.6 | 15.62 |
| on | two | 64 | 39993.0 | 8091.3 | 19.48 |

## ON/OFF ratio (prefill & decode >100% = faster; e2e <100% = faster)

| stream | bs | prefill | decode | e2e |
|---|---|---|---|---|
| single | 16 | 108.2% | 100.0% | 99.6% |
| single | 32 | 107.9% | 100.0% | 99.3% |
| single | 64 | 107.4% | 99.9% | 98.9% |
| two | 16 | 109.4% | 99.6% | 99.8% |
| two | 32 | 110.3% | 99.6% | 99.3% |
| two | 64 | 111.1% | 100.2% | 98.0% |
