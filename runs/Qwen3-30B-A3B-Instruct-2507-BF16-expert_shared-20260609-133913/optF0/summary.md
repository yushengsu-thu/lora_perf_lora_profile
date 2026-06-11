# single×two matrix

- OFF = `SGLANG_TWO_STREAM_MAX_TOKENS=256`
- ON  = `SGLANG_TWO_STREAM_MAX_TOKENS=8192`

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 35834.6 | 2123.2 | 16.3 |
| off | single | 32 | 35897.8 | 3962.3 | 18.4 |
| off | single | 64 | 36359.9 | 7051.3 | 22.2 |
| off | two | 16 | 35176.4 | 2572.9 | 13.7 |
| off | two | 32 | 36307.5 | 4694.8 | 15.8 |
| off | two | 64 | 36042.8 | 8114.4 | 19.8 |
| on | single | 16 | 35827.0 | 2130.3 | 16.3 |
| on | single | 32 | 36569.2 | 3981.4 | 18.3 |
| on | two | 16 | 32153.5 | 2578.2 | 13.7 |
| on | two | 32 | 33154.7 | 4687.8 | 16.0 |
| on | two | 64 | 33341.5 | 8071.6 | 20.2 |

## ON/OFF ratio @bs16 (prefill & decode >100% = faster)

| stream | prefill | decode | e2e |
|---|---|---|---|
| single | 100.0% | 100.3% | 99.7% |
| two | 91.4% | 100.2% | 100.4% |
