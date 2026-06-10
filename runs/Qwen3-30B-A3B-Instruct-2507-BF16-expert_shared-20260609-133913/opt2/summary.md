# single×two matrix — SGLANG_OPT_LORA_FUSED_TOPK_PACK off/on × stream

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 34009.1 | 2046.3 | 17.0 |
| off | single | 32 | 33988.6 | 3780.0 | 19.3 |
| off | single | 64 | 34315.0 | 6684.1 | 23.4 |
| off | two | 16 | 33640.3 | 2437.2 | 14.4 |
| off | two | 32 | 34421.6 | 4429.1 | 16.7 |
| off | two | 64 | 34325.6 | 7707.2 | 20.8 |
| on | single | 16 | 33420.9 | 2117.9 | 16.5 |
| on | single | 32 | 35951.4 | 3923.0 | 18.5 |
| on | single | 64 | 36410.6 | 6942.3 | 22.5 |
| on | two | 16 | 34654.3 | 2555.1 | 13.8 |
| on | two | 32 | 35548.8 | 4619.9 | 16.0 |
| on | two | 64 | 35334.2 | 7946.5 | 20.2 |

## decode tok/s matrix (rows=flag, cols=stream)

| flag \\ stream | single | two |
|---|---|---|
| off (bs16) | 2046.3 | 2437.2 |
| on (bs16) | 2117.9 | 2555.1 |

## opt effect (on/off) per stream, decode bs16
- single-stream: 103.5%
- two-stream:    104.8%
