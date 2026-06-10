# single×two matrix — SGLANG_OPT_LORA_FUSED_MERGED_ALIGN off/on × stream

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 34910.1 | 2010.1 | 17.2 |
| off | single | 32 | 35569.7 | 3712.0 | 19.5 |
| off | single | 64 | 35386.8 | 6414.6 | 24.1 |
| off | two | 16 | 35392.1 | 2314.4 | 15.1 |
| off | two | 32 | 35912.7 | 4217.5 | 17.4 |
| off | two | 64 | 35687.7 | 7301.8 | 21.6 |
| on | single | 16 | 35207.2 | 2115.7 | 16.4 |
| on | single | 32 | 35779.6 | 3950.7 | 18.4 |
| on | single | 64 | 36376.0 | 6883.9 | 22.6 |
| on | two | 16 | 34895.6 | 2565.3 | 13.7 |
| on | two | 32 | 35167.0 | 4641.7 | 16.0 |
| on | two | 64 | 35797.9 | 7946.6 | 20.2 |

## decode tok/s matrix (rows=flag, cols=stream)

| flag \\ stream | single | two |
|---|---|---|
| off (bs16) | 2010.1 | 2314.4 |
| on (bs16) | 2115.7 | 2565.3 |

## opt effect (on/off) per stream, decode bs16
- single-stream: 105.3%
- two-stream:    110.8%
