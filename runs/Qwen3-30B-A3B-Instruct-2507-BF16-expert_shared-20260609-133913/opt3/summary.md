# single×two matrix — SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC off/on × stream

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 35065.6 | 2115.5 | 16.4 |
| off | single | 32 | 36411.7 | 3936.5 | 18.4 |
| off | single | 64 | 36211.8 | 6924.1 | 22.5 |
| off | two | 16 | 34853.0 | 2551.2 | 13.8 |
| off | two | 32 | 36005.6 | 4631.7 | 16.0 |
| off | two | 64 | 35117.7 | 7880.8 | 20.4 |
| on | single | 16 | 35395.1 | 2115.2 | 16.4 |
| on | single | 32 | 36570.3 | 3954.8 | 18.4 |
| on | single | 64 | 36192.7 | 7053.5 | 22.2 |
| on | two | 16 | 35871.3 | 2595.1 | 13.5 |
| on | two | 32 | 33881.0 | 4698.4 | 15.9 |
| on | two | 64 | 36596.9 | 8070.3 | 19.8 |

## decode tok/s matrix (rows=flag, cols=stream)

| flag \\ stream | single | two |
|---|---|---|
| off (bs16) | 2115.5 | 2551.2 |
| on (bs16) | 2115.2 | 2595.1 |

## opt effect (on/off) per stream, decode bs16
- single-stream: 100.0%
- two-stream:    101.7%
