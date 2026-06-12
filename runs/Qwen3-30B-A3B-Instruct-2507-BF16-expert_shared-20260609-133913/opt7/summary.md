# single×two matrix

- OFF = `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT=0 SGLANG_OPT_BF16_MOE_GEMM1_FOLD=0`
- ON  = `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT=1 SGLANG_OPT_BF16_MOE_GEMM1_FOLD=1`

| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|---|
| off | single | 16 | 38837.8 | 2132.4 | 16.2 |
| off | single | 32 | 39638.3 | 3959.2 | 18.2 |
| off | single | 64 | 40090.2 | 7060.4 | 21.8 |
| off | two | 16 | 38718.7 | 2597.4 | 13.5 |
| off | two | 32 | 39658.9 | 4702.1 | 15.6 |
| off | two | 64 | 40059.7 | 8038.6 | 19.6 |
| on | single | 16 | 39860.3 | 2116.6 | 16.3 |
| on | single | 32 | 39614.8 | 3960.3 | 18.2 |
| on | single | 64 | 39861.8 | 7064.2 | 21.8 |
| on | two | 16 | 39189.4 | 2571.1 | 13.6 |
| on | two | 32 | 39702.7 | 4683.0 | 15.6 |
| on | two | 64 | 39678.9 | 8137.0 | 19.4 |

## ON/OFF ratio @bs16 (prefill & decode >100% = faster)

| stream | prefill | decode | e2e |
|---|---|---|---|
| single | 102.6% | 99.3% | 100.6% |
| two | 101.2% | 99.0% | 100.9% |
