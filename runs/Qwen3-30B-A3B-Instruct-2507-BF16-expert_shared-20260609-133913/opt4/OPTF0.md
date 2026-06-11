# optF0 — two-stream-at-prefill A/B (flag-only) — **NEGATIVE: do not adopt**

**Question** (journal_opti.md §5 F0): prefill is always serial today
(`SGLANG_TWO_STREAM_MAX_TOKENS` defaults 256 < 4096-token prefill chunks), so the whole
LoRA-Δ chain (~345 µs/layer incl. re-sort) sits on the main stream. Does raising the gate
to 8192 — letting prefill chunks take the two-stream path — hide it behind the main GEMMs?

**Answer: no — prefill gets ~8% SLOWER; keep the 256 default.**

## Setup
- A/B: `SGLANG_TWO_STREAM_MAX_TOKENS=256` (off) vs `=8192` (on), all other flags = current
  optimized defaults. Matrix driver `dev/bench_profile_matrix.sh <model> optF0 …`.
- In the **single** column the driver appends `…=0` after the delta (last-wins), so both
  single cells are identical serial runs → it serves as the **measured noise floor**.
- The **two** column is the real A/B: prefill serial (256) vs prefill two-streamed (8192);
  decode is two-stream in both cells.
- bs16/32/64, in=out=2048, bf16 Qwen3-30B-A3B expert_shared r32, GB300 TP4/EP4, cuda-graph.

## Result

| cell | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|
| off_single | 16 | 35,835 | 2,123.2 | 16.35 |
| off_single | 32 | 35,898 | 3,962.3 | 18.37 |
| off_single | 64 | 36,360 | 7,051.3 | 22.19 |
| on_single | 16 | 35,827 | 2,130.3 | 16.30 |
| on_single | 32 | 36,569 | 3,981.4 | 18.25 |
| off_two | 16 | 35,176 | 2,572.9 | 13.67 |
| off_two | 32 | 36,308 | 4,694.8 | 15.76 |
| off_two | 64 | 36,043 | 8,114.4 | 19.79 |
| **on_two** | 16 | **32,153** | 2,578.2 | 13.73 |
| **on_two** | 32 | **33,155** | 4,687.8 | 15.96 |
| **on_two** | 64 | **33,342** | 8,071.6 | 20.17 |

ON/OFF ratio (>100% = faster):

| stream | bs | prefill | decode | e2e |
|---|---|---|---|---|
| single (noise floor) | 16 | 100.0% | 100.3% | 99.7% |
| single (noise floor) | 32 | 101.9% | 100.5% | 99.4% |
| **two (real A/B)** | 16 | **91.4%** | 100.2% | 100.4% |
| **two (real A/B)** | 32 | **91.3%** | 99.9% | 101.2% |
| **two (real A/B)** | 64 | **92.5%** | 99.5% | 101.9% |

(on_single bs64 jsonl missing — bench-client hiccup; the noise-floor column is established
by bs16/32. Profiles: `profile/{single,two}_on/bs16-TP-0.trace.json.gz`, graph-off.)

## Interpretation
1. The noise floor (single column, identical cells) is ±0–2%; the two-column prefill delta
   of **−8~9% is real and consistent across all three batch sizes**.
2. Two-streaming the 4096-token prefill chunks adds side-stream event/sync overhead and SM
   contention that exceeds any overlap benefit. Decode is unaffected (~100%) as expected —
   both cells run decode two-stream.
3. This **confirms the prefill bottleneck is not serialization**: rearranging work
   (overlap) loses; work must be *removed* (kernels + launches). Consistent with the
   host-bound finding (≈half of prefill wall is GPU-idle, journal §4). **F1's priority is
   strengthened.**

## Verdict
Keep `SGLANG_TWO_STREAM_MAX_TOKENS=256` (default). No code or model.env change.
Next: F1-① (`SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE`, sglang commit `aabdaaafa`).
