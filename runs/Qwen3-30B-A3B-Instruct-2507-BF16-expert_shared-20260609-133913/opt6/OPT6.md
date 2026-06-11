# opt6 — drop redundant `activation_lora_input` side-capture (bf16-only) — ✗ NO CLEAR WIN

**Commits**: [`cf9d0e55e`](https://github.com/yushengsu-thu/sglang/commit/cf9d0e55e) (mechanism)
+ [`f4971ea4e`](https://github.com/yushengsu-thu/sglang/commit/f4971ea4e) (default OFF after
measurement) on PR #4 branch · **Flag**: `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE`
(**default False** — off = byte-identical to the pre-opt6 path).

## What
For bf16 the activation kernel's expanded-layout `activation_lora_input` side-capture holds
the **same values** as the permuted `activated` buffer (verified in the dev kernel: the same
packed value is written to both; the fp8 variant divides by `scaleOut`, which is why fp8/fp4
*must* keep the capture). The mechanism: pass `activated_out` (permuted) +
`expanded_to_permuted_out` (routing map, exported D2D) instead; the down-LoRA shrink kernel
translates expanded ids to permuted rows via an optional `a_row_map` (invalid −1 rows
contribute zeros, matching the old zero-filled capture rows). Output indexing unchanged.
Prefill-only (≥512 tokens); decode keeps the capture (two-stream needs the side buffer).

Expected: −50 MB/layer HBM write (≈half the activation kernel's write traffic) + one buffer.

## Result — within noise, no consistent direction

| stream | bs | prefill ON/OFF | decode | e2e |
|---|---|---|---|---|
| single | 16 | 97.3% | 99.8% | 100.3% |
| single | 32 | 100.3% | 100.0% | 99.9% |
| single | 64 | 98.2% | 100.3% | 100.0% |
| two | 16 | 98.0% | 100.4% | 99.7% |
| two | 32 | 100.4% | 100.5% | 99.5% |
| two | 64 | 101.6% | 99.6% | 100.1% |

Noise floor (opt4's identical-cell column): ±0–2%. Verdict: **no clear win.**

## Why the honest accounting says this was expected
- The write saving is ~13 µs/layer (33→~20 µs activation) × 384 layer-forwards ≈ **5 ms per
  917 ms prefill ≈ 0.5%** — below the noise floor. (The earlier "+2~4%" hope was wrong; the
  per-layer µs math was always <1%.)
- The map D2D export **adds one launch per layer** on the ~50% host-bound prefill path,
  eating part of the small kernel-side gain.

## Correctness + mechanism proof (the mechanism itself is sound)
- acc with the path ON: KL vs trainer **0.003530** ≈ vLLM noise floor 0.004243 — the
  map-translated shrink produces correct logprobs on the real prefill path. Coherence passed.
- In-pod trace check (graph-off, ON, single): **activation kernel 32.9 → 21.7 µs/call (−34%)**
  — matching the "half the write traffic" prediction almost exactly; the shrink's map
  indirection costs ~0 (21.7 vs 21.5 µs/call); the exported row map shows up as +384 small
  DtoD copies. The mechanism works precisely as designed — the total is just too small
  (~5 ms/prefill) to clear the bench noise floor.

## Disposition
- Flag **default False** → production behavior byte-identical to pre-opt6.
- The code stays: the **permuted-buffer + exported-row-map pipeline is exactly what the opt7
  fold needs** (its EVT epilogue writes only `activated`; the down shrink must then read it
  via this map). opt6 de-risked that plumbing and proved its numerics at the noise floor.
- FP8/NVFP4: untouched (optional params default to the old behavior).

## Files
`summary.md` (matrix), `bench/{off,on}_{single,two}/bs{16,32,64}.jsonl`.
Profiles not uploaded (flaky uplink; per-run decision) — traces remain on the pod at
`/tmp/mx/prof/{single,two}/`; the mechanism check (capture write gone, activation kernel
smaller) was run in-pod.
