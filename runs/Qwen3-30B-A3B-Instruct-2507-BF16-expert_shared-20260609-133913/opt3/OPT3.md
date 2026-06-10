# opt3 — drop elem/upcast / `_get_lora_info` (rec#3 remainder)

**Result: no clear win — deltas are within run-to-run noise. rec#3's remainder is low-ROI; the
elem/copy bulk was already removed by opt2.**

Model: Qwen3-30B-A3B-Instruct-2507-BF16, expert_shared, GB300 TP4/EP4. sglang `eaaadc363`.

## What was tried (reframed to prefill)
Decode runs under cuda-graph, so `_get_lora_info` (pure-Python, per-layer) and the small elementwise
kernels barely touch the decode critical path. They *can* matter on the **eager prefill** path, so
opt3 targeted prefill via two levers:
1. **`SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1`** — vectorize the MoE activation elementwise kernel
   (scalar → 4-elem/thread; the bf16 launcher already wires it, it was just unset). This is the
   "drop/shrink elem" lever.
2. **lean `_get_lora_info`** (`SGLANG_OPT_LORA_LEAN_INFO`, default True, commit `eaaadc363`) — cache
   the layer-static scalars (num_experts / max_lora_rank / hidden_size) instead of recomputing them
   every layer-forward.

## Bench — single × two matrix (graph-ON) — `summary.md`, `opt3_matrix.png`
A/B = `SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC` `0` vs `1` (lean `_get_lora_info` on in both).
| @bs16 | prefill off→on | decode off→on |
|---|---|---|
| single | 35066 → 35395 (+0.9%) | 2115.5 → 2115.2 (0%) |
| two | 34853 → 35871 (+2.9%) | 2551 → 2595 (+1.7%) |

**But the prefill spread across cells is ~4–8% (one cell measured 33881 vs ~36000 elsewhere), so the
+0.9–2.9% "gains" are not distinguishable from noise.** Decode is ~0–1.7%. → **no clear win.**

## Why (confirms the original scoping)
- **opt2 already removed the elem/copy bulk**: the fused topk+pack deleted the pack's cast/shift/or
  chain (`copy_` 1309→157, `BinaryFunctor` 576→0). What remained is mostly legitimate compute.
- The **activation-vec** saving is real but small relative to total prefill GEMM time at this config,
  so it doesn't move `input_throughput` beyond noise.
- **`_get_lora_info`** is microseconds of CPU per layer; trimming it is below the measurement floor.
- The copies that *do* remain live inside the decomposed bf16 `.cu` op (intermediate materialization)
  — removing those is the **in-MoE fold** (the big ② item, ~25 µs/layer), not a cheap elem cleanup.

## Verdict
opt3 as scoped (cheap elem/upcast/`_get_lora_info`) is **effectively subsumed by opt2** and not worth
shipping as a standalone win. The lean `_get_lora_info` change is harmless (kept, default-on);
`SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1` is safe (bitwise-identical) and fine to enable but its
measured benefit here is within noise. **The remaining real headroom is the in-MoE fold.**

## Artifacts
- `summary.md` — full single×two matrix (bs16/32/64 × off/on × single/two)
- `opt3_matrix.png` — prefill + decode off/on (deltas within noise)
- `profile/{single_on,two_on}/bs16-TP-0.trace.json.gz` — eager graph-OFF traces (flag on)
