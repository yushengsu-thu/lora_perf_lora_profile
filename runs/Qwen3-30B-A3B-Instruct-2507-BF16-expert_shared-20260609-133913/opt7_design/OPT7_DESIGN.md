# opt7 — the in-MoE fold: design (route b, CUTLASS grouped GEMM)

Status: **design + skeleton** (2026-06-12). Probe (`opt7_step0/PROBE.md`) confirmed route (a)
is dead: no unfused-activation gated GEMM1 cubin exists for bf16 at any tile, in either
weight layout. The fold is therefore a **new bf16-only CUTLASS grouped GEMM** replacing
`permute + GEMM1 + activation` in `Bf16LoraLauncher`.

## 1. Targets (measured, per layer-forward @ prefill 4096-tok chunk)

| removed | cost today |
|---|---|
| standalone `moe::dev::permute` (gather → HBM round-trip) | **180 µs** |
| `gate_up` [P,1536] HBM write+read between GEMM1 and activation | (inside GEMM1/act) |
| standalone activation kernel | 33 µs |
| **replaced by** | one grouped GEMM ≥ **57 µs** (tuned `bmm_Bfloat16` parity bar) |

Net prefill MoE target: **695 → ~360 µs/layer (−48%)** + 2 launches/layer (host-bound
prefill compounds). Decode win is small (~5.6 µs/layer) and NOT the goal: decode keeps the
tuned trtllm path (see §6 dual-layout de-risk).

## 2. Kernel contract (exact semantics — copied from the dev activation kernel)

Per permuted output row `r` (valid when `permuted_row_to_expert[r] >= 0`) and hidden index
`h ∈ [0, I)`, with `t = permuted_idx_to_token_idx[r]`, `x = permuted_idx_to_expanded_idx[r]`:

```
acc[2h]   = Σ_k hidden[t, k] · W[e, 2h,   k]      # interleaved gate/up columns (g0,u0,g1,u1,…)
acc[2h+1] = Σ_k hidden[t, k] · W[e, 2h+1, k]
x1 = acc[2h]   + Δ[x·2I + I + h]                  # Δ is HALF-CONTIGUOUS: 2nd half → x1
x2 = acc[2h+1] + Δ[x·2I     + h]                  #                       1st half → x2
activated[r, h] = silu(x2) · x1                   # silu(x) = x / (1 + e^(−x))
```

CRITICAL asymmetry: the GEMM output is **column-interleaved** (pairs 2h/2h+1) but the LoRA Δ
is **half-contiguous** ([x, 0..I) → x2-half, [x, I..2I) → x1-half). Getting this wrong is the
#1 correctness trap; the reference kernel encodes it and the unit test pins it.

Invalid rows (padding, `expert = -1`): skip (never read by finalize/down GEMM — but see §5:
GEMM2 consumes padded rows too, so either zero them or rely on the same uninitialized-padding
contract the current path uses; the ref kernel zeroes for determinism, CUTLASS phase decides).

Down-LoRA shrink input: the **opt6 plumbing** (read `activated` permuted buffer via the
exported `expanded_to_permuted` row map) — already implemented, proven at the acc noise
floor, default-off; opt7 turns it on as part of the fold path (no extra capture write).

## 3. Inputs / routing metadata (all already produced by trtllm-gen routing)

| tensor | shape | producer | use |
|---|---|---|---|
| `hidden_states` | [N, K=2048] bf16 | layer input | A-operand via **gather** |
| `W_fold` (NEW) | [E_local, 2I=1536, K] bf16 row-major | load-time re-prep (§4) | B-operand |
| `gate_up_lora_delta` | [N, top_k, 2I] bf16 | Triton LoRA-Δ chain (unchanged) | epilogue aux |
| `permuted_idx_to_token_idx` | [max_padded] i32 | routing | gather index (A rows) |
| `permuted_idx_to_expanded_idx` | [max_padded] i32 | routing (currently passed nullptr — just wire it) | Δ row index |
| `cta_idx_xy_to_batch_idx` / `cta_idx_xy_to_mn_limit` / `num_non_exiting_ctas` | [maxCtas] i32 | routing | CUTLASS per-CTA expert mapping (phase ≥1) |
| `permuted_row_to_expert` | [max_padded] i32 | trivial derivation | ref-kernel only |
| out: `activated` | [max_padded, I=768] bf16 | — | GEMM2 input + down-shrink (opt6 map) |

## 4. Weight re-prep (dual-layout)

`quant_info.gemm1_weights` today = trtllm shuffled + BlockMajorK (cubin-specific). The fold
needs plain row-major `[E_local, 2I, K]` with the SAME interleaved (g,u) column order.
- Re-prep at **load time** from the pre-shuffle source (lora_layer.py builds quant_info —
  hook there, before the trtllm prepare consumes the plain weights), behind
  `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT`.
- Cost: +9.66 GB/rank (48 layers × 32 experts × 1536×2048×2B) — fine on GB300 288 GB.
- The trtllm copy stays → **decode path untouched** (tuned cubin), zero regression risk.

## 5. Implementation phases (each independently testable)

- **P0 — reference kernel** (skeleton, this commit): naive CUDA fold kernel implementing §2
  exactly (1 thread→(row,h) grid-stride; slow, correctness-only). Unit test vs a pure-torch
  reference on random data + random routing. Pins semantics before any CUTLASS complexity.
- **P1 — CUTLASS grouped GEMM, plain epilogue**: Sm100 CollectiveBuilder (bf16 TN,
  TMA, 2SM MMA), grouped over experts via per-group problem sizes from routing counts
  (host-free: `cta_idx_xy_to_batch_idx` consumed by a scheduler shim or
  `GroupProblemShape` filled by a tiny device kernel from `num_tokens_per_expert`).
  A-operand = **pre-permuted buffer** (keep the existing permute kernel running) to isolate
  GEMM perf; output raw gate_up; keep dev activation. Gate: parity ±10% vs `bmm_Bfloat16`
  57 µs @4096-tok shapes. If this gate fails badly, STOP — re-evaluate (the tuned-cubin risk
  was always the big one).
- **P2 — fold epilogue (EVT)**: custom epilogue visitor: aux load Δ (row idx via
  `permuted_idx_to_expanded_idx`, two half-contiguous loads), 2:1 adjacent-column fold
  (pairs live in one CTA tile: N-tile ≥ 64 even → whole (g,u) pairs co-resident),
  silu·mul, store half-width D [.., I]. Removes the activation kernel + gate_up round-trip.
- **P3 — gather prologue**: A-operand loads via `permuted_idx_to_token_idx` indirection
  (CuTe gather tensor / custom TMA descriptor per row-block, or a fused cp.async gather
  stage), removing the standalone permute (the 180 µs item).
- **P4 — integration**: `Bf16LoraLauncher` branch behind `SGLANG_OPT_BF16_MOE_GEMM1_FOLD`
  (+ DUAL_LAYOUT), prefill-only (≥512 tokens); acc → bench triplet → matrix → upload.

P1..P3 each keep a kill-switch fallback to the previous stage. Perf is judged at P1 and P3
(the two risk points); P2 is mostly correctness work.

## 6. De-risking (decided earlier, restated)

- **Prefill-only dispatch**: decode keeps the tuned trtllm path entirely.
- **Dual-layout weights** funded by bf16's spare HBM (a 1:1 copy only bf16 can afford).
- **opt6 plumbing reuse**: activated-buffer sharing + row-map export already proven.
- Flag family: `SGLANG_OPT_BF16_MOE_GEMM1_FOLD`, `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT`,
  (P3 may add `SGLANG_OPT_BF16_MOE_FOLD_GATHER`); all default-off until each phase's gate
  passes; A/B baselines set `=0` explicitly.

## 7. Validation ladder

1. P0 unit test (in-pod, random data): ref kernel vs torch reference, fp32-accum tolerance.
2. P1/P2/P3 unit: CUTLASS output vs ref kernel on the same inputs.
3. Server acc (teacher-forced prefill logprobs): KL at the vLLM noise floor (~0.004).
4. Bench triplet + single×two matrix per the standing criteria (prefill win must clear ~10%
   in both stream columns; decode must be flat by construction — it never takes this path).
5. Graph-off profile: permute + activation launches → 0 on the prefill path; red-circle.

## 8. Skeleton (this commit)

- `csrc/bf16_moe_gemm1_fold.cu`: `sgl_bf16_fold_probe()` (CUTLASS version/availability
  check — proves the include path), `sgl_bf16_moe_gemm1_fold_ref(...)` (P0 kernel, §2
  semantics), CUTLASS P1 type-plan documented in-file behind a feature macro.
- `jit.py`: + source file, + `flashinfer/data/cutlass/include` include path (CUTLASS 4.x
  with SM100 collective builders — verified present in-pod).
- `core.py`: python wrappers. `dev/test_bf16_fold_ref.py`: P0 unit test driver.
