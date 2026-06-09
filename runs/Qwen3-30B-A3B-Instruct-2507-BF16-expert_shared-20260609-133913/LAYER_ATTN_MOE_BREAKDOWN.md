# Per-decode-layer breakdown — no-lora vs LoRA(single) vs two-stream (Attn / MoE sublayers)

Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared, GB300, TP4/EP4, bs16, **cuda-graph ON, decode**,
commit `526e0ae22`. One transformer layer inside an NVTX `step[DECODE bs=16]` (48 layers/step), GPU
main-stream (`tid 128`), packed by GPU-active time. Kernels attributed by name (position-assisted);
`shrink A` / `expand B` straddle q/k/v/o (attn) and shared_expert (MoE) and are split by position.

**Per layer: no-lora ~96 µs → LoRA-single ~158 µs (+63 µs, 1.65×).** Base compute (fmha, dense GEMM,
expert GEMM, allreduce) is ~unchanged by LoRA; the delta is the LoRA-added rows.

`two-stream Δ` = GPU-active time removed from the critical (main) stream by two-stream. **It is ~0 for
every group**: two-stream *overlaps* work (concurrent side stream), it does **not** reduce GPU-active
time — and on this BF16 path only ~3% of LoRA actually moves to side streams (11 ms of ~360 ms,
whole-trace). Two-stream's real benefit is **wall-clock**: decode +17% (2125→2481 tok/s ≈ −23 µs/layer)
via latency hiding / bubble fill, not by shrinking any group below.

## ① Attention sublayer:  base ~42 µs → LoRA ~55 µs  (+13 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization |
|---|---|---|---|---|---|
| attention fmha | 10.4 | 10.4 | ~0 | 0 | — |
| dense GEMM nvjet (qkv/o) | ~24 | ~24 | ~0 | 0 | — |
| allreduce (after attn) | ~8 | ~8 | ~0 | 0 | — |
| LoRA qkv_b expand | 0 | 4.2 | +4.2 | ~0 | cuBLAS expand |
| LoRA shrink A (q/k/v/o part) | 0 | ~4.2 | +4.2 | ~0 | fuse q/k/v shrink into one GEMM + cuBLAS |
| LoRA expand B (o part) | 0 | ~5.0 | +5.0 | ~0 | cuBLAS; fuse fp32→bf16 cast into expand |

## ② MoE sublayer:  base ~50 µs → LoRA ~96 µs  (+46 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization |
|---|---|---|---|---|---|
| MoE core (bmm expert GEMM / router / finalize) | 36.3 | 34.2 | ~0 | 0 | — |
| gate GEMM (nvjet) + allreduce (after MoE) | ~14 | ~14 | ~0 | 0 | — |
| **MoE-decomp extra** (permute / activation / moe_align / fused_moe) | 0 | **26.0** | **+26.0** | ~0 | **fold LoRA delta into the fused MoE** (FP8 path has this; BF16 does not) + JIT MoE-align |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | fold into expert GEMM |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | cuBLAS / fuse |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | fold into expert GEMM |

(~+6 µs/layer of elementwise/reshape from the decomposed path is split across both sublayers.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — dominated by **MoE-decomp extra (+26 µs)** (MoE leaving the fused
  path) plus the MoE LoRA shrink/expand (+13 µs) and shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs.

## Highest-leverage optimizations
1. **in-MoE LoRA fold** (port FP8's `sgl_fp8_moe.py` / dev-kernel path to BF16) — removes the +26 µs
   decomp-extra and folds the +13 µs MoE LoRA → ~+39 µs of the +63 (≈62%).
2. **cuBLAS + fused shrink** for the attention/dense LoRA (~+13 µs); merge q/k/v shrink into one GEMM.
   NB: flipping `SGLANG_OPT_LORA_CUBLAS_*` alone did **nothing** on BF16 (measured) — those flags are
   wired only for the FP8/NVFP4 path; the BF16 dispatch needs the branch added.
3. **two-stream** can only *hide* (wall-clock), not remove; today it overlaps only ~3% of LoRA on this
   BF16 path — raising the side-stream overlap is a separate lever, but it won't reduce the GPU-active
   groups above.

bs16 is small / latency-bound; allreduce excluded from GPU-active analysis (spin-wait inflated).
