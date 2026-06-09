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

"MoE-decomp extra" (+26 µs) is expanded inline into its individual kernels.

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization |
|---|---|---|---|---|---|
| MoE core (bmm expert GEMM / router / finalize) | 36.3 | 34.2 | ~0 | 0 | — |
| gate GEMM (nvjet) + allreduce (after MoE) | ~14 | ~14 | ~0 | 0 | — |
| routing (`routingCustom`) — **not LoRA-added** | 5.1 | 4.4 | −0.6 | 0 | — |
| **decomp: align/sort/scatter** (`moe_align_block_size_small_batch` 6.7 + `fused_align_scatter` 3.5) | 0 | **10.2** | **+10.2** | ~0 | fuse align; drop `fused_align_scatter` (LoRA-specific); fixed-cost at bs16 |
| **decomp: fused_moe** (expert GEMM, replaces `bmm`) | 0 | 7.2 | +7.2 | ~0 | in-MoE LoRA fold (absorb into fused expert GEMM) |
| **decomp: elem / copy / cast** | 0 | 3.9 | +3.9 | ~0 | drop upcast / `_get_lora_info` |
| **decomp: activation** (`moe::dev::activation`) | 0 | 3.2 | +3.2 | ~0 | fold into down-GEMM epilogue (in-MoE fold) |
| **decomp: topk/pack** (`_fused_virtual_topk_ids`) | 0 | 3.0 | +3.0 | ~0 | fuse topk + routed-pack single launch |
| **decomp: permute** (`moe::dev::permute`) | 0 | 2.4 | +2.4 | ~0 | fold into gather (in-MoE fold) |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | fold into expert GEMM |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | cuBLAS / fuse |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | fold into expert GEMM |

(MoE-decomp sub-rows sum to ~+29 µs incl. routing≈0; the +26 µs in the headline excludes the
LoRA shrink/expand GEMM rows below it. ~+6 µs/layer of elementwise is split across both sublayers.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — decomp cluster (~+26 µs, biggest item **align/sort +10.2 µs**) +
  MoE LoRA shrink/expand (+13 µs) + shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs.
- Note: the decomp cluster is ~46% of the per-layer LoRA overhead; `align/sort/scatter` (+10.2 µs) is
  larger than any single LoRA GEMM (`moe_align_block_size_small_batch` is fixed-cost at bs16).

## in-MoE fold vs (a) fusion — no double-counting
| absorbed by **in-MoE LoRA fold** | addressed only by **(a) fusion/removal** |
|---|---|
| `fused_moe` (7.2) + `activation` (3.2) + `permute` (2.4) + routed-expert LoRA shrink/expand (~13) ≈ **~25 µs** | `align/sort/scatter` (10.2) + `topk/pack` (3.0) + `elem/cast` (3.9) ≈ **~17 µs** |

→ **Stackable**: in-MoE fold (~−25 µs) + (a) align/topk/elem fusion (~−17 µs) ≈ **−42 µs of the +63 µs
overhead (~67%)**.

## Recommended order (decode, bs16)
1. **in-MoE LoRA fold** — biggest structural win (~25 µs); proven on the FP8 path (port `sgl_fp8_moe.py`
   / dev-kernel to BF16).
2. **align/sort/scatter fusion** (~10 µs) — largest (a) item; fixed-cost at small batch → decode
   benefits disproportionately.
3. **topk+pack single launch** (~3 µs) + **drop elem/upcast / `_get_lora_info`** (~4 µs) — PR #27329 /
   team action items; cheap, additive.

bs16 is latency / fixed-cost bound — which is why these small routing/align/elem kernels matter at
decode. allreduce excluded from GPU-active analysis (spin-wait inflated). Numbers are one steady
decode layer.
