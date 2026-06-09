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
| **MoE-decomp extra** (permute / activation / moe_align / fused_moe / topk / elem) | 0 | **26.0** | **+26.0** | ~0 | **fold LoRA delta into the fused MoE** (FP8 path has this; BF16 does not) + fuse routing/align/topk/elem — see expansion below |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | fold into expert GEMM |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | cuBLAS / fuse |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | fold into expert GEMM |

(~+6 µs/layer of elementwise/reshape from the decomposed path is split across both sublayers.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — dominated by **MoE-decomp extra (+26 µs)** (MoE leaving the fused
  path) plus the MoE LoRA shrink/expand (+13 µs) and shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs.

---

## Expansion of "MoE-decomp extra" — routing / topk / align / elem (the (a) cluster)

Expanding the +26 µs "MoE-decomp extra" row into its individual kernels (one decode layer, main stream):

| sub-group | no-lora | LoRA | Δ (LoRA adds) | kernels |
|---|---|---|---|---|
| **align/sort/scatter** | 0 | **10.2** | **+10.2** | `moe_align_block_size_small_batch` 6.7 + `moe_lora_merged::fused_align_scatter` 3.5 (latter LoRA-specific) |
| **fused_moe (expert GEMM)** | 0 | 7.2 | +7.2 | `fused_moe_kernel` (LoRA-path expert GEMM; replaces no-lora's `bmm`) |
| **elem / copy / cast** | 0 | 3.9 | +3.9 | vectorized elementwise / copy / upcast |
| **activation** (standalone) | 0 | 3.2 | +3.2 | `moe::dev::activation` |
| **topk / pack** | 0 | 3.0 | +3.0 | `_fused_virtual_topk_ids` |
| **permute** | 0 | 2.4 | +2.4 | `moe::dev::permute` |
| routing | 5.1 | 4.4 | **−0.6** | `routingCustom` — present in no-lora, **NOT added by LoRA** |
| **cluster total Δ** | | | **+29.3 µs/layer** | ≈ **46% of the +63 µs/layer LoRA overhead** |

Findings:
- This cluster is **~46% of the per-layer LoRA overhead**. The biggest single item is
  **`align/sort/scatter` +10.2 µs** — larger than any single LoRA GEMM. `moe_align_block_size_small_batch`
  is **fixed-cost at bs16** (doesn't scale down with the tiny batch); `fused_align_scatter` is LoRA-specific.
- **`routing` itself is NOT LoRA-added** (−0.6) — lumping it into "decomp" earlier was misleading.

### in-MoE fold vs (a) fusion — no double-counting
| absorbed by **in-MoE LoRA fold** | addressed only by **(a) fusion/removal** |
|---|---|
| `fused_moe` (7.2) + `activation` (3.2) + `permute` (2.4) + routed-expert LoRA shrink/expand (~13) ≈ **~25 µs** | `align/sort/scatter` (10.2) + `topk/pack` (3.0) + `elem/cast` (3.9) ≈ **~17 µs** |

→ **Stackable**: in-MoE fold (~−25 µs) + (a) align/topk/elem fusion (~−17 µs) ≈ **−42 µs of the +63 µs
overhead (~67%)**.

### Recommended order (decode, bs16)
1. **in-MoE LoRA fold** — biggest structural win (~25 µs); already proven on the FP8 path (port
   `sgl_fp8_moe.py` / dev-kernel to BF16).
2. **align/sort/scatter fusion** (~10 µs) — largest (a) item; fixed-cost at small batch → decode
   benefits disproportionately.
3. **topk+pack single launch** (~3 µs) + **drop elem/upcast / `_get_lora_info`** (~4 µs) — PR #27329 /
   team action items; cheap, additive.

bs16 is latency / fixed-cost bound — which is exactly why these small routing/align/elem kernels matter
at decode. allreduce excluded from GPU-active analysis (spin-wait inflated). Numbers are one steady
decode layer.
