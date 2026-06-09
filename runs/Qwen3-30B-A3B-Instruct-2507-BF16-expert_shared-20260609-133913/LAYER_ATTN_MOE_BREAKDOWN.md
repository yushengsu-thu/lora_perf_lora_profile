# Per-decode-layer breakdown — no-lora vs LoRA(single) vs two-stream (Attn / MoE sublayers)

Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared, GB300, TP4/EP4, bs16, **cuda-graph ON, decode**,
commit `526e0ae22`. One transformer layer inside an NVTX `step[DECODE bs=16]` (48 layers/step), GPU
main-stream (`tid 128`), packed by GPU-active time. Kernels attributed by name (position-assisted);
`shrink A` / `expand B` straddle q/k/v/o (attn) and shared_expert (MoE) and are split by position.

**Per layer: no-lora ~96 µs → LoRA-single ~158 µs (+63 µs, 1.65×).** Base compute (fmha, dense GEMM,
expert GEMM, allreduce) is ~unchanged by LoRA; the delta is the LoRA-added rows.

`two-stream Δ` = GPU-active time removed from the critical (main) stream by two-stream. **It is ~0 for
every group**: two-stream *overlaps* work (concurrent side stream), it does **not** reduce GPU-active
time — on this BF16 path only ~3% of LoRA moves to side streams (11 ms of ~360 ms, whole-trace). Its
real benefit is **wall-clock**: decode +17% (2125→2481 tok/s ≈ −23 µs/layer) via latency hiding.

`url` = (to fill) the optimization commit/PR that addresses this row.

## ① Attention sublayer:  base ~42 µs → LoRA ~55 µs  (+13 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | url |
|---|---|---|---|---|---|---|
| attention fmha | 10.4 | 10.4 | ~0 | 0 | — | |
| dense GEMM nvjet (qkv/o) | ~24 | ~24 | ~0 | 0 | — | |
| allreduce (after attn) | ~8 | ~8 | ~0 | 0 | — | |
| LoRA qkv_b expand | 0 | 4.2 | +4.2 | ~0 | cuBLAS expand | |
| LoRA shrink A (q/k/v/o part) | 0 | ~4.2 | +4.2 | ~0 | fuse q/k/v shrink into one GEMM + cuBLAS | |
| LoRA expand B (o part) | 0 | ~5.0 | +5.0 | ~0 | cuBLAS; fuse fp32→bf16 cast into expand | |

## ② MoE sublayer:  base ~50 µs → LoRA ~96 µs  (+46 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | url | MoE-decomp extra — components (µs/layer) |
|---|---|---|---|---|---|---|---|
| MoE core (bmm expert GEMM / router / finalize) | 36.3 | 34.2 | ~0 | 0 | — | | |
| gate GEMM (nvjet) + allreduce (after MoE) | ~14 | ~14 | ~0 | 0 | — | | |
| routing (`routingCustom`) — **not LoRA-added** | 5.1 | 4.4 | −0.6 | 0 | — | | |
| **MoE-decomp extra** | 0 | **26.0** | **+26.0** | ~0 | in-MoE LoRA fold (FP8 has it, BF16 doesn't) + fuse routing/align/topk/elem | | • **align/sort/scatter +10.2** (`moe_align_block_size_small_batch` 6.7 + `moe_lora_merged::fused_align_scatter` 3.5, latter LoRA-specific)<br>• **fused_moe +7.2** (expert GEMM, replaces `bmm`)<br>• **elem / copy / cast +3.9** (upcast / copy)<br>• **activation +3.2** (`moe::dev::activation`)<br>• **topk / pack +3.0** (`_fused_virtual_topk_ids`)<br>• **permute +2.4** (`moe::dev::permute`) |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | fold into expert GEMM | | |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | cuBLAS / fuse | | |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | fold into expert GEMM | | |

(~+6 µs/layer of elementwise/reshape from the decomposed path is split across both sublayers. The
"MoE-decomp extra" components sum to ~+29 µs incl. routing≈0; the headline +26 µs excludes the LoRA
shrink/expand GEMM rows.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — MoE-decomp extra (~+26 µs, biggest item **align/sort +10.2 µs**,
  larger than any single LoRA GEMM) + MoE LoRA shrink/expand (+13 µs) + shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs. The decomp cluster is ~46% of the per-layer LoRA overhead.

## in-MoE fold vs (a) fusion — no double-counting
| absorbed by **in-MoE LoRA fold** | addressed only by **(a) fusion/removal** |
|---|---|
| `fused_moe` (7.2) + `activation` (3.2) + `permute` (2.4) + routed-expert LoRA shrink/expand (~13) ≈ **~25 µs** | `align/sort/scatter` (10.2) + `topk/pack` (3.0) + `elem/cast` (3.9) ≈ **~17 µs** |

→ **Stackable**: in-MoE fold (~−25 µs) + (a) align/topk/elem fusion (~−17 µs) ≈ **−42 µs of the +63 µs
overhead (~67%)**.

## Recommended order (decode, bs16)
1. **in-MoE LoRA fold** — biggest structural win (~25 µs); proven on the FP8 path (port `sgl_fp8_moe.py`
   / dev-kernel to BF16).
2. **align/sort/scatter fusion** (~10 µs) — largest (a) item; fixed-cost at small batch → decode benefits
   disproportionately.
3. **topk+pack single launch** (~3 µs) + **drop elem/upcast / `_get_lora_info`** (~4 µs) — PR #27329 /
   team action items; cheap, additive.

bs16 is latency / fixed-cost bound — which is why these small routing/align/elem kernels matter at
decode. allreduce excluded from GPU-active analysis (spin-wait inflated). Numbers are one steady
decode layer.
