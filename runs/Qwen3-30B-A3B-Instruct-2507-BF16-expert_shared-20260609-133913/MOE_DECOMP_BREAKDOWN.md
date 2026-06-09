# MoE routing/topk/align/elem breakdown at decode — and how it splits vs the in-MoE-fold lever

Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared, GB300, TP4/EP4, bs16, cuda-graph ON, **decode**,
commit `526e0ae22`. One transformer layer inside an NVTX `step[DECODE bs=16]`, main stream (`tid 128`),
GPU-active µs. Compares `no-lora-graphon` vs `lora-single-graphon`. This expands the "MoE-decomp extra"
row of `LAYER_ATTN_MOE_BREAKDOWN.md` into its individual kernels (the routing/topk/align/elem cluster).

## Per-layer breakdown of the routing/topk/align/elem/permute/activation/fused_moe cluster

| sub-group | no-lora | LoRA | Δ (LoRA adds) | kernels |
|---|---|---|---|---|
| **align/sort/scatter** | 0 | **10.2** | **+10.2** | `moe_align_block_size_small_batch` 6.7 + `moe_lora_merged::fused_align_scatter` 3.5 (latter is LoRA-specific) |
| **fused_moe (expert GEMM)** | 0 | 7.2 | +7.2 | `fused_moe_kernel` (the LoRA-path expert GEMM; replaces no-lora's `bmm`) |
| **elem / copy / cast** | 0 | 3.9 | +3.9 | vectorized elementwise / copy / upcast |
| **activation** (standalone) | 0 | 3.2 | +3.2 | `moe::dev::activation` |
| **topk / pack** | 0 | 3.0 | +3.0 | `_fused_virtual_topk_ids` |
| **permute** | 0 | 2.4 | +2.4 | `moe::dev::permute` |
| routing | 5.1 | 4.4 | **−0.6** | `moe::dev::routing::routingCustom` — **present in no-lora; NOT added by LoRA** |
| **cluster total Δ** | | | **+29.3 µs/layer** | ≈ **46% of the total +63 µs/layer LoRA overhead** |

## Key findings
1. This cluster is **~46% of the per-layer LoRA overhead** (+29 of +63 µs) — large, worth attacking.
2. The biggest single item is **`align/sort/scatter` +10.2 µs** — larger than any single LoRA GEMM.
   `moe_align_block_size_small_batch` is **fixed-cost at bs16** (doesn't scale down with the tiny
   batch), and `fused_align_scatter` is LoRA-specific. Prime target for decode.
3. **`routing` itself is NOT LoRA-added** (−0.6; it already exists in no-lora). Earlier lumping it into
   "MoE-decomp" was misleading — the LoRA-added MoE-base cost is align/topk/permute/activation/fused_moe.

## How this splits vs the "in-MoE LoRA fold" lever (no double-counting)
The in-MoE fold (port FP8's `sgl_fp8_moe.py` / dev-kernel to BF16) and the (a) fusion/removal levers
attack **different** kernels:

| absorbed by **in-MoE fold** | addressed only by **(a) fusion/removal** |
|---|---|
| `fused_moe` (7.2) + `activation` (3.2) + `permute` (2.4) ≈ **13 µs** | `align/sort/scatter` (10.2) + `topk/pack` (3.0) + `elem/cast` (3.9) ≈ **17 µs** |
| (+ the routed-expert LoRA shrink/expand ~13 µs from the other table) | (PR #27329 / team actions: fuse topk+pack single launch; drop `_get_lora_info` / elem upcast; merge gate_up align) |

So the two are **stackable**:
- **in-MoE fold** ≈ −25 µs/layer (fused_moe + activation + permute + MoE LoRA shrink/expand)
- **(a) align/topk/elem fusion** ≈ −17 µs/layer (NOT covered by the fold)
- together ≈ **−42 µs of the +63 µs LoRA overhead (~67%)**.

## Recommended order (decode, bs16)
1. **in-MoE LoRA fold** — biggest single structural win (~25 µs), already proven on the FP8 path.
2. **align/sort/scatter fusion** (~10 µs) — the largest (a) item; fixed-cost at small batch, so decode
   benefits disproportionately. Streamline `moe_align_block_size_small_batch` + `fused_align_scatter`.
3. **topk+pack single launch** (~3 µs) and **drop elem/upcast / `_get_lora_info`** (~4 µs) — PR/team
   action items; cheap, additive.

Notes: bs16 is latency / fixed-cost bound, which is exactly why these small routing/align/elem kernels
matter at decode. allreduce excluded (spin-wait inflated). Numbers are one steady decode layer.
