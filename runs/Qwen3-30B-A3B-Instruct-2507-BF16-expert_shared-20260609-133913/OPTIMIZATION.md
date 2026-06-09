# Optimization analysis — where the LoRA decode overhead goes (bs16, graph-on)

From this run's traces (`no-lora-graphon` vs `lora-single-graphon`, rank0, non-allreduce GPU time).
Enabling LoRA drops bs16 decode from **4273 → 2125 tok/s** (graph-on). That +351 ms of non-allreduce
GPU time breaks down as:

![breakdown](lora_30b_optimization.png)

| source | Δ ms | nature |
|---|---|---|
| `moe::dev::activation` | **111** | standalone activation (fused path has none) |
| `moe::dev::permute` | **74** | token permute (expert grouping) |
| `fused_moe_kernel` | 41 | decomposed expert GEMM |
| `count_and_sort_expert_tokens` + `moe_align_block_size` | ~47 | routing / sort |
| **decomposed-MoE subtotal** | **~270** | **NOT the LoRA math — the cost of MoE running decomposed** |
| `_moe_lora_shrink_splitk`+`_sgemm_lora_a/b`+`_qkv_lora_b`+`_moe_lora_expand_add` | ~83 | the actual LoRA matmuls |

## Root cause
The no-lora path runs the **fused trtllm BF16 MoE** — its whole MoE is `bmm_Bfloat16` (42 ms) +
`finalize` (6 ms) + routing (3 ms), with permute/activation done *inside*. Enabling LoRA
(`--lora-use-virtual-experts`) kicks the MoE onto a **decomposed** path: separate `permute` →
`fused_moe`/GEMM → `activation` → routing/sort kernels. **That decomposition (~270 ms) dwarfs the
LoRA matmuls (~83 ms).** Same root cause seen on FP8 35B (the "elementwise/reshape + moe-base"
growth there).

## Optimization levers (ranked)
1. **Re-fuse the LoRA-MoE path toward the no-lora fused path (biggest, ~150 ms+).** Fold
   `activation` (111 ms) into the down-GEMM epilogue (SwiGLU-in-epilogue) and `permute` (74 ms) into
   the gather/shrink, injecting the LoRA delta — so the LoRA MoE looks like the fused path's
   `bmm`+`finalize` (~48 ms) instead of 6 separate kernels.
2. **`activation` 111 ms is pathologically fat at bs16** (bigger than the entire no-lora MoE bmm,
   42 ms). 16×topk token-expert pairs should not cost 111 ms → poor occupancy / per-expert loop.
   A small-batch / better-tiled activation kernel is a standalone win.
3. **`permute` 74 ms** — fuse into shrink/gather, or keep the fused path's token layout so no
   re-permute is needed.
4. **`count_and_sort` (32) + `moe_align` (15)** — the trace has a `moe_align_block_size_small_batch_expert`
   variant (1.3 ms); at bs16 the routing should take the small-batch path, not the full kernels.
5. **Extend two-stream to the big kernels.** Today two-stream overlaps only the ~9 ms of small LoRA
   GEMMs (→ +17% decode); `permute` (74) and `activation` (111) still run serially on the main
   stream. Overlapping those with base attention / the next layer is a much larger lever than +17%.
   (Note: two-stream **engages on BF16** — LoRA spread onto ~90 side streams — but **not on FP8 35B**,
   where LoRA stayed inline on the main stream; that's a separate FP8 gap worth fixing.)
6. **Batch size.** bs16 is fixed-cost / launch-bound (permute/sort/align scale poorly small). At
   bs64 the LoRA fraction is much smaller (FP8 bs64 sat at 73–80% of ceiling vs 50–58% here at bs16).

## Method note
Wall-clock decode tok/s is the metric (graph-off GPU-time sums are launch/sync-inflated). cuda-graph
itself is ~13–17× at bs16 (launch-bound). Two-stream's GPU-time sum is ~equal single-vs-two (it
**overlaps**, not removes work); the gain is wall-clock only (+17%).
