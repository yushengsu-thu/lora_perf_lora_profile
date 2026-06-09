# One decode layer — no-lora vs lora (where LoRA inserts), cuda-graph ON, bs16

![one layer](lora_30b_one_layer.png)

Both rows = **one transformer layer**, kernels packed by GPU-active time (input-`RMSNorm` →
attention → `RMSNorm` → MoE → allreduce). Same x-scale.

- **no-lora (top): 0.35 ms/layer, 20 kernels.** The MoE is one **compact fused block** — the blue
  `bmm` expert GEMM + `finalize`. Layout: RMSNorm(yellow) → qkv GEMM(purple) → attention(green) →
  o GEMM(purple) → allreduce(gray) → RMSNorm → gate GEMM → **MoE core (blue)** → allreduce.
- **lora-single (bottom): 1.32 ms/layer, 40 kernels (+0.98 ms).** Enabling LoRA replaces the compact
  blue MoE with a sprawling **ORANGE = decomposed-MoE extra (permute / activation / count_and_sort /
  fused_moe, 0.78 ms)** + **RED = LoRA GEMMs (shrink/expand/sgemm, 0.20 ms)** region.

**Takeaway:** per layer, LoRA's own matmuls (red, 0.20 ms) are small; the bulk of the +0.98 ms is the
**ORANGE decomposed-MoE overhead** — the MoE leaving the fused path. That's the optimization target
(re-fuse activation/permute into the GEMM; see `OPTIMIZATION.md`). The attention half of the layer
(green + purple) is unchanged by LoRA.
