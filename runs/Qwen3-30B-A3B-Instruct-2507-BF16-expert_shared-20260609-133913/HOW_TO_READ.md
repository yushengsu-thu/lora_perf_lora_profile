# How to read these GPU traces (where is one decode step?)

![how to read](lora_30b_howto_read.png)

Open any `*/bs*-TP-0.trace.json.gz` in **https://ui.perfetto.dev** (or chrome://tracing). Look at the
**GPU track** (CUDA stream — here `stream 128` is the main stream; the kernels live there).

## One decode step
A decode step = **one forward pass** that emits **1 token per sequence**. Its kernel sequence is:

```
[step head] logits-AllGather → sample(argmax) → embed-lookup → compute-position
   └─ ×48 (one per transformer layer):  RMSNorm → attention (qkv/attn/o) → RMSNorm
                                          → MoE (router → permute → experts GEMM → finalize)
                                          → allreduce (TP)
[step tail] final RMSNorm → lm_head GEMM → logits-AllGather → sample  → (next step)
```

## How to find the step boundary yourself
Count how many times each kernel fires in the profiled window, then:
- **fires N× (N = number of profiled steps)** → it's a **once-per-step** kernel = a **step marker**.
  Here the profiled window = **12 steps**, and `ncclDevKernel_AllGather_RING_LL` (the logits/vocab
  all-gather) + the sampling `reduce_kernel` fire exactly **12×** → the gap between two of them is **one step**.
- **fires N×48** → **once-per-layer-per-step** (here `moe::dev::routing` fires 576× = 12×48) → tells you the model has **48 layers**, and the layer block is what repeats inside a step.

So: **step boundary = the logits AllGather (or the sampling kernel)**; everything between two of them is one step; inside it the layer block repeats 48×.

## The figure
- **Panel A** — the full profiled window. Each **dashed line = one logits-AllGather (once per step)**; the gap between two dashes = **one decode step** (12 steps captured).
- **Panel B** — one step, kernels **packed by GPU-active time** (gaps removed), colored by phase. You can see the **MoE (blue) + attention (green) + norm (orange) + allreduce (red)** pattern repeat ×48, bracketed by the step head (left, red TP comm) and tail (right, lm_head GEMM + AllGather + sample).

## ⚠️ Timing caveat
These traces are from the **profiled** run (torch profiler active), so the **wall-clock per step is inflated** (trace shows ~tens of ms/step; the real served decode is ~3.7 ms/step = 4273 tok/s from the non-profiled bench). **Read the trace for kernel structure & composition, not absolute latency.** Per-kernel GPU *durations* are real; the gaps between kernels are profiler/launch-inflated — which is also why **allreduce looks huge** (spin-wait inflated), and why our analysis excludes it and uses non-allreduce GPU time + the non-profiled bench tok/s.

## For the LoRA cells
Same step structure, but the MoE block is **decomposed** (extra standalone `permute` / `activation` /
`count_and_sort` / `fused_moe` + the `_lora_*` GEMMs) instead of the fused `bmm`+`finalize` you see in
no-lora — that decomposition is the bulk of the LoRA overhead (see `OPTIMIZATION.md`).
