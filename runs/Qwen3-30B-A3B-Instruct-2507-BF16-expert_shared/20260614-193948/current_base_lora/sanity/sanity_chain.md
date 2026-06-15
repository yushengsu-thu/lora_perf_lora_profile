# sanity chain — Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared  run 20260614-193948
branch qwen3-30b-a3b-2507-bf16 @ 850faa87f  | TP4/EP4 | 2026-06-14 20:23 PDT

## 1. serverlog_sanity (bench, step 3) — bench decode vs server-log decode
all 6 cells OK (<=1.1% diff): no-lora bs16/32/64 -0.7/-0.8/-1.1%; lora bs16/32/64 +0.4/-0.8/-0.4%

## 2. gpu_busy_witness (gpu-only profile, step 5)
- [GPU-BUSY] GPU-active 439.2 ms / span 516.7 ms = 85%  (26121 GPU ops)  -> borderline — partial GPU idle
- [GPU-BUSY] GPU-active 410.9 ms / span 451.4 ms = 91%  (11351 GPU ops)  -> GPU-BOUND (busy) — kernel/compute opts can pay off
- [GPU-BUSY] GPU-active 2677.0 ms / span 3149.1 ms = 85%  (44865 GPU ops)  -> GPU-BOUND (busy) — kernel/compute opts can pay off
- [GPU-BUSY] GPU-active 1021.2 ms / span 1564.3 ms = 65%  (21015 GPU ops)  -> borderline — partial GPU idle

## 3. profile_metrics (graph-on decode, independent witness)
### no-lora graph-on TP0 (steps=24)
{
  "trace": "bs64-TP-0.trace.json.gz",
  "n_kernels": 24421,
  "layers": 0,
  "steps_arg": 24,
  "method": "top-(steps-1)-gap segmentation (no ProfilerStep markers)",
  "n_decode_steps": 23,
  "n_prefill_dropped": 1,
  "forward_pass_us": 30453.4,
  "forward_pass_ms": 30.453,
  "per_layer_us": null
}
### lora graph-on TP0 (steps=24)
{
  "trace": "bs64-TP-0.trace.json.gz",
  "n_kernels": 39517,
  "layers": 0,
  "steps_arg": 24,
  "method": "top-(steps-1)-gap segmentation (no ProfilerStep markers)",
  "n_decode_steps": 22,
  "n_prefill_dropped": 2,
  "forward_pass_us": 156282.0,
  "forward_pass_ms": 156.282,
  "per_layer_us": null
}

## 4. sanity_check_opt (graph-off prefill host-bound verdict)
### no-lora graph-off TP0
== 1. host-bound check (profiler-inflated wall; ratios are what matter) ==
  PREFILL: wall 474 ms, GPU-busy 224 ms, idle 53%, 8139 launches -> HOST-BOUND — GPU-side opts will NOT move e2e
  DECODE: wall 888 ms, GPU-busy 812 ms, idle 9%, 2634 launches -> GPU-bound — kernel opts pay off
### lora graph-off TP0
== 1. host-bound check (profiler-inflated wall; ratios are what matter) ==
  PREFILL: wall 3552 ms, GPU-busy 1006 ms, idle 72%, 13180 launches -> HOST-BOUND — GPU-side opts will NOT move e2e
  DECODE: wall 2810 ms, GPU-busy 1417 ms, idle 50%, 5992 launches -> HOST-BOUND — GPU-side opts will NOT move e2e
