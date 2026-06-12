# current_base_lora — base vs LoRA (current optimized), bs16/32/64

Current benchmark + profiling for **Qwen3-30B-A3B-Instruct-2507-BF16, expert_shared**
(adapter `alpha`, r32), GB300 TP4/EP4, in=out=2048.

**Code: sglang PR #4 commit [`850faa87f`](https://github.com/yushengsu-thu/sglang/pull/4/commits/850faa87fbcc7d54210bc86866d2f9b3ecf4abce)**
(opt5 — prefill routing reuse, the e2e-proven set: opt1 fused-align + opt2 topk-pack + opt3
lean-info + opt5; no opt6/7/8). Re-measured on a dedicated isolated pod 2026-06-12.

Two configs:
- **base** = no-LoRA, **cuda-graph ON** (same MoE backend `experimental_sgl_trtllm`, no adapter).
- **lora** = LoRA, **cuda-graph ON + two-stream**, with the current optimized defaults
  (opt1 fused-align + opt2 topk-pack + opt3 lean-info + common opt-in flags; two-stream is
  default-on for decode at `SGLANG_TWO_STREAM_MAX_TOKENS=256`).

## Benchmark (`benchmark_summary.md`, raw jsonl in `bench/{base,lora}/`)
| cell | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|
| base | 16 | 188,952 | 3,926 | 8.52 |
| base | 32 | 190,476 | 6,902 | 9.84 |
| base | 64 | 192,230 | 11,708 | 11.88 |
| lora | 16 | 33,240 | 2,537 | 13.90 |
| lora | 32 | 38,680 | 4,664 | 15.75 |
| lora | 64 | 39,090 | 8,044 | 19.65 |

**lora / base:** decode 64.6 / 67.6 / 68.7% · prefill ~20% (bs16 17.6% — see note) · e2e 1.60–1.65×.
(decode is two-stream; prefill is always serial — in=2048 > the 256-token two-stream gate.)

> Decode tok/s is the headline (run-to-run variance ~0.06%; all cells sanity-checked vs
> server-log ≤1.5%). Prefill tok/s is noisy (~6–14%); the **lora bs16 prefill (33,240) is the
> first measured batch on a fresh server and runs mildly cold** (~14% below the bs32/64
> steady-state ~38.7k) — treat it as a lower bound, not a regression.

## Profiling (`profile/{base,lora}/bs16-TP-{0,1,2,3}.trace.json.gz`)
Torch profiler (CPU+GPU), **cuda-graph ON** decode window (start-step 8, 16 steps), **bs16**, with
**all 4 TP ranks (TP0–TP3) as separate files**.
- `profile/base/` — base (no-LoRA), cuda-graph.
- `profile/lora/` — lora, cuda-graph + two-stream.
Open each per-rank `.gz` in perfetto / `ui.perfetto.dev`.
