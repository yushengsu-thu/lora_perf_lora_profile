# current_base_lora — base vs LoRA (current optimized), bs16/32/64

Current benchmark + profiling for **Qwen3-30B-A3B-Instruct-2507-BF16, expert_shared**
(adapter `alpha`, r32), GB300 TP4/EP4, in=out=2048.

Two configs:
- **base** = no-LoRA, **cuda-graph ON** (same MoE backend `experimental_sgl_trtllm`, no adapter).
- **lora** = LoRA, **cuda-graph ON + two-stream**, with the current optimized defaults
  (opt1 fused-align + opt2 topk-pack + opt3 lean-info + common opt-in flags; two-stream is
  default-on for decode at `SGLANG_TWO_STREAM_MAX_TOKENS=256`).

## Benchmark (`benchmark_summary.md`, raw jsonl in `bench/{base,lora}/`)
| cell | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|
| base | 16 | 188,142 | 3,933 | 8.51 |
| base | 32 | 192,148 | 6,920 | 9.81 |
| base | 64 | 193,852 | 11,714 | 11.87 |
| lora | 16 | 35,751 | 2,574 | 13.65 |
| lora | 32 | 36,331 | 4,686 | 15.79 |
| lora | 64 | 36,647 | 8,088 | 19.78 |

**lora / base:** decode 65.5 / 67.7 / 69.0% · prefill ~19% · e2e 1.60–1.67×.
(decode is two-stream; prefill is always serial — in=2048 > the 256-token two-stream gate.)

## Profiling (`profile/{base,lora}/bs{16,32,64}-TP-0.trace.json.gz`)
Torch profiler (CPU+GPU), **cuda-graph ON** decode window (start-step 8, 16 steps), TP0.
- `profile/base/` — base (no-LoRA), cuda-graph.
- `profile/lora/` — lora, cuda-graph + two-stream.
Open in perfetto / `ui.perfetto.dev`.
