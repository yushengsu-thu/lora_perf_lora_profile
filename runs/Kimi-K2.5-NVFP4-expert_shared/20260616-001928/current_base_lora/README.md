# current_base_lora — base vs LoRA (Kimi-K2.5-NVFP4, expert_shared r32), bs16

Benchmark + profiling for **Kimi-K2.5-NVFP4, shared-outer (TML) LoRA** — adapter `alpha`
(`lora-diff-Kimi-K2.5`, r=32, all-linear, served with `--experts-shared-outer-loras
--lora-use-virtual-experts`), GB300 **TP8/EP8, 2-node MNNVL**, in=out=2048, sglang **PR #4
`850faa87f`** (branch `qwen3-30b-a3b-2507-bf16`), image `nightly-dev-cu13-20260603` + flashinfer
`0.6.11.post1`.

Two cells:
- **base** = no-LoRA, **STOCK** MoE backend `flashinfer_trtllm`, cuda-graph ON — the real no-LoRA
  production ceiling (cookbook baseline; *not* the experimental LoRA-only backend).
- **lora** = shared-outer r32, `experimental_sgl_trtllm` + virtual-experts, cuda-graph ON
  (two-stream decode) + the required kimi NVFP4 opt envs (`PER_TOKEN_ACTIVATION=1`,
  `GEMM_SWIGLU_FUSION=0`, `KIMI_GATE`/`MOE_ALIGN`/`FUSED_PERMUTE_QUANT`/`FUSED_MOE_ACTIVATION_QUANT_FUSE`).

## Benchmark (`bench/summary.md`; raw jsonl + serverlog in `bench/{no-lora,lora}/`)

| cell | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |
|---|---|---|---|---|---|
| base (no-lora, stock `flashinfer_trtllm`) | 16 | 54,734 | 1,246 | 12.84 | 26.89 |
| lora (shared-outer r32) | 16 | 20,530 | 880 | 18.19 | 38.85 |

**LoRA retained % (cookbook = lora vs STOCK no-LoRA):**  prefill **37.5%** · decode **70.6%** ·
e2e **69.2%** (lora e2e 1.44× slower). The LoRA cost is concentrated in **prefill** (~2.7× slower —
the shared-outer triton-LoRA path); decode retains ~71%. Cross-checked against the server's own
decode median (`serverlog_sanity`: base **+0.0%**, lora **−0.7%** — both OK), so the numbers are
trustworthy, not phantom-bench artifacts.

## Profiling (torch profiler, bs16) — traces in the GitHub Release

Per cell × cuda-graph {ON, OFF}: **ON** = real timing (TP0–TP7, 8 ranks); **OFF** = kernel structure
(TP0). Each pass also has a **GPU-only** witness trace and a **perfetto-compatible** copy.

| pass | trace ranks | GPU-active (busy witness) |
|---|---|---|
| base graph-on | TP0–7 | 78% — borderline (partial host-idle) |
| base graph-off | TP0 | 88% — GPU-bound |
| lora graph-on | TP0–7 | 79% — borderline (partial host-idle) |
| lora graph-off | TP0 | 99% — GPU-bound |

The cuda-graphed decode windows (graph-on) sit at 78–79% GPU-active (some host-bound idle between
launches); eager (graph-off) is more GPU-bound — consistent with the host-bound decode pattern on
these LoRA paths. The per-rank `*.trace.json.gz` open in `ui.perfetto.dev` (use the
`perfetto-compatible-*` copy if the raw trace reports overlapping events).

**Traces — GitHub Release** `Kimi-K2.5-NVFP4-expert_shared-20260616-001928` (40 assets: graph-on /
graph-off / gpu-only / perfetto; asset name = `cell__graph_mode__file`):
- <https://github.com/yushengsu-thu/lora_perf_lora_profile/releases/tag/Kimi-K2.5-NVFP4-expert_shared-20260616-001928>
- ```
  gh release download Kimi-K2.5-NVFP4-expert_shared-20260616-001928 \
    -R yushengsu-thu/lora_perf_lora_profile -D ./traces
  ```

The `<cell>/graph_<mode>/` (and `gpuonly`) directories in this repo are preserved as README pointers
to the release — the trace files themselves live only in the release.

## Notes
- **`--cuda-graph-max-bs 16`** (this run benches/profiles bs16 only): the lora cell reliably
  silent-crashes capturing the large bs=128 cuda graph on this cluster; capping capture at 16 avoids
  it, and the bs16 graph (hence every number here) is captured either way. base/lora bench numbers at
  bs16 are unaffected by the cap.
- Server launches on this env **intermittently silent-crash** at MoE-backend init / cuda-graph
  capture (no traceback; OOM / hardware / install / flashinfer all ruled out — env flakiness; the
  exact config ran clean 3 days earlier). Results were gathered with a clean-and-retry loop that
  banks each cell / profile-pass once it lands.
