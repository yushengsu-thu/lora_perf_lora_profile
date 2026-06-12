# Per-layer breakdown — no-lora vs LoRA (Attn / MoE sublayers) — REBUILT 2026-06-12

Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared, GB300, TP4/EP4, branch = **opt1+2+5**
(`850faa87f`), run [`20260612-051818`]: graph-on **decode bs64** (by-stage DECODE trace, TP0,
two-stream default-on) + graph-on **prefill** (bs64 EXTEND chunks, TP0). All numbers below are
from THIS round's traces — no carried-over figures. The pre-restructure (bs16, `526e0ae22`)
table is preserved in git history.

**Decode bs64, per layer (CPU step-wall): base 30.8 µs → LoRA-single 42.1 → LoRA-two 42.6 µs;
GPU-active (sum over streams): 130.9 → 180.5 → 195.0 µs.** Production wall = bench ITL:
base 5.46 ms/step, lora-two 7.90 ms (ratio 69.1%); single ≈ 9.1 ms (historical matrices —
single→two decode +15~21% wall).
**Reading the single vs two columns**: two-stream costs **+14.5 µs/layer MORE GPU-active
(+8%)** (side-stream bookkeeping: align/sort +3.4, elem/copy +4.9) yet wins wall via overlap
— i.e. the overlap is already in the production baseline and removes no per-row GPU work, so
no row below can be "hidden by two-stream instead of optimized". The big single→two per-row
moves (shrink A 12.1→5.1) are work *relocation* to the side stream, not removal.
**Prefill (4096-tok chunk), per layer-forward: base wall 525 µs → LoRA 4558 µs (8.7×; tok/s ratio 20.9%).**

## ① Attention sublayer (decode bs64, µs/layer GPU-active; base / LoRA-single / LoRA-two)

| group | base | LoRA single | LoRA two | optimization | url |
|---|---|---|---|---|---|
| attention fmha | 16.5 | 16.7 | 16.8 | — | |
| dense GEMM nvjet (qkv/o + MoE gate) | 20.5 | 19.5 | 23.2 | — | |
| qknorm/rope | 4.9 | 4.4 | 4.3 | — | |
| LoRA qkv_b expand | 0 | 4.4 | 4.8 | cuBLAS expand | ✅ enabled (PR#4 baseline: `SGLANG_OPT_LORA_CUBLAS`, `SGLANG_OPT_LORA_QKV_B_STORE`) |
| LoRA shrink A (attn+shared, both sublayers) | 0 | 12.1 | 5.1 | fused qkv shrink + cuBLAS; single→two = side-stream **relocation**, not removal | ✅ enabled (PR#4 baseline) |
| LoRA expand B (o+shared, both sublayers) | 0 | 5.5 | 5.5 | cuBLAS + fused cast | ✅ enabled (PR#4 baseline) |

## ② MoE sublayer (decode bs64, µs/layer GPU-active; base / LoRA-single / LoRA-two)

| group | base | LoRA single | LoRA two | optimization | url |
|---|---|---|---|---|---|
| expert GEMM (bmm cubin; base = fused MoE runner) | 39.2 | 37.8 | 39.6 | — | |
| routing | 6.4 | 4.8 | 4.5 | — | |
| finalize | 4.9 | 2.9 | 2.9 | — | |
| align/sort (Triton) | 0 | 10.9 | 14.2 | align/sort fusion + routing reuse; two-stream pays +3.4 side-stream bookkeeping | ✅ [opt1](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea) · ✅ [opt5](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) |
| topk/pack | 0 | 4.3 | 4.1 | fused topk+pack | ✅ opt2 (flag-only) |
| LoRA MoE shrink (routed) | 0 | 10.2 | 11.1 | | |
| fused_moe (LoRA-Δ B-expand) | 0 | 7.8 | 7.9 | | |
| LoRA MoE expand (routed) | 0 | 3.4 | 3.6 | | |
| permute (standalone) | 0 | 4.5 | 5.0 | | |
| activation (standalone) | 0 | 2.7 | 2.5 | | |
| allreduce/comm | 27.1 | 20.4 | 26.9 | — (sync-point cost; see host view) | |
| elem/copy/norm (merged — naming differs across captures) | 11.5 | 8.3 | 13.2 | — | |
| **TOTAL GPU-active** | **130.9** | **180.5** | **195.0** | | |

## Prefill view (µs/layer-forward, 4096-tok chunk, graph-on bs64 TP0 — production path)

`two-stream?` = can this row be hidden by two-stream overlap instead of being optimized?
**✗ for EVERY prefill row — measured, not assumed**: (a) prefill is serial by design
(`SGLANG_TWO_STREAM_MAX_TOKENS=256` < 4096-tok chunks; this round's prefill trace: **100% of
kernel time on the single main stream, side-stream = 0**); (b) raising the gate to cover
prefill was **opt4: prefill −8~9%, REJECTED** (sync overhead + SM contention exceed the
overlap); (c) the dominant row is an allreduce sync point — overlap cannot hide a collective,
and overlap removes no launches on a host-bound path.

| group | base | LoRA | Δ | two-stream? | addressed by |
|---|---|---|---|---|---|
| **allreduce/comm** | 256 | **3654** | **+3398** | ✗ (sync point) | **host-side skew** (see host view) — **opt8 target** |
| permute (standalone) | 0 | 176 | +176 | ✗ (opt4) | (kernel asset on [PR#8](https://github.com/yushengsu-thu/sglang/pull/8), default-OFF, host-bound-blocked) |
| fused_moe (LoRA-Δ B-expand) | 0 | 100 | +100 | ✗ (opt4) | |
| LoRA MoE shrink | 0 | 58 | +58 | ✗ (opt4) | |
| align/sort (Triton re-sort) | 0 | 57 | +57 | ✗ (opt4) | ✅ opt5 took it 4×→2×/layer (was ~119) |
| LoRA qkv_b expand | 0 | 38 | +38 | ✗ (opt4) | |
| LoRA MoE expand | 0 | 36 | +36 | ✗ (opt4) | |
| LoRA shrink A | 0 | 33 | +33 | ✗ (opt4) | |
| activation (standalone) | 0 | 31 | +31 | ✗ (opt4) | (PR#8 fold asset, same condition) |
| LoRA expand B | 0 | 30 | +30 | ✗ (opt4) | |
| topk/pack | 0 | 11 | +11 | ✗ (opt4) | ✅ opt2 |
| expert GEMM (bmm) | 94 | 109 | +15 | — | — |
| base compute (fmha/nvjet/norm/finalize/routing) | 124 | 116 | −8 | — | — |
| **wall total** | **525** | **4558** | **+4033** | | |

LoRA compute extras sum to ~+570 µs/layer; **allreduce spin is +3398 µs/layer = 83% of the
lora prefill wall**. The host-bound cost manifests as comm spin (every sync point waits for
the slowest rank's CPU dispatch), not as visible idle.

## Host view (this round)

| metric | base | LoRA | source |
|---|---|---|---|
| prefill launches / 4096-tok chunk | 837 | 1532 | graph-on EXTEND window |
| prefill launches / layer-forward | 17.4 | 31.9 | ″ |
| prefill: graph-off wall vs GPU-busy | idle 53%, 8.1k launches | **idle 49%, 13.1k launches → HOST-BOUND** | `sanity_check_opt`, graph-off bs16 |
| prefill allreduce µs/call (graph-on) | 149 | **1672 (11×)** | rank-skew spin at every sync point |
| decode launches / step (graph-on) | 29 (graph replay) | **304** (replay + eager two-stream side work) | by-stage DECODE trace |
| eager compute share of prefill wall | — | ~9% (327 ms of 3588 ms; allreduce spin excluded) | graph-off |

## Expected-gain rule (sanity_check_opt only — no hand-estimates)

- **Any GPU-side µs/layer removal at prefill: e2e ceiling 0.0%** (host-bound absorption;
  tool verdict "DO NOT proceed on e2e grounds").
- **opt8 (host-side)**: lora chunk wall 219.7 ms → compute+comm floor ≈ 60 ms ⇒ **prefill
  tok/s ceiling ~3.6× (39k → ~140k; base 190k)**; e2e @bs16 ceiling ≈ +4% (prefill share
  0.84 s / 13.56 s) — above the ±2% noise floor.
- Kernel triage (config-vs-bandwidth): permute 14.5× theo / 11% occ (config-bound — PR#8 fold
  covers it off-branch); count_and_sort 166× but ~16 ms/prefill = 0.4% of wall (sub-noise);
  activation 3.6×. No on-branch GPU kernel clears the payoff rule.

## Ladder (opt8 onward; selection: dtype-common first, flag conventions per F)

| # | what | flag | dtype scope | status |
|---|---|---|---|---|
| **opt8** | piecewise CUDA graph for LoRA prefill — `server_args` condition 7 force-disables piecewise whenever `enable_lora`; runner already handles `lora_ids`, token ladder reaches 4096 | probe: `--enforce-piecewise-cuda-graph` (zero-code); productize as `SGLANG_OPT_LORA_*` if code needed | **common** (host/Python level; fp8/nvfp4 prefill is the same eager pipeline) | step0 probe queued (`dev/probe_opt8_piecewise.sh`) |

## Commit & code-size ledger (history, incl. no-e2e-win items)

Working branch (`qwen3-30b-a3b-2507-bf16`, PR#4) carries **opt1+2+5 only** (base `526e0ae22` →
`850faa87f`). Full pre-restructure history archived at
[`archive/qwen3-30b-a3b-2507-bf16-opt6-7-20260612`](https://github.com/yushengsu-thu/sglang/tree/archive/qwen3-30b-a3b-2507-bf16-opt6-7-20260612).
Post-reset health: acc KL 0.003727 < floor 0.004243; bench = opt5 baseline within noise.

| opt | where | lines | verdict |
|---|---|---|---|
| **opt1** align/sort fusion | [`869882a3a`](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea) | **+14/−5** (1 file) | ✅ decode +11% |
| **opt2** topk+pack | none (flag-only) | **0** | ✅ decode +5.6% |
| **opt3** lean info | [`1536c6e4e`](https://github.com/yushengsu-thu/sglang/commit/1536c6e4e65515f5ee7403c48b0726d55307d430) | +30/−11 (2 files) | ✗ **無 e2e 收益** (kept on branch, harmless) · [results](opt3/) |
| **opt4** two-stream prefill | none (flag experiment) | **0** | ✗ **無 e2e 收益** (prefill −8%, NOT adopted) · [results](opt4/) |
| **opt5** routing reuse | [`850faa87f`](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) | **+20** (2 files) | ✅ prefill +8~11% |
| **opt6** act-capture drop | [**PR #7**](https://github.com/yushengsu-thu/sglang/pull/7) (off-branch) | +183/−34 (5 files) | ✗ **無 e2e 收益** (sub-noise; mechanism verified, default-OFF) · [results](opt6/) |
| **opt7** in-MoE fold (probe+P0–P4) | [**PR #8**](https://github.com/yushengsu-thu/sglang/pull/8) (off-branch, stacked on #7) | +1,515/−34 (14 files) | ✗ **無 e2e 收益** (kernels −62% all-gates-PASS; host-bound-absorbed, default-OFF) · [results](opt7/OPT7.md) |

Notes: all SHIPPED perf (decode +11~12%, prefill +14~18% cumulative) comes from **34 lines**
(opt1 + opt5); opt2/opt4 were zero-code. The opt7 CUTLASS fold asset is correctness-proven and
flag-gated OFF on PR #8 — enable condition: prefill no longer host-bound (= opt8's target).
FP8/NVFP4 byte-identical throughout.

Method note: decode numbers = whole by-stage DECODE trace kernel totals ÷ (12 steps × 48
layers) (`dev/profile_decode_bystage.sh`; `--profile-start-step` is an ABSOLUTE scheduler
counter and cannot reach a decode window with the dev recipe). Do NOT window graph-replay
kernels with CPU NVTX spans — replayed kernels lag the host span and get diluted (an earlier
revision of this table undercounted expert GEMM 6.4 vs the true 39.6 µs/layer that way);
GPU-aligned spans fragment under the two-stream capture. Sanity: fmha = 48 calls/step in all
three cells. "GPU-active" = sum of kernel durations across streams (concurrency-blind — the
graph runs multiple streams, so it exceeds step-wall). Bench numbers taken DURING by-stage
profiling are invalid (profiler overhead: decode 1313 tok/s vs the real 7028/8103) — wall
verdicts come from `3_run_benchmark`/matrix runs only. Prefill numbers are 12 steady mid-run
`step[EXTEND bs=2 toks=4096]` chunks (eager path — no graph, CPU spans are safe there).
allreduce kept as its own row (spin-inflated at prefill — a host symptom, not GPU work).
