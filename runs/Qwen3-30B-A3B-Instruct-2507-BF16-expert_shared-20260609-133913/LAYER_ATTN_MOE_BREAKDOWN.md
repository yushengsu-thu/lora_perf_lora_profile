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

`url` = the optimization commit/PR that addresses this row (only landed, e2e-proven items).

> **Updated 2026-06-12**: opt6/opt7 were moved off the working branch into standalone PRs
> ([#7](https://github.com/yushengsu-thu/sglang/pull/7), [#8](https://github.com/yushengsu-thu/sglang/pull/8));
> the branch (and PR#4) is back to the opt1+2+5 state. This table keeps **measurements and landed
> optimizations only** — no planned/forecast entries. The decode table is decode-only and
> under-weights prefill-dominant items (permute); see the prefill view below.

## ① Attention sublayer:  base ~42 µs → LoRA ~55 µs  (+13 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | url |
|---|---|---|---|---|---|---|
| attention fmha | 10.4 | 10.4 | ~0 | 0 | — | |
| dense GEMM nvjet (qkv/o) | ~24 | ~24 | ~0 | 0 | — | |
| allreduce (after attn) | ~8 | ~8 | ~0 | 0 | — | |
| LoRA qkv_b expand | 0 | 4.2 | +4.2 | ~0 | cuBLAS expand | ✅ enabled (PR#4 baseline: `SGLANG_OPT_LORA_CUBLAS`, `SGLANG_OPT_LORA_QKV_B_STORE`, default-on) |
| LoRA shrink A (q/k/v/o part) | 0 | ~4.2 | +4.2 | ~0 | fuse q/k/v shrink into one GEMM + cuBLAS | ✅ enabled (PR#4 baseline: fused qkv shrink + `SGLANG_OPT_LORA_CUBLAS`) |
| LoRA expand B (o part) | 0 | ~5.0 | +5.0 | ~0 | cuBLAS; fuse fp32→bf16 cast into expand | ✅ enabled (PR#4 baseline: `SGLANG_OPT_LORA_CUBLAS` + fused cast) |

*(Remaining attention-LoRA ROI is low: prefill attn-LoRA is ~103 µs/layer vs the MoE side's much
larger removable cost.)*

## ② MoE sublayer:  base ~50 µs → LoRA ~96 µs  (+46 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | MoE-decomp extra — components (µs/layer) | url |
|---|---|---|---|---|---|---|---|
| MoE core (bmm expert GEMM / router / finalize) | 36.3 | 34.2 | ~0 | 0 | — | | |
| gate GEMM (nvjet) + allreduce (after MoE) | ~14 | ~14 | ~0 | 0 | — | | |
| routing (`routingCustom`) — **not LoRA-added** | 5.1 | 4.4 | −0.6 | 0 | — | | |
| **MoE-decomp extra** | 0 | **26.0** | **+26.0** | ~0 | align/sort fusion (opt1) + topk/pack + elem (opt2) | • **align/sort/scatter +10.2** (`moe_align_block_size_small_batch` 6.7 + `moe_lora_merged::fused_align_scatter` 3.5, latter LoRA-specific) — ✅ opt1<br>• **fused_moe +7.2** (LoRA-Δ B-expand GEMM producing gate_upΔ)<br>• **elem / copy / cast +3.9** (upcast / copy) — ✅ opt2<br>• **activation +3.2** (`moe::dev::activation`, standalone activation kernel)<br>• **topk / pack +3.0** (`_fused_virtual_topk_ids`) — ✅ opt2<br>• **permute +2.4** (`moe::dev::permute`, standalone row permute; **180 µs/layer at prefill**, see prefill view) | 1. ✅ [opt1 — align/sort fusion: decode +11%](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea) · [results](opt1/)<br>2. ✅ [opt2 — topk/pack: decode +5.6%](opt2/) _(flag-only, no code commit)_<br>5. ✅ [opt5 — prefill routing reuse: prefill +8~11%, align/sort 4×→2×](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) · [results](opt5/)<br>— history (no e2e win): see ledger below |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | | | |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | | | |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | | | |

(~+6 µs/layer of elementwise/reshape from the decomposed path is split across both sublayers. The
"MoE-decomp extra" components sum to ~+29 µs incl. routing≈0; the headline +26 µs excludes the LoRA
shrink/expand GEMM rows.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — MoE-decomp extra (~+26 µs, biggest item **align/sort +10.2 µs**,
  larger than any single LoRA GEMM) + MoE LoRA shrink/expand (+13 µs) + shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs. The decomp cluster is ~46% of the per-layer LoRA overhead.

## Prefill view (this is where the decomposed-path cost dominates; full data in [`journal_opti.md`](journal_opti.md) §4)
Per layer-forward, 4096-token chunk (`current_base_lora` bs16-TP0 trace): **base fused MoE ≈ 140 µs
vs LoRA decomposed ≈ 695 µs (5×)**, prefill tok/s = 19% of no-LoRA.

| item (µs/layer prefill) | cost | addressed by |
|---|---|---|
| standalone `permute` | **180** (largest single item; 3× GEMM1 itself; decode-shaped launch grid, 11% occupancy) | |
| standalone `activation` | 33 | |
| Triton re-sort ×4 (`moe_align`+`count_and_sort`) | **119** (routing already computed by trtllm-gen; decode has opt1's fused path, prefill falls back) | ✅ **opt5 DONE 2026-06-11: prefill +7.4~8.2% (single) / +9.4~11.1% (two) @bs16/32/64, decode flat; align/sort 4×→2×/layer (−50%), −2688 launches.** Remaining 2×/layer = genuinely different sorts (shared-outer A vs per-expert B). [commit 850faa87f](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) · [opt5/](opt5/) |
| LoRA-Δ GEMMs (shrink/fused_moe/expand) | 226 | |
| attention-LoRA GEMMs (qkv_b 39 + sgemm_a 34 + sgemm_b 30) | ~103 | (cuBLAS opts already on, low remaining ROI) |
| expert GEMMs + routing + finalize | ~137 | — |

Plus: prefill is **~half host-bound** (917 ms real wall vs ~340 ms compute kernels; 15.9k launches
vs base 8.1k, eager+serial) — launch-count reductions compound beyond the kernel-µs accounting.

## Commit & code-size ledger (history, incl. no-e2e-win items)

Working branch (`qwen3-30b-a3b-2507-bf16`, PR#4) carries **opt1+2+5 only** (base `526e0ae22` →
`850faa87f`). Full pre-restructure history is archived at
[`archive/qwen3-30b-a3b-2507-bf16-opt6-7-20260612`](https://github.com/yushengsu-thu/sglang/tree/archive/qwen3-30b-a3b-2507-bf16-opt6-7-20260612).

| opt | where | lines | verdict |
|---|---|---|---|
| **opt1** align/sort fusion | [`869882a3a`](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea) | **+14/−5** (1 file) | ✅ decode +11% |
| **opt2** topk+pack | none (flag-only) | **0** | ✅ decode +5.6% |
| **opt3** lean info | [`1536c6e4e`](https://github.com/yushengsu-thu/sglang/commit/1536c6e4e65515f5ee7403c48b0726d55307d430) | +30/−11 (2 files) | ✗ **無 e2e 收益** (kept on branch, harmless) · [results](opt3/) |
| **opt4** two-stream prefill | none (flag experiment) | **0** | ✗ **無 e2e 收益** (prefill −8%, NOT adopted) · [results](opt4/) |
| **opt5** routing reuse | [`850faa87f`](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) | **+20** (2 files) | ✅ prefill +8~11% |
| **opt6** act-capture drop | [**PR #7**](https://github.com/yushengsu-thu/sglang/pull/7) (off-branch) | +183/−34 (5 files) | ✗ **無 e2e 收益** (sub-noise; mechanism verified, default-OFF) · [results](opt6/) |
| **opt7** in-MoE fold (probe+P0–P4) | [**PR #8**](https://github.com/yushengsu-thu/sglang/pull/8) (off-branch, stacked on #7) | +1,515/−34 (14 files) | ✗ **無 e2e 收益** (kernels −62% all-gates-PASS; e2e host-bound-absorbed, default-OFF) · [results](opt7/OPT7.md) |

Notes: all SHIPPED perf (decode +11~12%, prefill +14~18% cumulative) comes from **34 lines**
(opt1 + opt5); opt2/opt4 were zero-code. The opt7 CUTLASS fold asset is correctness-proven and
flag-gated OFF on PR #8 — enable condition: prefill no longer host-bound. FP8/NVFP4 byte-identical
throughout (the working branch now contains zero bf16-launcher-only kernel code).

bs16 is latency / fixed-cost bound — which is why these small routing/align/elem kernels matter at
decode. allreduce excluded from GPU-active analysis (spin-wait inflated). Numbers are one steady
decode layer.
