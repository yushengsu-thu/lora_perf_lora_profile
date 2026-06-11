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

`url` = (to fill) the optimization commit/PR that addresses this row.

> **Updated 2026-06-11** (post-opt1/2/3 profile deep-read, see [`journal_opti.md`](journal_opti.md)):
> corrected which rows the in-MoE fold can actually absorb (LoRA-Δ GEMMs stay — FP8/FP4 don't fold
> them either), upgraded the fold scope to gather-prologue + EVT epilogue, and added the **prefill
> view** below — this table is decode-only and under-weights the prefill-dominant items (permute).

## ① Attention sublayer:  base ~42 µs → LoRA ~55 µs  (+13 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | url |
|---|---|---|---|---|---|---|
| attention fmha | 10.4 | 10.4 | ~0 | 0 | — | |
| dense GEMM nvjet (qkv/o) | ~24 | ~24 | ~0 | 0 | — | |
| allreduce (after attn) | ~8 | ~8 | ~0 | 0 | — | |
| LoRA qkv_b expand | 0 | 4.2 | +4.2 | ~0 | cuBLAS expand | |
| LoRA shrink A (q/k/v/o part) | 0 | ~4.2 | +4.2 | ~0 | fuse q/k/v shrink into one GEMM + cuBLAS | |
| LoRA expand B (o part) | 0 | ~5.0 | +5.0 | ~0 | cuBLAS; fuse fp32→bf16 cast into expand | |

*(No further attention-LoRA work planned in the opt4–opt7 ladder: the cuBLAS/fuse opts above are
already enabled and the remaining ROI is low — prefill attn-LoRA is ~103 µs/layer vs the MoE
side's ~330 µs of removable cost. Revisit only after opt7.)*

## ② MoE sublayer:  base ~50 µs → LoRA ~96 µs  (+46 µs)

| group (µs/layer) | base | LoRA | Δ | two-stream Δ | optimization | MoE-decomp extra — components (µs/layer) | url |
|---|---|---|---|---|---|---|---|
| MoE core (bmm expert GEMM / router / finalize) | 36.3 | 34.2 | ~0 | 0 | — | | |
| gate GEMM (nvjet) + allreduce (after MoE) | ~14 | ~14 | ~0 | 0 | — | | |
| routing (`routingCustom`) — **not LoRA-added** | 5.1 | 4.4 | −0.6 | 0 | — | | |
| **MoE-decomp extra** | 0 | **26.0** | **+26.0** | ~0 | in-MoE LoRA fold (bf16 = **NVFP4 sibling**, decomposed + missing-unfused-cubin wall → **NOT a port of FP8**; bf16-only CUTLASS grouped GEMM: **gather-prologue + SwiGLU·LoRA EVT epilogue**, no quant to fuse into) + fuse routing/align/topk/elem | • **align/sort/scatter +10.2** (`moe_align_block_size_small_batch` 6.7 + `moe_lora_merged::fused_align_scatter` 3.5, latter LoRA-specific) — ✅ opt1<br>• **fused_moe +7.2** (LoRA-Δ B-expand GEMM producing gate_upΔ — **NOT fold-absorbed**; the fold consumes a precomputed Δ, same as FP8/FP4's `gateUpLoraDeltaPtr`)<br>• **elem / copy / cast +3.9** (upcast / copy) — ✅ opt2<br>• **activation +3.2** (`moe::dev::activation`) — fold (opt7); prefill side-write killed first by **opt6** (`SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE`)<br>• **topk / pack +3.0** (`_fused_virtual_topk_ids`) — ✅ opt2<br>• **permute +2.4** (`moe::dev::permute`) — fold (opt7 gather-prologue, `SGLANG_OPT_BF16_MOE_GEMM1_FOLD`; **180 µs/layer at prefill**, see prefill view) | 1. [opt1 — align/sort fusion: decode +11% bs16](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea)<br>2. [opt2 — topk/pack: decode +5.6% bs16](https://github.com/yushengsu-thu/lora_perf_lora_profile/tree/main/runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913/opt2) _(flag-only, no code commit)_<br>3. [opt3 — drop elem/upcast + lean `_get_lora_info`: no clear win](https://github.com/yushengsu-thu/sglang/commit/1536c6e4e65515f5ee7403c48b0726d55307d430)<br>4. [opt4 — two-stream prefill A/B: NEGATIVE, keep 256](opt4/) _(flag-only, no code commit)_<br>5. [opt5 — prefill routing reuse: prefill +8~11%, align/sort 4×→2×](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce)<br>6. **planned: opt6–opt7** (see future-work ladder below) |
| LoRA MoE shrink (routed experts) | 0 | 9.2 | +9.2 | ~0 | **NOT a fold target** (FP8/FP4 keep it separate) — two-stream overlap / cuBLAS; planned: **opt7-②** prefill reads `permuted_hidden_bf16` (`SGLANG_OPT_BF16_MOE_SHRINK_PERMUTED`, bundled into opt7's pipeline) | | |
| LoRA shrink A (shared_expert part) | 0 | ~7.4 | +7.4 | ~0 | cuBLAS / fuse | | |
| LoRA MoE expand (routed experts) | 0 | 3.4 | +3.4 | ~0 | **NOT a fold target** (FP8/FP4 keep it separate) — two-stream overlap / cuBLAS | | |

(~+6 µs/layer of elementwise/reshape from the decomposed path is split across both sublayers. The
"MoE-decomp extra" components sum to ~+29 µs incl. routing≈0; the headline +26 µs excludes the LoRA
shrink/expand GEMM rows.)

## Where the +63 µs/layer goes
- **MoE sublayer ≈ +46 µs (73%)** — MoE-decomp extra (~+26 µs, biggest item **align/sort +10.2 µs**,
  larger than any single LoRA GEMM) + MoE LoRA shrink/expand (+13 µs) + shared_expert shrink (~+7 µs).
- **Attention sublayer ≈ +13 µs (21%)** — the q/k/v/o LoRA GEMMs.
- elementwise ≈ +6 µs. The decomp cluster is ~46% of the per-layer LoRA overhead.

## in-MoE fold vs (a) fusion — no double-counting (corrected 2026-06-11)
| absorbed by **in-MoE LoRA fold** | addressed only by **(a) fusion/removal** | **NOT absorbed by either** (stays; two-stream/cuBLAS) |
|---|---|---|
| `activation` (3.2) + `permute` (2.4) + the gate_up HBM round-trip ≈ **~5.6 µs decode** | `align/sort/scatter` (10.2) + `topk/pack` (3.0) + `elem/cast` (3.9) ≈ **~17 µs** — ✅ done (opt1+opt2) | LoRA-Δ GEMMs: `fused_moe` (7.2) + routed shrink/expand (~13) — FP8/FP4 don't fold these either; the fold consumes a precomputed Δ |

→ (a) fusion is **done** (opt1+opt2, decode +11% +5.6%). The fold's remaining **decode** win is
~5.6 µs/layer + 2 launches (launch-bound). Its real payoff is **prefill** (below) — the earlier
"~25 µs" figure wrongly counted the LoRA-Δ GEMMs as fold-absorbed (see `ref_bf16_opt.md` §9).

## Prefill view (added 2026-06-11 — this is where the fold pays; full data in [`journal_opti.md`](journal_opti.md) §4)
Per layer-forward, 4096-token chunk (`current_base_lora` bs16-TP0 trace): **base fused MoE ≈ 140 µs
vs LoRA decomposed ≈ 695 µs (5×)**, prefill tok/s = 19% of no-LoRA.

| item (µs/layer prefill) | cost | addressed by |
|---|---|---|
| standalone `permute` | **180** (largest single item; 3× GEMM1 itself) | fold **gather-prologue** (opt7) |
| standalone `activation` | 33 | fold EVT epilogue (opt7); **opt6** first kills its redundant `activation_lora_input` side-write (−50 MB/layer ≈ half its write traffic, 33 → ~20 µs expected) |
| Triton re-sort ×4 (`moe_align`+`count_and_sort`) | **119** (routing already computed by trtllm-gen; decode has opt1's fused path, prefill falls back) | ✅ **opt5 DONE 2026-06-11: prefill +7.4~8.2% (single) / +9.4~11.1% (two) @bs16/32/64, decode flat; align/sort 4×→2×/layer (−50%), −2688 launches.** Remaining 2×/layer = genuinely different sorts (shared-outer A vs per-expert B) — collapsing them = opt7's metadata pipeline. [commit 850faa87f](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce) · [opt5/](opt5/) |
| LoRA-Δ GEMMs (shrink/fused_moe/expand) | 226 | stays; **opt7-②** feeds the shrink from the base path's `permuted_hidden_bf16` (contiguous expert-grouped reads instead of its own gather) + two-stream overlap |
| attention-LoRA GEMMs (qkv_b 39 + sgemm_a 34 + sgemm_b 30) | ~103 | — (separate bucket; cuBLAS opts already on, low remaining ROI) |
| expert GEMMs + routing + finalize | ~137 | — (parity target for CUTLASS) |

Plus: prefill is **~half host-bound** (917 ms real wall vs ~340 ms compute kernels; 15.9k launches
vs base 8.1k, eager+serial) — launch-count reductions compound beyond the kernel-µs accounting.
Fold + routing-reuse projected: **695 → ~363 µs/layer MoE kernel time (−48%)**. Diagram:
[`in_moe_fold_before_after.png`](in_moe_fold_before_after.png).

## Shared bf16 activations — the bf16-unique reuse angle (2026-06-11)
In fp8/nvfp4, base GEMMs consume *quantized* inputs while LoRA GEMMs consume *bf16* — every
activation exists twice (quantized + bf16 capture). In the bf16 path they are the **same
tensor**, verified in code but not yet exploited:
- `permuted_hidden_bf16` (expert-grouped, padded — already grouped-GEMM layout) is directly
  readable by the gate_up LoRA shrink (**opt7-②**). fp8 never materializes it; nvfp4 materializes
  it as fp4 — unusable for LoRA.
- `activated_bf16` is **bitwise the same values** as the `activation_lora_input` side-capture
  (dev-kernel writes the same packed value to both; the fp8 variant divides by `scaleOut`,
  which is *why* fp8/fp4 must keep the capture) → bf16 can drop it (**opt6**).
- No scale bookkeeping in a bf16 epilogue — this property is what makes the CUTLASS-EVT fold
  (opt7) feasible for bf16 only.
- **Caveat:** sharing adds a dependency on main-stream permute/activation — at decode it would
  serialize what two-stream overlaps. Apply on the **prefill path only** (gate by token count).

## Future-work ladder (full detail in [`journal_opti.md`](journal_opti.md) §5)
Execution order: **opt4 ✗done → opt5 ✅done → opt6 → opt7-step0 (½ day) → opt7.**
Selection criteria: (1) prefer dtype-common fp8/nvfp4/bf16, (2) low code invasiveness,
(3) high ROI, (4) every step validated with the prefill/decode/e2e triplet + single×two
matrix, (5) **flag convention**: bf16-specific changes ship behind a
`SGLANG_OPT_BF16_<MODULE/KERNEL>` env flag; dtype-agnostic changes keep the
`SGLANG_OPT_LORA_*` namespace. All default-on like the existing family — an A/B baseline
must set the flag `=0` explicitly (unset = ON = non-measurement).

| # | what | flag | invasiveness | targets |
|---|---|---|---|---|
| **opt4** | two-stream-at-prefill A/B (`SGLANG_TWO_STREAM_MAX_TOKENS` 256→8192) | existing env (common) | **zero code** (flag A/B) | ✗ **DONE 2026-06-11, NEGATIVE — prefill −8~9% @bs16/32/64 (noise floor ±2%), decode flat; keep 256.** Confirms prefill bottleneck ≠ serialization → remove work (opt5), don't rearrange it. See `opt4/` |
| **opt5** | routing reuse at prefill (unify A/B stage routing block → cache hits) | `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE` (common) | Python/Triton, +1 line ([850faa87f](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce)) | ✅ **DONE 2026-06-11: prefill +8~11% both columns, decode flat, align/sort 4×→2× (−50%); acc at noise floor.** Dtype-agnostic — helps the FP8 deliverable. See `opt5/` |
| **opt7-②** | gate_up LoRA shrink reads `permuted_hidden_bf16` (prefill) | `SGLANG_OPT_BF16_MOE_SHRINK_PERMUTED` | Triton index logic (bf16-gated) | contiguous expert-grouped reads; Δ in permuted order; bundled into opt7's metadata pipeline |
| **opt6** | drop redundant `activation_lora_input` side-write (prefill) | `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE` | tiny `Bf16LoraLauncher`-internal .cu (FP8/NVFP4 *need* the capture — untouched) | −50 MB/layer HBM write; activation 33→~20 µs expected |
| **opt7-step0** | bf16 unfused-cubin probe (analogue of `sgl_trtllm_fp4_probe_unfused`, launcher.cu:4047) | — (diagnostic) | diagnostic only | decides fold route (a) wiring vs (b) CUTLASS |
| **opt7** | in-MoE fold: CUTLASS grouped GEMM, gather-prologue + SwiGLU·LoRA EVT epilogue | `SGLANG_OPT_BF16_MOE_GEMM1_FOLD` (+ `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT` for the weight copy) | high (new bf16-only kernel) | permute 180 + activation 33 µs/layer + gate_up HBM round-trip; **de-risk: prefill-only + dual-layout gemm1 weights (+9.7 GB/rank, affordable on GB300 288 GB — decode keeps the tuned cubin, zero regression)** |

## Recommended order (decode, bs16)
1. **in-MoE LoRA fold** — biggest structural win. Corrected accounting (2026-06-11): the fold-only
   decode remainder is **~5.6 µs/layer** (activation 3.2 + permute 2.4) + 2 launches — `fused_moe`
   7.2 is the LoRA-Δ B-expand GEMM and is **not** absorbed (the fold consumes a precomputed Δ, like
   FP8/FP4's `gateUpLoraDeltaPtr`); align/sort taken by opt1, topk/pack+elem by opt2. **The fold's
   real payoff is prefill** (permute 180 + activation 33 µs/layer + the gate_up HBM round-trip —
   see prefill view). **bf16 is the NVFP4 sibling**, NOT FP8: both bf16 and NVFP4 are
   decomposed and hit the *missing-unfused-cubin wall*, so this is **NOT a port of `sgl_fp8_moe.py`**
   (FP8 isn't truly "folded" — it only keeps permute fused). NVFP4 gets the fold by fusing
   activation+quant (`launchFusedActivationQuant` → `activated_bf16` never hits HBM); **bf16 has no
   quant**, so the equivalent is **one bf16-only CUTLASS grouped GEMM: gather-prologue (= fused
   permute, the original V1 epilogue-only framing misses the prefill-dominant permute) + SwiGLU·LoRA
   EVT epilogue** — kills standalone `permute` + `activation` + the `gate_up` HBM round-trip
   (ref §5/§6 + journal opt7). De-risked by **prefill-only dispatch + dual-layout gemm1 weights**
   (see the future-work ladder). Do **opt4** (two-stream-prefill A/B, done), **opt5/opt6**
   (routing-metadata + shared-buffer reuse) and the **opt7-step0 unfused-cubin probe** first —
   see the ladder above and `journal_opti.md` §5.
2. **align/sort/scatter fusion** (~10 µs) — largest (a) item; fixed-cost at small batch → decode benefits
   disproportionately. ✅ **DONE — [opt1](https://github.com/yushengsu-thu/sglang/commit/869882a3ab87ec3c1983f8808d382ef2aa1d0cea): decode +11.0/9.9/8.8% (bs16/32/64), e2e −9%, prefill flat; `moe_align_block_size_small_batch` 384→0 launches.** See `opt1/`.
3. **topk+pack single launch** (~3 µs) + **drop elem/upcast / `_get_lora_info`** (~4 µs) — PR #27329 /
   team action items; cheap, additive.
   - **topk+pack ✅ DONE — opt2** (`SGLANG_OPT_LORA_FUSED_TOPK_PACK`, already wired/default-on): fuses
     `_pack_topk_for_flashinfer_routed` (cast/`<<16`/`|`) into the gating kernel → **decode +5.6%/3.6%/3.1%
     (bs16/32/64)**, `BinaryFunctor` 576→0, `bitwise` 12→0, total launches 24178→21874. See `opt2/`.
   - **drop elem/upcast / `_get_lora_info` — investigated & measured (opt3): no clear win**
     ([opt3](https://github.com/yushengsu-thu/lora_perf_lora_profile/tree/main/runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913/opt3)).
     opt2 already removed the elem/copy bulk (the pack's cast/shift/or chain); residual
     activation-vec (`SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC`) + lean `_get_lora_info` gains are within
     run-to-run noise (prefill ~±few %, decode ~0). Low-ROI; remaining copies live in the decomposed
     `.cu` op → the in-MoE fold. **Next real headroom: in-MoE fold (the big ① item).**

bs16 is latency / fixed-cost bound — which is why these small routing/align/elem kernels matter at
decode. allreduce excluded from GPU-active analysis (spin-wait inflated). Numbers are one steady
decode layer.
