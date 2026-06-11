# Optimization Journal — bf16 MoE-LoRA (`experimental_sgl_trtllm`)

Running log of the bf16 LoRA-path optimization work, from the first profile to the
current state, plus the planned next steps.

- **Target**: Qwen3-30B-A3B-Instruct-2507-BF16, expert_shared (adapter `alpha`, rank 32),
  GB300, TP4/EP4, in=out=2048, decode bs16/32/64.
- **Code under test**: https://github.com/yushengsu-thu/sglang/pull/4
  (fork branch `qwen3-30b-a3b-2507-bf16`; base `526e0ae22` → opt1 `869882a3a` → opt3 `1536c6e4e`).
- **Hard constraint**: FP8 (Qwen3-VL/3.5) and NVFP4 (Kimi K2.5) are the contractual
  deliverables — bf16 work must be additive/isolated, byte-for-byte safe for those paths.
- **Metric rules**: judge by **decode tok/s** (run-to-run variance ~0.06%) and e2e; prefill
  tok/s is noisy (~6–14%) — treat prefill deltas <~10% as noise, and a real prefill win must
  appear in BOTH single- and two-stream columns (prefill is stream-independent).
  Always report the prefill / decode / e2e triplet.

---

## 1. Architecture findings (the mental model)

- All three LoRA paths (fp8 / nvfp4 / bf16) are **decomposed** (standalone
  `moe::dev::permute` / `activation` / `finalize`), unlike no-LoRA which runs the fused
  trtllm-gen `MoE::Runner` (permute+SwiGLU baked into the GEMM1 cubin).
- The decisive split is the `fusedAct` cubin:
  - **FP8**: permute stays fused in PermuteGemm1; activation is *forced* unfused anyway
    (weight-shuffle constraint), so the LoRA delta hooks into an existing kernel for free.
  - **NVFP4 + bf16**: the unfused-activation gated GEMM1 cubin **does not exist**
    (the "missing-unfused-cubin wall"; FP4 probe = `known dead (-1)`), so both fall all the
    way back to raw `Gemm2::Runner` + standalone permute + standalone activation.
    `Bf16LoraLauncher` (launcher.cu ~L3427) is literally the FP4 launcher minus the two
    quant stages.
- ⇒ **bf16 ≈ NVFP4 structurally, NOT FP8.** NVFP4's biggest wins are quant-fusion
  (fused permute+quant, fused activation+quant via `launchFusedActivationQuant` — the
  activated tensor never hits HBM). bf16 has no quant step, so those are **not portable**.
  bf16's analogue of "fuse into the mandatory next step" is **fuse SwiGLU+LoRA into the
  GEMM itself** (the in-MoE fold, §5).
- Two-stream is **default-on for decode** (`SGLANG_TWO_STREAM_MAX_TOKENS=256`, installed
  whenever `SGLANG_EXPERIMENTAL_LORA_OPTI=1`); prefill (2048 > 256) is always serial.

## 2. Optimizations done

### opt1 — align/sort fusion ✅ SHIPPED (the real win)
At decode the bf16 shared_outer path fell to the unfused routing-align pair
(`_fused_virtual_topk_ids` + `moe_align_block_size_small_batch`, ~10.2 µs/layer). The fused
single-launch `moe_lora_merged::fused_align_scatter` already supported shared_outer — it was
just gated off in Python. **2-line change** in `virtual_experts.py::_get_routing`
(gate `not shared_outer and ep_local` → `(shared_outer or ep_local)`; `compact=not shared_outer`).
- Flag `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN` (**default True**). Commit `869882a3a`.
- **Decode +11.0 / +9.9 / +8.8 % (bs16/32/64), e2e −9~10 %, prefill flat.**
- Routing proven bitwise-equivalent to the fallback for bf16/FP8 (128/EP4) AND NVFP4/Kimi
  (384/EP8) shapes → dtype-independent → FP8/NVFP4 unaffected. Results: `opt1/`.

### opt2 — topk+pack single launch ✅ (flag-only)
`_pack_topk_for_flashinfer_routed` (cast/`<<16`/`|` chain) fused into the gating kernel
(`fused_topk(packed_out=…)` → `StandardTopKOutputPacked`); dispatch reuses `packed_topk_ids`.
Already wired; this model meets the gate (128 experts pow2, softmax, no bias/shared-experts).
- Flag `SGLANG_OPT_LORA_FUSED_TOPK_PACK` (**default True**).
- **Decode +5.6 / +3.6 / +3.1 %.** Launches 24178 → 21874. Results: `opt2/`.

### opt3 — lean `_get_lora_info` + elem/upcast cleanup ✗ no clear win
Cached layer-static scalars (`SGLANG_OPT_LORA_LEAN_INFO`, default True, commit `1536c6e4e`).
opt2 had already removed the elem/copy bulk; `_get_lora_info` is CPU/capture-time only under
cuda-graph. Prefill within noise, decode ~0. Honest negative result: `opt3/`.

### Common opt-in flags audit (NVFP4 ∩ bf16) ✗ no clear win, kept
`SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1` + `SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1` enabled in
model.env (bitwise-safe, kept). NVFP4/Kimi-only flags are no-ops for bf16 Qwen-128.
bf16 already adopted all of FP4's dtype-agnostic launcher tricks (permute-buffer reuse,
skip padded-row memset, activation-vec wiring). **No cheap common headroom left.**

## 3. Current numbers (`current_base_lora/`, two-stream decode)

| cell | bs | prefill tok/s | decode tok/s | e2e s |
|---|---|---|---|---|
| base (no-LoRA) | 16 | 188,142 | 3,933 | 8.51 |
| base | 32 | 192,148 | 6,920 | 9.81 |
| base | 64 | 193,852 | 11,714 | 11.87 |
| LoRA | 16 | 35,751 | 2,574 | 13.65 |
| LoRA | 32 | 36,331 | 4,686 | 15.79 |
| LoRA | 64 | 36,647 | 8,088 | 19.78 |

- LoRA / base: **decode 65.5 / 67.7 / 69.0 %**, **prefill ~19 %**, e2e 1.60–1.67×.
- Cumulative opt effect (vs pre-opt1): decode **+11 / +10.5 / +12.7 %**, e2e −9~10 %.
- **The remaining gap is prefill (5.3× slower than no-LoRA).**

## 4. Profile deep-read (2026-06-11, `current_base_lora/profile` bs16-TP0)

Method note: decode steps are cuda-graph replays whose kernels share one correlation id —
bucket kernels by the `step[…]` time windows (8× EXTEND 4096-tok chunks + 8× DECODE bs16),
not by correlation dedup.

**Per layer-forward MoE cost (prefill, 4096-tok chunk):**

| component | base (fused) | LoRA (decomposed) |
|---|---|---|
| expert GEMMs (tuned bmm cubins) | 111 µs (72+39, permute+SwiGLU folded in) | 114 µs (57+57, raw) |
| `moe::dev::permute` (standalone) | — | **180 µs** |
| `moe::dev::activation` (standalone) | — | 33 µs |
| Triton re-sort ×4 (`moe_align` + `count_and_sort`) | — | **119 µs** |
| LoRA-Δ GEMMs (shrink / fused_moe / expand) | — | 226 µs |
| routing + finalize | ~29 µs | ~23 µs |
| **total** | **~140 µs** | **~695 µs (5×)** |

Key findings:
1. **Standalone permute (180 µs/layer) is the single biggest LoRA-only kernel** — 5.5× the
   activation, 3× GEMM1 itself. Pure data movement (gather of 4096×8 expanded rows × 2048
   hidden, write + re-read through HBM).
2. **The Triton LoRA-Δ chain re-sorts routing 4×/layer at prefill** (~46 ms per full prefill)
   while trtllm-gen routing has already computed the same metadata. Decode is already saved
   by opt1's `fused_align_scatter` (13 µs); prefill falls back to native align.
3. **Prefill is ~half host-bound**: real prefill wall ≈ 917 ms (bench) vs ~340 ms of compute
   kernels in-window; 15.9k launches vs base 8.1k, eager and serial (two-stream covers decode
   only). Kernel-µs-only accounting undercounts any fix that also cuts launch count.
4. Decode MoE LoRA-extra is now small post-opt1/2: permute 2.8 + activation 2.1 µs/layer
   (+ the fold would also drop 2 launches/layer; decode is launch-bound).
5. Attention-LoRA prefill ≈ 103 µs/layer (qkv_b 39 + sgemm_a 34 + sgemm_b 30) — separate
   bucket, untouched by the MoE fold.

## 5. Future work (priority order)

### F1 — routing-metadata reuse at prefill (cheap, do first)
Eliminate the Triton chain's 4×/layer re-sort by reusing the trtllm routing metadata (or by
extending the opt1 fused align/scatter to the large-batch path). Targets **−119 µs/layer
kernel time (~46 ms/prefill) and −8 launches/layer** — likely an order of magnitude simpler
than the fold, isolated to the LoRA Python/Triton layer.

### F2 — bf16 unfused-cubin probe (decides fold route a vs b)
Write the bf16 analogue of `sgl_trtllm_fp4_probe_unfused` (launcher.cu ~L4047):
Bfloat16/Bfloat16 + Swiglu + `unfuseActForLora=true`, check `getValidConfigIndices()`.
`>0` ⇒ route (a) (trtllm-gen cubin exists, just wire it); `-1` ⇒ route (b) below.
Build/run on GB300; expected `-1` (same wall NVFP4 hit).

### F3 — the in-MoE fold (the big one; bf16-only, additive)
Replace `permute + GEMM1 + activation` with **one CUTLASS grouped GEMM**:
- **Prologue**: gather A-operand rows via `expanded_idx_to_permuted_idx`
  (= fused permute — **scope upgrade from the original V1 epilogue-only framing**, justified
  by finding §4.1: permute is the dominant prefill item, 180 vs 33 µs).
- **EVT epilogue**: + `gate_up_lora_delta` (aux input) → SwiGLU → write only `activated`
  `[P, 768]` (+ `activation_lora_input` for the down-proj LoRA shrink).
- Kills per layer: permute kernel (P180/D2.8 µs), activation kernel (P33/D2.1 µs), the
  `permuted_hidden` [P,2048] and `gate_up` [P,1536] HBM round-trips (~25 MB W+R per layer at
  4096 tokens), 2 launches. Interleaved (g,u) column pairs are co-resident in one CTA N-tile —
  not a blocker.
- Projected prefill MoE kernel time: **695 → ~363 µs/layer (−48 %)**, before the compounding
  host-side launch savings. LoRA-Δ GEMMs (226 µs) stay — FP8/FP4 don't fold them either;
  they remain two-stream/overlap targets.
- Risks: hand-rolled CUTLASS grouped GEMM must approach the tuned `bmm_Bfloat16` cubin
  (57 µs target); gemm1 weights need a CUTLASS-friendly layout re-prep at load (BlockMajorK
  is trtllm-gen-specific); routing-metadata (`cta_idx_xy_to_batch_idx`, dynamic per-expert
  counts) integration into CUTLASS Grouped GEMM.
- Isolation: new bf16-only kernel + `Bf16LoraLauncher` changes only — FP8/NVFP4 untouched.

**Diagram** (current path vs fold, with measured per-layer costs):
[`in_moe_fold_before_after.png`](in_moe_fold_before_after.png)

## 6. Index

- `LAYER_ATTN_MOE_BREAKDOWN.md` — per-layer decode cost map (opt1/2/3 links).
- `opt1/ opt2/ opt3/` — each: OPT\*.md, summary.md, matrix png, graph-off profiles.
- `current_base_lora/` — base-vs-LoRA bench (bs16/32/64) + graph-on profiles TP0–3.
- `HANDOFF.md` — session handoff (2026-06-10).
- Harness: `tune-lora-perf/dev/` (GB300-only; warm pod `bf16test-20260607` on node 6zvh).
