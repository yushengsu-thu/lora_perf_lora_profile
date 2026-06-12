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
- **Shared bf16 activations (bf16-unique advantage, 2026-06-11).** In fp8/nvfp4, base GEMMs
  consume *quantized* inputs while LoRA GEMMs consume *bf16* — every activation exists twice
  (quantized + bf16 capture). In the bf16 path the two are the **same tensor**, but the code
  doesn't yet exploit it:
  - the base path's `permuted_hidden_bf16` (expert-grouped, padded — exactly grouped-GEMM
    layout) is directly readable by the gate_up LoRA shrink (fp8 never materializes it;
    nvfp4 materializes it as fp4 — unusable for LoRA);
  - the base path's `activated_bf16` holds the same values as the `activation_lora_input`
    side-capture the activation kernel writes for the down-proj LoRA shrink — fp8/fp4 *must*
    write that bf16 copy (their activated output is quantized); for bf16 it's pure redundancy
    (~50 MB/layer extra HBM write at prefill, ≈half the activation kernel's write traffic);
  - and it is the reason the EVT fold (opt7) is feasible at all: bf16 accum + bf16 Δ → SwiGLU
    → bf16 out needs no per-expert/per-token scale bookkeeping in the epilogue.
  **Caveat:** sharing adds a dependency on the main-stream permute/activation — at decode this
  would serialize what two-stream currently overlaps. Apply the sharing on the **prefill path
  only** (gate by token count); decode keeps the current side-stream structure.

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

Selection criteria (user directive 2026-06-11): (1) prefer dtype-common fp8/nvfp4/bf16,
(2) low code invasiveness, (3) high ROI, (4) validate every step with the prefill/decode/e2e
triplet + single×two matrix, (5) **flag convention** — bf16-specific changes ship behind a
`SGLANG_OPT_BF16_<MODULE/KERNEL>` env flag (e.g. `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE`,
`SGLANG_OPT_BF16_MOE_SHRINK_PERMUTED`, `SGLANG_OPT_BF16_MOE_GEMM1_FOLD`); dtype-agnostic
changes keep the `SGLANG_OPT_LORA_*` namespace (e.g. opt5
`SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE`). All default-on like the existing family — A/B
baselines must set `=0` explicitly.

### opt4 (was opt4) — two-stream-at-prefill A/B (flag-only) — ✗ DONE 2026-06-11, NEGATIVE
Prefill is always serial today (`SGLANG_TWO_STREAM_MAX_TOKENS` defaults 256 < 4096-tok chunks),
so the whole LoRA-Δ chain (~345 µs/layer incl. re-sort) sits on the main stream. Raising the
gate to 8192 (zero-code A/B, `opt4/`) was measured: **prefill −8~9% at all of bs16/32/64**
(two column: 91.4/91.3/92.5% ON/OFF), decode flat, noise floor (single column, identical
cells) ±0–2%. Two-streaming the 4096-token chunks adds side-stream sync overhead + SM
contention that exceeds the overlap benefit. **Verdict: keep the 256 default; no change.**
Informative loss: the prefill bottleneck is NOT serialization — work must be *removed*
(kernels + launches), not rearranged. opt5's priority is strengthened. Results: `opt4/`.

### opt5 (✅ done) + opt6 (next) — routing-metadata + shared-buffer reuse at prefill
**Scope upgraded 2026-06-11** (shared-bf16-activation insight, §1 last bullet): not just reuse
the trtllm routing *metadata* — reuse the base path's bf16 *data buffers* too. Prefill-only
(decode keeps the current two-stream structure; see §1 caveat). Three pieces:
1. ✅ **opt5 DONE 2026-06-11** — implemented as a 1-line unify of the A (shrink) stage's
   routing BLOCK_SIZE_M with the B stage's at prefill (≥512 tokens), making the per-layer
   `routing_cache` key match across stages. Commit
   [`850faa87f`](https://github.com/yushengsu-thu/sglang/commit/850faa87fbcc7d54210bc86866d2f9b3ecf4abce)
   (pushed to PR #4 branch). **Result: prefill +7.4~8.2% (single) / +9.4~11.1% (two-stream)
   @bs16/32/64, decode flat, e2e −0.2~2%; align/sort 4×→2×/layer (−50% kernel time, −2688
   launches); acc KL at the vLLM noise floor.** The remaining 2×/layer are genuinely
   different sorts (shared-outer A routes by lora id, per-expert B by expert id) — collapsing
   them needs the trtllm-metadata integration (opt7's pipeline). Honest cost: shrink kernel
   +1.9 ms/window from the tile change (32→64), far outweighed. Results: `opt5/`.
   Flag: `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE` (common namespace — dtype-agnostic).
2. **gate_up LoRA shrink reads `permuted_hidden_bf16`** (the base permute output) instead of
   re-gathering raw hidden via its own sorted ids: contiguous expert-grouped reads, and the Δ
   comes out in permuted order (simplifies the activation/epilogue indexing).
   Flag: `SGLANG_OPT_BF16_MOE_SHRINK_PERMUTED` (bf16-only; bundle into opt7's pipeline).
3. ✗ **opt6 DONE 2026-06-11 — NO CLEAR WIN.** Implemented (commits `cf9d0e55e` mechanism,
   `f4971ea4e` default-off): down shrink reads `activated_bf16` via an exported
   expanded→permuted row map; the activation kernel skips the side-capture. **Mechanism
   verified**: activation 32.9→21.7 µs/call (−34%, matching the half-write-traffic
   prediction), map-read shrink ≈0 extra cost, acc KL 0.003530 at the vLLM noise floor.
   **But the bench is flat** (prefill 97.3~101.6% both columns, decode flat): the ~5 ms/
   prefill saving is below the ±2% noise floor and the map D2D export adds one launch on
   the host-bound prefill path. Honest math correction: the saving was always <1% — the
   earlier "+2~4%" hope was wrong. **Flag default False** (off = byte-identical); the
   permuted-read + row-map plumbing is exactly what the opt7 fold pipeline needs and is
   now proven at the noise floor. Results: `opt6/` (profiles in-pod only — flaky uplink).
   Flag: `SGLANG_OPT_BF16_MOE_ACT_DROP_LORA_CAPTURE` (bf16-only).
Pieces 1–2 are LoRA Python/Triton-layer; an order of magnitude simpler than the fold. All
three are bf16-unique sharing wins except 1, which is dtype-agnostic (fp8/nvfp4 prefill runs
the same native-align fallback — fixing it benefits the FP8 deliverable too).

### opt7-step0 (✅ done 2026-06-11) — bf16 unfused-cubin probe: route (b) CONFIRMED
Write the bf16 analogue of `sgl_trtllm_fp4_probe_unfused` (launcher.cu ~L4047):
Bfloat16/Bfloat16 + Swiglu + `unfuseActForLora=true`, check `getValidConfigIndices()`.
`>0` ⇒ route (a) (trtllm-gen cubin exists, just wire it); `-1` ⇒ route (b) below.
Build/run on GB300; expected `-1` (same wall NVFP4 hit).
**Result (GB300, H=2048/I=768/topk8/32 local experts): [1] unfused BlockMajorK = −1, [2]
unfused MajorK = −1, [3] Identity = −1 — at every tile (8–128) and both 16/4096 tokens;
[0] fused sanity = 144/64/4 configs. Route (a) is dead; opt7 = route (b) CUTLASS grouped
GEMM. Probe commit `d12fe74a7`; results `opt7_step0/PROBE.md`.**

### opt7 — the in-MoE fold (IN PROGRESS: design + P0 done 2026-06-12, see opt7_design/)
Flags: `SGLANG_OPT_BF16_MOE_GEMM1_FOLD` (+ `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT` for the weight copy).
**Progress**: `opt7_design/OPT7_DESIGN.md` (phases P0–P4; exact fold semantics incl. the
interleaved-GEMM-cols vs half-contiguous-Δ trap; P1 has a hard 57 µs parity gate vs the tuned
bmm cubin). **P0 done** (sglang `1a82c2111`+`7cea5ed86`): CUTLASS 4.5 include path wired into
the JIT module (probe [4,5,1]); naive reference fold kernel pins the semantics — unit test vs
torch fp32 reference PASS (max rel err 3.8e-3, `dev/test_bf16_fold_ref.py`). **P1 done 2026-06-12 — PARITY GATE PASS**: CUTLASS Sm100 ptr-array grouped GEMM
(2SM UMMA, 256×128×64 cluster tile, device-built group args, sglang `7e1e69eb6`).
At the CORRECT per-rank shapes (EP4: 8192 expanded rows/rank, E=32, N=1536, K=2048):
**65.6 µs end-to-end (~52 µs pure kernel) vs the 57 µs tuned cubin (gate ≤68.4) — PASS**;
cuBLAS bmm = 44.8 µs shows ~20% further tuning headroom. NOTE the first gate run "MISSED"
at 172 µs because the bench forgot the EP4 divide (32768 rows = 4× real work; 57 µs at that
size would be 3.6 PF/s = 3× cuBLAS, impossible). Debug ladder that got it working: global
`Tensor` name clash (TU split from tvm-ffi), explicit PtrArray schedules (Auto picks the
non-array mainloop), ArchTag Sm100 (builder rejects Sm103 for dense ptr-array),
hw_info.sm_count (persistent scheduler needs it or run() = kErrorInternal pre-launch).
**P2 step1 done (`aec7c2fc1`)**: standard EVT is elementwise-only — the 2:1 interleaved
column fold needs a forked epilogue collective. Fork base:
`sm100_epilogue_array_nosmem.hpp` (TMEM→reg→direct gmem; pairs are intra-thread). Cost
calibration: NoSmem full-width epilogue 84.4 µs vs TMA 65.6 µs (+19 µs) — the folded
version stores HALF the bytes and replaces the 33 µs activation kernel + ~6 µs gate_up
round-trip, so net fold value stays ~+20 µs/layer even before tuning. P1-parity config
kept on the TMA epilogue meanwhile.
**P2 DONE 2026-06-12 (`f2247b5a8`) — fold epilogue WORKS & WINS**:
`sgl_bf16_fold_epilogue.hpp` (fork of the Sm100 NoSmem array collective, wrapped in
`Sm100TmaWarpSpecializedAdapter` exactly like the stock WarpSpecialized variant) folds the
interleaved (g,u) accumulator pairs + half-contiguous LoRA Δ (per-row gather via the
per-group perm2exp segment pointers) + silu directly in the epilogue, half-width D[R,768].
**Correctness: BITWISE identical to the P0 ref kernel (max_abs=0.0). Perf: 84.8 µs vs the
≤99 µs net-win gate (P1 GEMM 65.6 + activation 33 it replaces) — PASS.** v1 scalar fold
loop was 183.6 µs; the chunked rewrite (per-row hoist, 8B vector Δ loads + 8B vector
stores) won 54%. opt6's capture-drop is subsumed (no aux write exists at all).
Debug ladder addenda: adapter requirement (`ThreadEpilogueOp`/`EpilogueTile` aliases,
pointer-type stride template param), detached `setsid` runs for flaky-uplink test cycles.
**P3 DONE 2026-06-12 (`f7e5d51`) — the 180 µs "prize" fell to a 30-line kernel.**
Value-audit first: the dev permute moves only ~8 µs of HBM traffic — its 180 µs is a
decode-shaped launch grid (128 blocks, 11% occupancy) starved at prefill shapes. No cpasync
mainloop surgery needed: a properly-gridded 16B-vectorized row gather runs **12.7 µs**
(bitwise vs torch index_select). **Full fold pipeline (gather 12.7 + fold GEMM 85.0) =
101.5 µs vs the 270 µs it replaces (permute 180 + GEMM1 57 + activation 33) — −62%,
gate ≤189 PASS.** All three opt7 kernel pieces (P1 parity, P2 fold, P3 gather) are done.
Next: **P4 integrate** — Bf16LoraLauncher branch behind SGLANG_OPT_BF16_MOE_GEMM1_FOLD
(+_DUAL_LAYOUT weight re-prep at load: plain [E,2I,K] from the pre-shuffle source),
prefill-only dispatch, down-shrink via the opt6 map path → acc → bench triplet+matrix →
upload opt7/.
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
  counts) integration into CUTLASS Grouped GEMM (shared with opt5 piece 2 — same pipeline).
- **De-risking strategy: prefill-only fold + dual-layout gemm1 weights (added 2026-06-11).**
  Keep BOTH weight layouts at load: trtllm shuffled+BlockMajorK (decode keeps the tuned bmm
  cubin — zero regression risk, it's launch-bound and already good) and a CUTLASS layout
  (prefill fold), dispatched by token count. Cost: gemm1 duplicate ≈ **+9.7 GB/rank**
  (48 layers × 32 local experts × 1536×2048 × 2 B) — affordable on GB300 288 GB (model is
  only ~15.3 GB/rank). This is a trade **only bf16 can afford 1:1** (an unquantized layout
  copy of an fp4 model would be 4× its base weights). CUTLASS then only has to win at
  prefill sizes — where the permute+activation+round-trip savings dominate and the tuned
  cubin's edge is smallest. Failure mode is safe: if CUTLASS doesn't win at prefill, don't
  switch; decode is never touched.
- Enabler: the shared-bf16-activation property (§1) — the EVT epilogue needs no quant scale
  bookkeeping, which is exactly what makes this fold bf16-only feasible.
- Isolation: new bf16-only kernel + `Bf16LoraLauncher` changes only — FP8/NVFP4 untouched.

**Diagram** (current path vs fold, with measured per-layer costs):
[`in_moe_fold_before_after.png`](in_moe_fold_before_after.png)

## 6. Index

- `LAYER_ATTN_MOE_BREAKDOWN.md` — per-layer decode cost map (opt1/2/3 links).
- `opt1/ opt2/ opt3/` — each: OPT\*.md, summary.md, matrix png, graph-off profiles.
- `current_base_lora/` — base-vs-LoRA bench (bs16/32/64) + graph-on profiles TP0–3.
- `HANDOFF.md` — session handoff (2026-06-10).
- Harness: `tune-lora-perf/dev/` (GB300-only; warm pod `bf16test-20260607` on node 6zvh).
