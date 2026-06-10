================================================================================
HANDOFF / WARMUP — bf16 MoE-LoRA performance optimization (Qwen3-30B-A3B)
Prepared for: Fable 5 (taking over the next task)
Date: 2026-06-10
================================================================================

0. TL;DR
--------------------------------------------------------------------------------
We are optimizing the **bf16** LoRA path of SGLang's `experimental_sgl_trtllm`
MoE-LoRA backend (PR https://github.com/yushengsu-thu/sglang/pull/4). Target model:
Qwen3-30B-A3B-Instruct-2507-BF16, expert_shared (adapter `alpha`, rank 32), GB300,
TP4/EP4, decode bs16/32/64.

Done so far (3 optimizations + a common-flags audit):
  - opt1 (align/sort fusion): **decode +11% @bs16** — SHIPPED (2-line python change).
  - opt2 (topk+pack single launch): **decode +5.6%** — flag-only, already wired.
  - opt3 (drop elem/upcast + lean _get_lora_info): **no clear win** (subsumed by opt2).
  - common opt-in flags (MOE_ALIGN + ACTIVATION_VEC): no clear win, kept (safe).

Cumulative (after vs before, two-stream decode): **+11/+10.5/+12.7% @bs16/32/64**,
e2e -9~10%. LoRA decode is now 66-69% of the no-LoRA ceiling; **prefill is still
only ~18% of no-LoRA (5.7x slower)** — that gap is the next big target.

NEXT TASK (recommended): **the in-MoE fold** (bf16-specific, CUTLASS-EVT GEMM1
epilogue). It is the only substantive headroom left and it directly attacks the
prefill gap. Details in section 7.

================================================================================
1. PROJECT CONTEXT
--------------------------------------------------------------------------------
- sglang source (local): /Users/yushengsu/Downloads/tml/sglang
  fork: github.com/yushengsu-thu/sglang  branch: qwen3-30b-a3b-2507-bf16
  base commit: 526e0ae22 (= the LAYER_ATTN_MOE_BREAKDOWN profile commit)
  HEAD now: 1536c6e4e  (526e0ae22 -> 869882a3a opt1 -> 1536c6e4e opt3-lean_info)
- dev harness (local): /Users/yushengsu/Downloads/tml/tune-lora-perf/dev
- results repo: github.com/yushengsu-thu/lora_perf_lora_profile
  run dir: runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913/
- reference investigation doc: /Users/yushengsu/Downloads/tml/ref_bf16_opt.md  (READ THIS)
- per-layer cost breakdown (the optimization map, kept updated with opt1/2/3 links):
  .../runs/.../LAYER_ATTN_MOE_BREAKDOWN.md
- Laptop is macOS, NO CUDA. ALL build/bench/profile run on GB300 via the dev/ harness.

Key code files (sglang):
  python/sglang/srt/lora/trtllm_lora_temp/lora_dispatch.py   - fp8/bf16/fp4 LoRA dispatch fns
  python/sglang/srt/lora/trtllm_lora_temp/moe_overlap.py     - TWO-STREAM dispatch (the DEFAULT decode path!)
  python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py - _get_routing, merged_experts_fused_moe_lora_add (opt1 lives here)
  python/sglang/srt/lora/trtllm_lora_temp/lora_layer.py      - bf16/fp4/fp8 quant-info build + dispatch routing
  python/sglang/srt/lora/trtllm_lora_temp/environ.py         - lora_envs flags (gated by SGLANG_EXPERIMENTAL_LORA_OPTI)
  python/sglang/srt/lora/layers.py:_get_lora_info            - opt3 lean change
  python/sglang/srt/layers/moe/topk.py                       - gating + fused topk+pack (opt2), KIMI_GATE
  python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu
        - Bf16LoraLauncher (~L3427), FP4BlockScaleLoraLauncher (~L2758), Fp8BlockScaleLauncher (~L973)
  python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu - the fused align/scatter kernel

================================================================================
2. KEY ARCHITECTURE FINDINGS (the mental model)
--------------------------------------------------------------------------------
- All three LoRA paths (fp8/fp4/bf16) are DECOMPOSED (use moe::dev::permute /
  activation / finalize), unlike no-LoRA which uses the FUSED trtllm-gen MoE::Runner.
- The decisive split:
    * FP8: keeps permute FUSED in PermuteGemm1; only activation is forced-unfused
      (weight-shuffle constraint) so the LoRA delta hooks into an already-existing
      activation kernel "for free".
    * NVFP4 + bf16: hit the "missing-unfused-cubin wall" - the unfused-activation
      GATED GEMM1 cubin does not exist for them, so they fall ALL the way back to
      raw Gemm2::Runner + standalone permute + standalone activation. Bf16LoraLauncher
      is literally written as the FP4 sibling.
- => bf16 ~= NVFP4 structurally, NOT fp8. You can reference NVFP4's launcher
  structure / plumbing, but NVFP4's biggest wins are QUANT-FUSION
  (fused permute+quant, fused activation+quant so activated_bf16 never hits HBM)
  which bf16 CANNOT use (no quant step). bf16's equivalent of "fuse into the
  mandatory next step" is "fuse SwiGLU+lora into a GEMM epilogue" (the in-MoE fold).

- TWO-STREAM is DEFAULT-ON for decode: SGLANG_TWO_STREAM_MAX_TOKENS defaults 256,
  and install_two_stream_overrides() runs whenever SGLANG_EXPERIMENTAL_LORA_OPTI=1
  (it is NOT gated by SGLANG_LORA_TWO_STREAM - that docstring is stale). So decode
  (bs <=256 tokens) is two-stream; prefill (in=2048 > 256) is always serial.
  single-stream baseline = set SGLANG_TWO_STREAM_MAX_TOKENS=0.

================================================================================
3. OPTIMIZATIONS DONE (each: what / where / flag / commit / result)
--------------------------------------------------------------------------------
opt1 - align/sort fusion  [SHIPPED, the real win]
  What: at decode the bf16 shared_outer path fell to the unfused routing-align pair
        (_fused_virtual_topk_ids + moe_align_block_size_small_batch, ~10.2us/layer).
        The fused single-launch align/scatter kernel (moe_lora_merged_align /
        fused_align_scatter) ALREADY supported shared_outer; it was just gated off in
        python. Opened it.
  Change (python only): virtual_experts.py _get_routing gate
        `... and not shared_outer and ep_local ...` -> `... and (shared_outer or ep_local) ...`
        and `compact=True` -> `compact=not shared_outer`.
  Flag: SGLANG_OPT_LORA_FUSED_MERGED_ALIGN (DEFAULT TRUE).
  Commit: 869882a3ab87ec3c1983f8808d382ef2aa1d0cea (pushed to fork).
  Result: decode +11.0/+9.9/+8.8% (bs16/32/64), e2e -9~10%, prefill flat. Routing
        bitwise-equivalent to fallback (proved via dev/check_fused_align_equiv.py for
        both bf16/FP8 shape 128/EP4 AND NVFP4/Kimi shape 384/EP8 -> dtype-independent ->
        FP8/NVFP4 unaffected). Profile (graph-off): moe_align_block_size_small_batch
        384->0 launches. Results: runs/.../opt1/

opt2 - topk+pack single launch  [flag-only, no code change]
  What: _pack_topk_for_flashinfer_routed (a cast/<<16/| elementwise chain after gating)
        is fused INTO the gating kernel (fused_topk packed_out -> StandardTopKOutputPacked),
        so the dispatch reuses packed_topk_ids and skips the separate pack. Already wired;
        experimental_sgl_trtllm meets the gate conditions (128 experts pow2, softmax, no
        correction-bias, no shared-experts).
  Flag: SGLANG_OPT_LORA_FUSED_TOPK_PACK (DEFAULT TRUE).
  Result: decode +5.6/+3.6/+3.1%. Profile: BinaryFunctor 576->0, bitwise 12->0, copy_
        1309->157, total launches 24178->21874. Results: runs/.../opt2/

opt3 - drop elem/upcast + lean _get_lora_info  [NO CLEAR WIN]
  What: _get_lora_info rebuilds layer-static scalars every layer-forward; lean it
        (cache num_experts/max_lora_rank/hidden_size). Plus the activation-vec flag.
  Change: layers.py _get_lora_info caches scalars; flag SGLANG_OPT_LORA_LEAN_INFO
        (default True). Commit: 1536c6e4e (pushed).
  Result: NO CLEAR WIN. opt2 already removed the elem/copy bulk; _get_lora_info is
        CPU/capture-time (decode is cuda-graph, sees it only at capture); residual is in
        the decomposed .cu op (=> the fold). Measured prefill within noise, decode ~0.
        Results: runs/.../opt3/  (honest negative result documented)

common opt-in flags audit (NVFP4 intersect bf16):
  - SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 + SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1
    added to dev model.env LORA_ENVS. Measured: no clear win (prefill within ~6% noise;
    decode flat). Kept (both bitwise-equivalent/safe).
  - NVFP4-only / Kimi-only flags that are NO-OPS for bf16 Qwen-128 (code path not reached):
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION (NVFP4 quant scale),
    SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION (deepseek_v2 / NVFP4),
    SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE (topk.py gated to num_experts==384).
  Conclusion: ALL common, bf16-applicable opts are now enabled (mostly default-on).
    There is no cheap common headroom left.

================================================================================
4. CURRENT NUMBERS - base vs LoRA (two-stream decode), bs16/32/64
--------------------------------------------------------------------------------
(reproducible; see runs/.../current_base_lora/ and the 3-way run)
                  prefill tok/s   decode tok/s        e2e s
  base(no-LoRA)   ~186k-194k      3933/6920/11714     8.5/9.8/11.9
  LoRA before     ~34k            2335/4252/7190      15.0/17.3/22.0
  LoRA after      ~33-37k         2574/4686/8088      13.6/15.8/19.8

  LoRA-after vs no-LoRA: decode 66/68/69% . prefill ~18-19% . e2e ~1.6x
  LoRA-after vs before (opt effect): decode +11/+10.5/+12.7% . e2e -9~10% . prefill noise

  IMPORTANT METRIC NOTE: decode tok/s is reliable (run-to-run variance ~0.06%).
  prefill tok/s is NOISY (~6-14% across runs) because the bench does one short prefill
  pass vs 2048 decode steps. Judge perf by DECODE (and e2e); treat prefill deltas <~10%
  as noise. (Prefill is also stream-independent, so a real prefill win must appear in
  BOTH the single and two-stream columns.)

================================================================================
5. WHAT'S COMMON vs NVFP4-ONLY vs BF16-ONLY  (so you don't chase no-ops)
--------------------------------------------------------------------------------
COMMON (dtype-agnostic, bf16 already has all): opt1 fused-align, opt2 topk-pack,
  shrink-splitK, qkv-b-store, overlaps (main-alloc/shared-add), cublas, lean-info,
  two-stream (O1 gate_up + O7/O8/O9 attn), MOE_ALIGN, ACTIVATION_VEC. Also FP4 launcher
  tricks bf16 already adopted: permute-buffer reuse, skip padded-row memset.
NVFP4-ONLY (NOT portable to bf16 - quant): fused permute+quant, fused activation+quant
  (activated_bf16 never materialized), per-token activation scale.
BF16-ONLY (the remaining work): the in-MoE fold via CUTLASS-EVT GEMM1 epilogue.

================================================================================
6. INFRASTRUCTURE - how to run on GB300
--------------------------------------------------------------------------------
- Cluster: kubectl context gcp-radixark-02 (GB300). NEVER leira (gone).
- WARM POD currently up & reusable: sglang-gb300-qwen330-yushengsu-bf16test-20260607
  on node 6zvh (ID=bf16test-20260607). Has the model weights + warm JIT cache + the
  current sglang code (HEAD 1536c6e4e). The dev state file dev/.state/<model>.env points
  to it. Reuse it (it's idle) instead of launching a fresh node.
  ** Cluster note (2026-06-10): the two FREE GB300 nodes (24wq, 5wsb) were disk-broken
     (24wq: /var/lib/cni full; 5wsb: low ephemeral-storage, evicted the pod). A fresh
     launch (dev/1_launch_node.sh) may fail to schedule. Reusing the warm pod sidesteps it. **
- dev/ harness (each step self-verifies; reads dev/.state/<model>.env):
    1_launch_node.sh / 2_upload_code.sh (uploads committed HEAD; Python-only change = no
      recompile) / 3_run_benchmark.sh (no-lora vs lora, prefill/decode/e2e) /
    4_run_acc.sh / 5_run_profile.sh (graph on+off) / 6_upload_results.sh / 8_save_jit_cache.sh
- Drivers written this session (in dev/, all committed to tune-lora-perf):
    check_fused_align_equiv.py  - routing bitwise-equiv micro-check (in-pod)
    bench_opt1_ab.sh / bench_opt2_ab.sh / profile_opt1_ab.sh / profile_opt2_ab.sh / profile_opt1_graphon.sh
    bench_profile_matrix.sh     - GENERALIZED A/B: <model> <out-subdir> "<off-delta>" "<on-delta>",
                                  runs {off,on}x{single,two} bench(graph-on)+profile(graph-off).
    bench_3way.sh               - base vs lora-before vs lora-after, prefill/decode/e2e.
    profile_base_lora.sh / reprofile_bs16_allranks.sh - base+lora graph-on profiling.
- model.env: dev/models/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared/model.env
  LORA_ENVS now has all common opts on; max-lora-rank 32; BENCH_BS="16 32 64" in=out=2048.

================================================================================
7. NEXT TASK - the in-MoE fold (recommended)
--------------------------------------------------------------------------------
Why: it's the only substantive headroom left (~13us/layer post-opt1/opt2 =
  fused_moe 7.2 + activation 3.2 + permute 2.4, all fold-only) AND it's where the
  prefill gap lives (LoRA prefill is 5.7x no-LoRA). The decomposed bf16 pipeline
  materializes a standalone permute + activated_bf16 HBM round-trip + per-token LoRA GEMMs.

How (ref_bf16_opt.md sections 5/6 - V1): fuse SwiGLU+LoRA into the GEMM1 epilogue so the
  gate_up tile stays on-chip and only `activated` is written (gate_up never to HBM,
  activation kernel gone). NOT a port of sgl_fp8_moe.py (bf16 hit the missing-cubin
  wall like NVFP4) and NOT NVFP4's quant-fusion (bf16 has no quant). Two routes:
    (a) trtllm-gen emits an unfused-act gated GEMM1 cubin - risks FP8/FP4 cubins, and
        may be a "missing cubin" (needs the probe below).
    (b) CUTLASS grouped GEMM + EVT epilogue - additive, bf16-only, isolated (PREFERRED).
        Costs: re-prep gemm1_weights to a CUTLASS layout at load; perf risk vs the tuned
        bmm_Bfloat16 cubin; integrate routing metadata (cta_idx_xy_to_batch_idx,
        total_num_padded_tokens, per-expert dynamic counts) into CUTLASS Grouped GEMM.
  Interleaved gate/up pairing is NOT a blocker (whole pairs fit one CTA's N-tile, ref 6).

First step (ref section 8 open item): determine whether the bf16 fusedAct=false GATED
  GEMM1 cubin is "not generated" vs "not registered" - write a bf16 analogue of
  sgl_trtllm_fp4_probe_unfused (launcher.cu ~L4047) using Bfloat16/Bfloat16 + Swiglu +
  unfuseActForLora=true; getValidConfigIndices()>0 => route (a) is just wiring, -1 =>
  must do route (b). Build/run on GB300.

================================================================================
8. GOTCHAS & LESSONS (will save you hours)
--------------------------------------------------------------------------------
- FLAGS DEFAULT TRUE: SGLANG_OPT_LORA_FUSED_MERGED_ALIGN / FUSED_TOPK_PACK / LEAN_INFO
  default True. For an A/B baseline you MUST set the flag =0 explicitly - an unset var
  = feature ON = no real baseline. (First opt1 bench was a non-measurement because of this.)
- CUDA-GRAPH PROFILE DOESN'T SHOW THE FUSION: under cuda-graph, torch --profile traces
  for flag-off vs flag-on come out byte-identical (the fused routing kernels aren't
  exposed). Use the EAGER (graph-OFF) trace to SHOW which kernel is removed; use the
  cuda-graph BENCH (no --profile) for the TIMING win. They're consistent; the profiler
  just can't visualize it under graph.
- bench_one_batch_server returns nonzero on its client shutdown (benign "leaked
  shared_memory") -> with `set -o pipefail` it aborts the driver after a VALID cell.
  The matrix/3way drivers now tolerate it (warn + validate jsonl). Don't re-add `|| exit`.
- BIG-FILE UPLOAD: GitHub contents API (gh api -X PUT) handles files up to ~40MB base64
  via --input; ~50MB+ FAILS. For those, git push via a partial sparse clone:
  `git clone --depth 1 --filter=blob:none --sparse <repo>; git sparse-checkout set <dir>;
   cp files; git add; git push` (push handles the big blobs; the partial clone avoids
   pulling the repo's existing GB of traces).
- macOS is case-insensitive: SUMMARY.md == summary.md (bit me once).
- zsh: a shell function or `| while` pipe can lose PATH (gh/base64 "command not found").
  Inline commands in the main shell instead of wrapping in a function.
- Build/bench/profile are GB300-only; the laptop has no CUDA.

================================================================================
9. WHERE EVERYTHING IS (quick index)
--------------------------------------------------------------------------------
Uploaded to results repo runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913/:
  LAYER_ATTN_MOE_BREAKDOWN.md   - the optimization map; rec#1 + MoE-decomp-extra row + url
                                  cell updated with opt1/opt2/opt3 links and the corrected
                                  in-MoE-fold framing (bf16=NVFP4 sibling, CUTLASS-EVT).
  opt1/  opt2/  opt3/           - each: OPT*.md, summary.md, *_matrix.png + before/after png,
                                  profile/ traces. (opt1/opt2 = wins; opt3 = no win.)
  current_base_lora/            - base vs lora bench (bs16/32/64) + profiling bs16 TP0-3
                                  (base = cuda-graph; lora = cuda-graph + two-stream).
sglang fork branch qwen3-30b-a3b-2507-bf16: commits 869882a3a (opt1), 1536c6e4e (opt3).
tune-lora-perf (harness) main: all dev/ drivers + model.env flags committed/pushed.

Memory file (claude project memory, has the running log):
  ~/.claude/projects/-Users-yushengsu-Downloads-tml-tune-lora-perf/memory/opt1-goal-align-sort-fusion.md
================================================================================
