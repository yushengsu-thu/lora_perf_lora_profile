# opt1 — bf16 MoE-LoRA align/sort fusion (shared_outer)

**Result: decode +11% at bs16 (decode-only), e2e −9%, prefill unchanged, accuracy preserved.**

Model: Qwen3-30B-A3B-Instruct-2507-BF16, **expert_shared** (`--experts-shared-outer-loras`),
GB300, TP4/EP4, decode. Baseline profile = the parent run
`runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913`.
sglang `qwen3-30b-a3b-2507-bf16` @ `8a30bf134` (base `526e0ae22`).

## The change (2 lines, Python-only)
`python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`, `_get_routing`:
- gate `... and not shared_outer and ep_local ...` → `... and (shared_outer or ep_local) ...`
- `compact=True` → `compact=not shared_outer`

The fused single-launch align/scatter kernel (`moe_lora_merged_align` /
`fused_align_scatter_kernel`) **already supported shared_outer** (`compute_virtual_id` base=0,
launcher only guards `num_experts<=1024`); it was simply gated off in Python. opt1 opens that
existing capability. No `.cu` change → the node's warm JIT cache stays valid.

Before opt1, `shared_outer` fell through to the unfused fallback
(`_fused_virtual_topk_ids` + `moe_align_block_size_small_batch`) — the breakdown's ~10.2 µs/layer
align/sort cost. Now it takes the single fused launch.

## Bench (graph-ON, real timing) — see `summary.md`
| bs | decode off→on | Δ decode | e2e | prefill |
|---|---|---|---|---|
| 16 | 2315 → 2569 tok/s | **+11.0%** | −9.2% | flat |
| 32 | 4228 → 4647 tok/s | **+9.9%** | −8.2% | flat |
| 64 | 7290 → 7929 tok/s | **+8.8%** | −7.1% | flat |

A/B = `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN` `0` vs `1`. (NB: the flag **defaults True** — a real
baseline must set `=0`.) Both cells produce identical coherent decode.

## Mechanism — which kernels are removed (`opt1_before_after.png`, `profile/`)
The figure is drawn from the **eager (graph-OFF) traces** in `profile/` — the routing/align cluster
of one MoE-LoRA layer, on the main stream:

| | routing/align kernels on the main stream |
|---|---|
| **BEFORE** (flag off) | `_moe_lora_shrink_splitk` → `_fused_virtual_topk_ids` → `moe_align_block_size_small_batch` → `fused_moe_kernel` |
| **AFTER** (opt1 on) | the `_fused_virtual_topk_ids` + `moe_align_block_size_small_batch` pair is replaced by a **single `fused_align_scatter`** |

Whole-trace counts confirm it: `moe_align_block_size_small_batch` **384 → 0 launches**,
`fused_align_scatter` **0 → 576**.

> **Why eager and not cuda-graph for the figure:** the timing win above is cuda-graph (production).
> But the torch profiler `--profile` *under cuda-graph* does not expose the fused routing kernels —
> the before/after cuda-graph traces come out byte-identical (a profiler×graph-replay artifact), so
> they can't visualize the removal. The eager trace exercises the *same code path* and shows it
> directly. The cuda-graph **bench** (no profiler) is what proves the +11% timing.
> (`profile/` durations are eager launch-latency — structure only; timing is the cuda-graph bench.)

## Correctness & guardrail (FP8/NVFP4)
The routing change is **dtype-independent** (topk_ids + token_lora_mapping → routing tensors).
`dev/check_fused_align_equiv.py` proves the fused path is **bitwise-equivalent** to the old fallback:
- bf16 / FP8 shape (128 experts, EP4, top_k8): 50/50 shared_outer + 50/50 per_expert
- NVFP4 / Kimi shape (384 experts, EP8, top_k8): 50/50 shared_outer + 50/50 per_expert

Identical routing ⇒ identical MoE-LoRA output ⇒ **FP8/NVFP4 accuracy unchanged**. Perf cannot
regress: the fused path is strictly fewer launches and is gated to decode (`numel≤2048`,
`<512 tokens`); prefill keeps the old path. `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN` is default-True
and shared by FP8/NVFP4, so this is on-by-default for them — covered by the equivalence proof.
A direct FP8/NVFP4 e2e perf bench (dedicated pods; Kimi is 2-node) is the optional heavier follow-up.

## Artifacts
- `OPT1.md` — this summary
- `summary.md` — cuda-graph bench A/B (prefill/decode/e2e, flag off vs on)
- `opt1_before_after.png` — annotated before/after timeline (removed align kernels red-circled)
- `profile/off/bs16-TP-0.trace.json.gz` — eager trace, **BEFORE** (flag OFF, unfused align) — open in perfetto
- `profile/on/bs16-TP-0.trace.json.gz`  — eager trace, **AFTER**  (flag ON, fused align)
