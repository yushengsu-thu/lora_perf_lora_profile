# opt5 — prefill routing reuse (unify A/B stage routing block size) — ✅ WIN

**Commit**: [`850faa87f`](https://github.com/yushengsu-thu/sglang/commit/850faa87f) (PR #4 branch
`qwen3-30b-a3b-2507-bf16`) · **Flag**: `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE` (default **True**;
set `=0` for the A/B baseline) · dtype-agnostic (fp8/nvfp4/bf16 share the chain).

## What
At prefill the Triton LoRA-Δ chain re-sorted routing per stage: the A (shrink) stage routes
with `BLOCK_SIZE_M` 32, the B (expand) stage with the tuned fused-moe config (64), so the
per-layer `routing_cache` key `(num_experts, shared_outer, block_size)` never matched across
stages → `moe_align_block_size` + `count_and_sort_expert_tokens` ran **4×/layer**
(~119 µs/layer, ~46 ms per full prefill). One line in `virtual_experts.py` matches the A
stage's routing block to the B stage's at prefill (≥512 tokens) → the cache hits.

**4× → 2×/layer** (not 1×: the remaining two sorts are *genuinely different* — the
shared-outer A stage routes by lora id with num_experts=1, the per-expert B stage by real
expert id over 128. Collapsing those two needs the trtllm-routing-metadata integration —
opt7's pipeline.) Decode (<512 tokens) is untouched: it keeps the opt1 fused merged-align.

## Bench (graph-ON, bs16/32/64, in=out=2048; `summary.md` for full table)

| stream | bs | prefill | decode | e2e |
|---|---|---|---|---|
| single | 16 | **+8.2%** | 100.0% | −0.4% |
| single | 32 | **+7.9%** | 100.0% | −0.7% |
| single | 64 | **+7.4%** | 99.9% | −1.1% |
| two | 16 | **+9.4%** | 99.6% | −0.2% |
| two | 32 | **+10.3%** | 99.6% | −0.7% |
| two | 64 | **+11.1%** | 100.2% | −2.0% |

Prefill win appears in BOTH stream columns at all three batch sizes; measured noise floor
(opt4's identical-cell single column) is ±0–2%. Decode is flat — the gate works.

## Profile proof (graph-OFF bs16, old code vs flag=1; `profile/{single,two}_on/`)

| kernel (per profile window) | old (opt4 trace) | opt5 | Δ |
|---|---|---|---|
| `moe_align_block_size_kernel` | 1536 (13.8 ms) | 768 (6.8 ms) | **−50%** |
| `count_and_sort_expert_tokens` | 1536 (32.0 ms) | 768 (16.0 ms) | **−50%** |
| `_fused_virtual_topk_ids` | 1536 (2.8 ms) | 768 (1.4 ms) | **−50%** |
| total kernel launches | 21,682 | 18,994 | **−2,688** |
| `_moe_lora_shrink_splitk` | 1152 (22.9 ms) | 1152 (24.8 ms) | +1.9 ms (tile 32→64, honest cost) |

Net: align/sort cluster −24.3 ms per window; the small shrink regression is far outweighed
(bench net +8~11%). The launch-count cut also attacks the ~50% host-bound prefill wall.

## Correctness
- acc (teacher-forced prefill logprobs, flag ON): KL vs trainer 0.005407 ≈ vLLM noise floor
  0.004243 — exercises exactly the changed prefill path. Coherence checks passed per cell.
- The align layout change (BLOCK_SIZE_M 32→64 for the A stage) is consumed consistently:
  the same `a_stage_config` feeds both the align and the shrink kernel launch.

## Files
- `summary.md` — full matrix table.
- `bench/{off,on}_{single,two}/bs{16,32,64}.jsonl` — raw bench.
- `profile/{single,two}_on/bs16-TP-0.trace.json.gz` — graph-off traces (flag=1).
  Old-code baseline traces: `../opt4/profile/{single,two}_on/`.
