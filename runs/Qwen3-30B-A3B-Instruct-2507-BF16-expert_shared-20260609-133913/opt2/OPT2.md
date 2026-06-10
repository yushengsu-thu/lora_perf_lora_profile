# opt2 — bf16 MoE-LoRA topk+pack single launch

**Result: decode +5.6% at bs16, e2e −5%, prefill unchanged, accuracy preserved. Stacks on opt1.**

Model: Qwen3-30B-A3B-Instruct-2507-BF16, **expert_shared**, GB300, TP4/EP4, decode.
Parent run: `runs/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260609-133913`.

## What opt2 is — no code change, already wired
Before each trtllm MoE-LoRA dispatch the routed top-k must be packed into FlashInfer's int32
format by `_pack_topk_for_flashinfer_routed` — a chain of elementwise ops:
`topk_ids.to(int32)`, `topk_weights.to(bf16)`, `(ids << 16) | weights.view(int16).to(int32)`
(several cast + shift + or launches). opt2 = **fuse that pack INTO the gating kernel**: when
`SGLANG_OPT_LORA_FUSED_TOPK_PACK` is on (default True), `fused_topk(packed_out=…)` emits
`StandardTopKOutputPacked.packed_topk_ids` directly, and the dispatch reuses it
(`getattr(topk_output, "packed_topk_ids")`) instead of running the separate pack.

This machinery already exists. `experimental_sgl_trtllm` reaches the fused-pack branch
(`topk.py` select_experts `else`, not the raw-logits `is_flashinfer_trtllm_routed` branch) and
meets its conditions (128 experts = power-of-2 ≤512, softmax, no correction-bias, no shared
experts, no EPLB) — so it is on by default. **opt2 is the validation + measurement of that path.**

## Bench (graph-ON, real timing) — `summary.md`
A/B = `SGLANG_OPT_LORA_FUSED_TOPK_PACK` `0` vs `1` (opt1's fused-align held on for both, isolating opt2).
| bs | decode off→on | Δ decode | e2e | prefill |
|---|---|---|---|---|
| 16 | 2422 → 2559 tok/s | **+5.6%** | −5.1% | flat |
| 32 | 4453 → 4612 tok/s | +3.6% | −2.8% | flat |
| 64 | 7690 → 7925 tok/s | +3.1% | −3.0% | flat |

Fixed-cost kernels → smaller batches benefit most. Both cells produce identical coherent decode.
(NB: the flag **defaults True** — a real baseline must set `=0`.)

## Mechanism (eager/graph-OFF, kernel structure) — `profile/`, `opt2_before_after.png`
| kernel category | off | on |
|---|---|---|
| `BinaryFunctor` (the `<<16 | ` pack op) | 576 | **0** |
| `bitwise` | 12 | **0** |
| `copy_` (casts) | 1309 | 157 |
| `vectorized_elementwise` | 3416 | 1688 |
| **total kernel launches (12-step window)** | 24178 | **21874** |

The pack's shift/or (`BinaryFunctor`+`bitwise`) is fully eliminated; casts drop sharply — the whole
`_pack_topk_for_flashinfer_routed` chain is folded into the gating kernel.

> Same profiler caveat as opt1: the cuda-graph `--profile` trace doesn't expose this difference;
> the figure/structure is from the eager trace, the +5.6% timing is the cuda-graph bench.

## Correctness
The fused pack computes the **same** `(id<<16)|weight` value inside the gating kernel; decode output
is identical (coherence: both cells emit the same text). Dtype-independent, so FP8/NVFP4 unaffected.

## Artifacts
- `OPT2.md` (this), `summary.md` (bench A/B), `opt2_before_after.png` (annotated before/after)
- `profile/off/bs16-TP-0.trace.json.gz` — eager BEFORE (pack off), `profile/on/…` — eager AFTER (fused)
