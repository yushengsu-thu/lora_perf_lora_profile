# opt7-step0 — bf16 unfused-cubin probe — **route (b) CONFIRMED**

**Commit**: [`d12fe74a7`](https://github.com/yushengsu-thu/sglang/commit/d12fe74a7) (probe fn
`sgl_trtllm_bf16_probe_unfused`, launcher.cu; driver `dev/probe_bf16_unfused.py`).
Run on GB300 (warm pod), shapes H=2048, I=768/rank, top_k=8, 32 local experts (EP4).

## Result (valid cubin configs; −1 = Runner ctor throws / family absent)

| tokens | tile | [0] Swiglu fused BMK | [1] Swiglu UNFUSED BMK | [2] Swiglu UNFUSED MK | [3] Identity |
|---|---|---|---|---|---|
| 16 | 8 | 144 | −1 | −1 | −1 |
| 16 | 16 | 144 | −1 | −1 | −1 |
| 16 | 32 | 64 | −1 | −1 | −1 |
| 16 | 64 | 4 | −1 | −1 | −1 |
| 16 | 128 | 4 | −1 | −1 | −1 |
| 4096 | 8–128 | 144/144/64/4/4 | −1 | −1 | −1 |

- [0] sanity passes (the normal no-LoRA bf16 fused path has plentiful configs → probe works).
- [1] the route-(a) question: **no unfused-activation gated GEMM1 cubin exists for bf16**, at
  any tile, in bf16's BlockMajorK layout.
- [2] not in FP4's MajorK layout either; [3] no plain raw-output (Identity) GEMM1 variant.

## Verdict
**Route (a) is dead — bf16 hits the same missing-unfused-cubin wall NVFP4 did** (FP4 probe:
`known dead (-1)`). The in-MoE fold (opt7) must be **route (b)**: a new bf16-only CUTLASS
grouped GEMM with gather-prologue (= fused permute) + SwiGLU·LoRA EVT epilogue, de-risked by
prefill-only dispatch + dual-layout gemm1 weights (+9.7 GB/rank; decode keeps the tuned
`bmm_Bfloat16` cubin). opt6's permuted-read + exported-row-map plumbing feeds this pipeline.
