# opt7 — the bf16 in-MoE fold (route b, CUTLASS) — kernels ✅ all gates / e2e ✗ host-bound

**Commits** (PR #4 branch): P0 `1a82c2111`+`7cea5ed86` · P1 `850faa…7e1e69eb6` · P2 `f2247b5a8`
· P3 `f7e5d5119` · P4 `22feb0e73`+`022d547e2`+fixups `0f60bec38`/`f3e89dc28`/`dfa3493c9`.
**Flags**: `SGLANG_OPT_BF16_MOE_DUAL_LAYOUT` + `SGLANG_OPT_BF16_MOE_GEMM1_FOLD`
(**default OFF** — see verdict). Design: `../opt7_design/OPT7_DESIGN.md`.

## Kernel-level results (all gates passed)

| phase | gate | result |
|---|---|---|
| step0 probe | route decision | unfused bf16 cubin absent at every tile/layout → route (b) |
| P0 ref kernel | semantics | pinned (interleaved cols / half-contiguous Δ / silu) |
| P1 grouped GEMM | ≤68.4µs (57µs cubin +20%) | **65.6µs** ✅ (cuBLAS 44.8µs = headroom) |
| P2 fold epilogue | bitwise + ≤99µs | **bitwise (0.0) + 84.8µs** ✅ |
| P3 gather | replace permute 180µs | **12.7µs (−93%)** ✅ — the 180µs was a decode-shaped grid (11% occupancy), not bandwidth |
| **pipeline** | ≤189µs | **101.5µs vs 270µs replaced (−62% MoE prefill kernel time)** ✅ |

## e2e (P4 integrated, real server)

- acc flags-off: PASS (KL 0.004807 ≈ floor) — fold code inert by default.
- **acc fold-ON: PASS (KL 0.004132 < floor 0.004243)** — full pipeline numerically correct:
  load-time W_fold capture (interleave perm) → routing cta-map segments → gather →
  CUTLASS fold GEMM (Δ+SwiGLU epilogue) → opt6 row-map → down-shrink.
- **bench matrix: prefill +1.2~2.6% (within noise), decode flat.**

| stream | bs16 prefill | decode |
|---|---|---|
| single | 102.6% | 99.3% |
| two | 101.2% | 99.0% |

## Why the kernel win doesn't reach e2e (the structural finding)

The fold removes ~168µs/layer of GPU kernel time (~64ms per prefill ≈ 7% of wall), but
prefill is **~50% GPU-idle / host-bound** (15.9k eager launches; journal §4). The removed
kernel time is absorbed by gaps where the GPU was already waiting on the host; the fold
only removes 2–3 launches/layer of ~40+. **Per-layer kernel fusion has hit the host-bound
wall** — the next real prefill lever is HOST-side: piecewise CUDA graph for prefill /
launch batching, after which this fold's −62% kernel time becomes immediately valuable.

## Verdict

- Flags **default OFF** (e2e-neutral today; zero risk — off = byte-identical).
- Code stays on PR #4: correct (at-floor acc), gated, isolated (FP8/NVFP4 untouched —
  one scripted-edit leak into the FP4 launcher was caught at compile time and reverted
  verbatim, `0f60bec38`), and becomes the payoff once prefill is host-unbound.

## e2e debug ladder (cost: 3 acc rounds)

1. Scripted replace leaked the fold block into FP4BlockScaleLoraLauncher (same alloc
   pattern in both launchers) — compile error, restored verbatim. Lesson: `grep -c` for
   pattern uniqueness before scripted edits on launcher.cu.
2. Illegal memory access #1: group segments derived from `num_tokens_per_expert` under an
   unverified semantics assumption → rebuilt from the authoritative
   `cta_idx_xy_to_batch_idx` map (same data the Gemm2 runner consumes).
3. Illegal memory access #2: `permuted_idx_to_expanded_idx` is UNINITIALIZED at pad rows
   (not −1) — garbage indices into the delta. Bound-check with num_expanded.
   Lesson: synthetic-map unit tests can't catch real-routing edge contracts; e2e acc can.
