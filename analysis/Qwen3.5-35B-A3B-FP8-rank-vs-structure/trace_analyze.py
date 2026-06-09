#!/usr/bin/env python3
# Analyze a kineto GPU trace: kernel GPU-time by class, two-stream usage, LoRA overlap,
# and dump a steady-state decode-step window for plotting.
import gzip, json, sys, collections, re

def cls(n):
    if "_lora" in n: return "lora"
    if "all_reduce" in n or "allreduce" in n or "nccl" in n.lower(): return "allreduce/TP"
    if any(k in n for k in ("chunk_gated_delta","causal_conv1d","chunk_fwd","recompute_w_u","fused_qkvzba","gated_delta")): return "gdn/mamba"
    if any(k in n for k in ("moe::dev","bmm_","count_and_sort_expert","moe_align_block","topkGatingSoftmax","finalizeKernel","activationDeepSeek")): return "moe-base"
    if any(k in n for k in ("deep_gemm","fp8_fp4_gemm","nvjet","per_token_group_quant","cutlass")): return "gemm/quant"
    if any(k in n for k in ("mha","ttention","flash","decode_mla")): return "attention"
    if "rmsnorm" in n.lower() or "RMSNorm" in n: return "rmsnorm"
    return "other/elementwise"

def main(path, tag=None):
    with gzip.open(path) as f: ev=json.load(f)["traceEvents"]
    K=[e for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel" and "dur" in e]
    streams=collections.Counter(e["tid"] for e in K)
    main_tid=streams.most_common(1)[0][0]
    # by class (all streams)
    bycls=collections.defaultdict(float); bycls_main=collections.defaultdict(float)
    lora_k=collections.defaultdict(float)
    for e in K:
        c=cls(e["name"]); bycls[c]+=e["dur"]
        if e["tid"]==main_tid: bycls_main[c]+=e["dur"]
        if c=="lora": lora_k[e["name"].split("(")[0][:40]]+=e["dur"]
    tot=sum(bycls.values())
    print(f"\n=== {path.split('/')[-1]} ===")
    print(f"streams (tid->kernels): {dict(streams)}  main={main_tid}")
    print(f"total kernel GPU-time: {tot/1000:.1f} ms")
    print("by class (ms, %% of total):")
    for c,d in sorted(bycls.items(),key=lambda x:-x[1]):
        print(f"  {c:18} {d/1000:8.1f} ms  {100*d/tot:5.1f}%")
    print(f"LoRA kernels breakdown (ms):")
    for n,d in sorted(lora_k.items(),key=lambda x:-x[1]):
        print(f"    {d/1000:7.1f}  {n}")
    # non-allreduce compute (allreduce is spin-wait inflated)
    nar=tot-bycls.get("allreduce/TP",0)
    print(f"non-allreduce compute = {nar/1000:.1f} ms ; LoRA = {bycls['lora']/1000:.1f} ms = {100*bycls['lora']/nar:.1f}% of it")
    if tag:
        out={"bycls":{k:v/1000 for k,v in bycls.items()},
             "lora_k":{k:v/1000 for k,v in lora_k.items()},
             "tot":tot/1000,"nonallreduce":nar/1000,
             "streams":{str(k):v for k,v in streams.items()},"main_tid":main_tid}
        json.dump(out, open(f"/tmp/sum_{tag}.json","w"), indent=1)
        print(f"  wrote /tmp/sum_{tag}.json")
    return bycls, lora_k, tot

if __name__=="__main__":
    main(sys.argv[1], *([sys.argv[2]] if len(sys.argv)>2 else []))
