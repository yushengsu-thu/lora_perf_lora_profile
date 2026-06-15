# profile traces — in the GitHub Release (not committed to the repo)
The `*.trace.json.gz` that profiling wrote in this directory were uploaded to this run's
GitHub Release to keep the repo small; this directory is preserved as a pointer to them.

- release: <https://github.com/yushengsu-thu/lora_perf_lora_profile/releases/tag/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260614-193948>
- download all traces for this run: `gh release download Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared-20260614-193948 -R yushengsu-thu/lora_perf_lora_profile -D ./traces`
- assets originally written in `lora/graph_off/` (release asset name  <-  original filename):
  - `lora__graph_off__bs16-TP-0.trace.json.gz`  <-  `bs16-TP-0.trace.json.gz`
  - `lora__graph_off__perfetto-compatible-bs16-TP-0.trace.json.gz`  <-  `perfetto-compatible-bs16-TP-0.trace.json.gz`
  - `lora__graph_off__bs16-TP-0.gpuonly.trace.json.gz`  <-  `bs16-TP-0.gpuonly.trace.json.gz`
