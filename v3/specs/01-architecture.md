# Workspace architecture

`v3/Cargo.toml` contains 18 workspace crates. Shared types flow from `rvllm-core` into
loading/memory, kernel backends, metadata, runtime, and interfaces. The runtime
may depend on backend crates; low-level crates must not depend on serving.

```text
core
├─ loader ─ mem
├─ kernels ─ cutlass ─ attention ─ fused ─ graph ─ sampling
├─ metadata
└─ runtime
   ├─ serve / mcp
   └─ bench / invariants

optional: metal; vision + imageio
```

CUDA, Metal, serving, vision, and validation remain
separate functional surfaces. Their presence in the DAG is not a support claim.
Only tests on the relevant platform establish a supported route.
