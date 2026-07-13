# Architecture

`v3/Cargo.toml` is the authoritative workspace manifest. The serving path is:

1. `rvllm-serve` parses the local HTTP API and submits bounded work.
2. `rvllm-runtime` builds plans and executes model steps.
3. `rvllm-loader`, `rvllm-metadata`, and `rvllm-mem` validate model data,
   metadata, and device storage.
4. `rvllm-attention`, `rvllm-fused`, `rvllm-cutlass`, `rvllm-kernels`, and
   `rvllm-sampling` dispatch accelerator work.
5. `rvllm-graph` owns graph-capture state and replay contracts.

`rvllm-core` contains shared model types. `rvllm-bench` and
`rvllm-invariants` provide validation surfaces. `rvllm-metal` contains Apple
Metal kernels; `rvllm-vision` and `rvllm-imageio` contain the optional Gemma
vision path. `rvllm-mcp` is a separate adapter.

CUDA is the primary integrated runtime. Metal code builds on Apple Silicon but
is not presented as an end-to-end serving release until real-weight parity and
API smoke gates pass. Vision is feature-gated and carries the same limitation.

The server binds loopback only. It implements health/status/metrics, model
listing, text completions, and chat completions. Remote exposure requires an
audited proxy with TLS and authentication.
