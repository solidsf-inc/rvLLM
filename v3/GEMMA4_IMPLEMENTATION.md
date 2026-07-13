# Gemma implementation status

The `v3` workspace contains Gemma configuration parsing, safetensors loading,
CUDA bring-up, layer execution, generation, serving, benchmarking, optional
vision modules, and an Apple Metal host path.

Current public status:

- CUDA text execution is the primary integrated path.
- Prompt-lookup n-gram speculative decoding is opt-in for greedy requests.
- Vision is feature-gated; publishable support still needs a real-weight API
  parity receipt.
- Metal kernels and host code compile on Apple Silicon; end-to-end serving is
  not release-gated yet.
- Missing artifacts and unsupported shapes are expected to fail closed.

The source and tests are authoritative. This document intentionally carries no
private checkpoint paths, deployment state, or performance claims.
