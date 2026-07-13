# Current forward path

The CUDA path loads validated configuration and safetensors through
`rvllm-loader`, allocates device state through `rvllm-mem`, and constructs a
batch plan in `rvllm-runtime`. Each transformer layer applies normalization,
Q/K/V projection, RoPE and KV-cache update, attention, output projection,
feed-forward projection, activation, and residual updates. Final normalization,
LM-head projection, and sampling produce the next token.

Prefill and decode use distinct shapes but share model state and checked
metadata. Paged KV state, slot mappings, sequence lengths, and graph replay
inputs must be refreshed together; a mismatch is an error, not a fallback.

Backend dispatch is explicit. Missing CUDA libraries, kernels, policies, or
unsupported shapes fail closed. Metal components exist behind feature and
platform gates, but full Metal serving remains release-blocked on real-weight
parity and API validation.
