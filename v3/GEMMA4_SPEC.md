# Gemma architecture contract

rvLLM derives dimensions and behavior from a local model's validated
`config.json`; model weights and model-specific licenses are not distributed by
this repository.

The implementation supports the fields required by its Gemma text path,
including layer count, hidden/intermediate sizes, query and KV head counts,
head dimension, RoPE parameters, sliding/global attention, normalization,
vocabulary size, and soft-capping where present. Unknown or inconsistent
dimensions must fail during configuration or loading.

Weights are read from safetensors through an index constrained to the model
root. Tensor shape, dtype, byte length, and scale layout are validated before
device upload. Compiled kernels are local artifacts verified by a manifest;
there is no personal or mutable download fallback.

Architecture support is established by real-weight logits/perplexity parity,
not by successful compilation alone.
