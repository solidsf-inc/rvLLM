# Layer execution

A layer step validates dimensions and then performs normalization, projections,
Q/K normalization where configured, RoPE/KV update, attention, output
projection, feed-forward activation/projections, and residual updates. Gemma
sliding/global attention and scale/soft-cap parameters come from validated
configuration.

CUDA dispatch is explicit among verified kernel/library variants. GB10 and
Metal use separate feature/platform routes. Metal lacks some FP8 and sliding
attention combinations. Unsupported routes return typed errors; a fallback is
permitted only when it is implemented, parity tested, and surfaced in the run
receipt.

Each layer requires reference and eager/replay parity across supported shapes.
