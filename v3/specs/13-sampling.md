# Sampling

Greedy decoding selects the maximum logit deterministically. Sampled decoding
applies validated temperature, top-p, top-k, and seed values. The current GPU
candidate pool is bounded at 1024 entries; larger top-k values therefore do not
mean an unbounded full-vocabulary selection.

No zero-filled or placeholder candidate buffer may be consumed as a successful
sample. If GPU selection is unavailable, the request must use a tested explicit
path or fail. RNG state belongs to the request and advances exactly once per
committed token; speculative drafts do not consume sampling RNG.

Tests cover greedy ties, NaN/Inf policy, top-k/p boundaries, seed replay,
distribution sanity, cancellation, vocabulary tails, and host/device parity.
Sampled serving remains release-blocked wherever the active path cannot satisfy
these gates.
