# Speculative decoding status

The released source contains prompt-lookup n-gram drafting and target-model
verification for greedy generation. It does not require a second model and does
not alter output semantics. `RVLLM_SPEC_DECODE=1` enables it; the draft budget
is bounded by `RVLLM_SPEC_K`.

Sampling requests do not use this path. No challenge-derived multi-token
prediction implementation is part of this specification or claimed by the
public release.

Release gates are greedy identity over long sequences, EOS/stop correctness,
page/ring-wrap safety after rejected drafts, eager/graph parity, bounded draft
and workspace sizes, and receipt-bound performance. Unit coverage for the
n-gram matcher alone is necessary but not sufficient.
