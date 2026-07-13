# Prompt-lookup speculative decoding

rvLLM implements an opt-in clean-room n-gram drafter for greedy generation.
The drafter searches the committed token stream for a matching suffix, proposes
at most `RVLLM_SPEC_K` tokens, verifies them in one target-model forward, and
accepts only the longest matching prefix plus the target bonus token.

Correctness invariant: output must be byte-for-byte identical to ordinary
greedy decode. Rejected positions may leave physical KV bytes, but logical
length and future addressing must expose only committed positions. Drafting is
disabled for sampled requests and falls back to a single-token step when no
match exists.

Tests must cover no match, repeated suffixes, overlapping matches, budget
limits, EOS/stop boundaries, ring/block wrap, graph and eager execution, and a
long real-model sequence compared with greedy decode. Throughput is not claimed
without a released-SHA receipt.
