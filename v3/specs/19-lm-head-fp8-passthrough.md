# LM-head FP8 passthrough

The optimization question is whether the final normalized activation can stay
in FP8 through the LM-head path without an intervening dequantize/requantize.
This is valid only when the producer scale layout and consumer GEMM contract are
identical and the final logits/argmax remain within the declared parity bound.

The current source must be inspected at both call sites before enabling a
passthrough; no historical line reference or task note is authoritative. Test
zero/extreme scales, vocabulary tails, greedy ties, sampled logits, and
eager/replay. Do not claim bandwidth or latency benefit without a released-SHA
receipt.
