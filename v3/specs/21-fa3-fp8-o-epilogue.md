# FA3 FP8 output epilogue

Status: design only. The released path must not assume an FP8 attention output
epilogue exists unless the loaded shared library exports a versioned, validated
ABI and parity tests cover it.

The proposed epilogue reduces each output row to an absolute maximum, derives a
finite nonzero E4M3 scale, rounds with reviewed E4M3FN semantics, writes the
quantized row and scale, and preserves the existing higher-precision output
option. NaN, infinity, zero rows, tails, and head dimensions 128/256/512 require
explicit tests.

Before implementation, record the exact FlashAttention upstream commit,
license/NOTICE, source files, local patch, compiler flags, symbols, and artifact
hash. No latency or bandwidth expectation is a release claim.
