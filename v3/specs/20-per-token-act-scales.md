# Per-token activation scales

Per-token FP8 activation rows require one finite positive scale per row. The
GEMM route must either consume that exact vector through an attribute supported
by the pinned cuBLASLt/CUTLASS revision or apply an explicit, parity-tested
ratio correction. Attribute names and support are verified at build/runtime;
they are not inferred from newer documentation.

Scale buffers are shape-checked, device-resident for the launch lifetime, and
part of the graph fingerprint. Tests cover uniform versus per-token scales,
zero rows, extremes, tails, multiple M values, and higher-precision reference
parity. Any ratio-correction path must use canonical E4M3FN rounding and report
unsupported scale layouts rather than silently collapsing them.
