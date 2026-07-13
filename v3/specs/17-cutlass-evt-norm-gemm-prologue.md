# CUTLASS EVT normalization prologue

Status: unimplemented optimization design.

The proposal fuses input normalization and activation quantization into a GEMM
prologue while preserving the existing GEMM/epilogue result. Implementation
must target the repository's exact CUTLASS gitlink and use only APIs verified in
that revision. Descriptor, schedule, alignment, scale, and workspace contracts
must be explicit at the C ABI boundary.

Parity is required for every supported M/N/K, tail, dtype, scale, and graph
route before dispatch can change. Performance expectations are hypotheses until
an immutable hardware receipt is published.
