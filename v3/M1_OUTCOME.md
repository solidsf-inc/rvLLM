# M=1 kernel outcome

rvLLM contains specialized FP8 GEMV work for single-row decode shapes. The
code remains an optional dispatch path, not a general performance conclusion.

Any default-selection change must compare the specialized kernel with the
existing GEMM path on identical inputs and include:

- byte/ULP error against a higher-precision reference;
- E4M3FN edge-vector coverage;
- eager and graph-replay results;
- all relevant matrix shapes, not one favorable shape;
- hardware, driver/toolchain, model revision, source SHA, and raw receipt.

Prior unreceipted timing and roofline estimates are retracted.
