# FP8 GEMM debugging

Reproduce with one immutable model revision, one source SHA, fixed inputs, and
the smallest failing matrix shape. Record dtype, layout, leading dimensions,
scales, epilogue, workspace, CUDA/cuBLASLt/CUTLASS versions, and architecture.

Correctness gates:

1. Compare decoded inputs and output against an FP32 reference.
2. Test zero, signed zero, subnormal, saturation, NaN, and rounding vectors.
3. Run eager and graph-replay paths with identical metadata.
4. Check every size/stride multiplication and ABI field before launch.
5. Fail if a requested FP8 path silently selects a different kernel or dtype.

`scripts/test_fp8_encode.py` checks canonical E4M3FN bytes. Model-scale analysis
is available in `fp8_precision_check.py` and requires an explicit local model.
Do not publish checkpoint-derived tensors.
