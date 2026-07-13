# Fused operations

The fused crate contains host references and accelerator launchers for selected
normalization, quantization, RoPE/cache, activation, residual, projection, and
sampling-adjacent operations. The source inventory, not historical tables, is
authoritative.

Launchers validate all dimensions, extents, alignment, aliasing rules, dtypes,
scales, workspace, and stream before dispatch. Non-CUDA builds return an
explicit unavailable error for CUDA-only work; they must never report success
with untouched output.

Every fusion is tested against the unfused reference, including tails, zero
length, extreme scales, NaN/Inf policy, overlapping buffers, and graph replay.
