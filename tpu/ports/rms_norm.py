"""RMSNorm family ports.

CUDA source: ../kernels/rms_norm.cu, rms_norm_f16.cu,
             fused_residual_rmsnorm.cu, fused_residual_rmsnorm_f16.cu,
             fused_rmsnorm_fp8_quant.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds


def rms_norm(x, weight, eps: float = 1e-6):
    # x: [num_tokens, hidden], weight: [hidden]
    ss = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    scale = jax.lax.rsqrt(ss + eps)
    y = x.astype(jnp.float32) * scale
    return (y * weight.astype(jnp.float32)).astype(x.dtype)


def rms_norm_trace_spec(shapes, dtype="bf16"):
    t, h = shapes["num_tokens"], shapes["hidden"]
    return (sds((t, h), dtype), sds((h,), dtype)), {}


def fused_residual_rms_norm(x, residual, weight, eps: float = 1e-6):
    # Adds residual first, then RMSNorm; returns (normed, new_residual).
    new_residual = (x + residual).astype(x.dtype)
    normed = rms_norm(new_residual, weight, eps)
    return normed, new_residual


def fused_residual_rms_norm_trace_spec(shapes, dtype="bf16"):
    t, h = shapes["num_tokens"], shapes["hidden"]
    return (sds((t, h), dtype), sds((t, h), dtype), sds((h,), dtype)), {}


def fused_rmsnorm_fp8_quant(x, weight, inv_scale, eps: float = 1e-6):
    # TODO: StableHLO has no native FP8 E4M3 quant path; route through
    # stablehlo.custom_call once jax.numpy.float8_e4m3fn is first-class on TPU.
    raise NotImplementedError(
        "fused_rmsnorm_fp8_quant: FP8 quant requires custom_call; unblock once "
        "TPU v5p float8_e4m3fn StableHLO op is available."
    )


def fused_rmsnorm_fp8_quant_trace_spec(shapes, dtype="bf16"):
    t, h = shapes["num_tokens"], shapes["hidden"]
    return (sds((t, h), dtype), sds((h,), dtype), sds((), "f32")), {}
