"""GEMM / GEMV family.

CUDA: cutlass_gemm.cu, cutlass_hgemm_autotune.cu, persistent_gemm.cu,
      gemv_f16.cu, tc_gemv_decode.cu, tma_gemv_fp16.cu, wgmma_gemv.cu,
      cutlass_fp8_gemm.cu (+ autotune + residual), gemv_fp8.cu,
      gemv_int4.cu, cutlass_qkv_bias.cu, cutlass_gateup_silu.cu (+ autotune),
      cutlass_oproj_residual.cu (+ autotune), fused_silu_down.cu,
      fused_silu_down_gemv.cu, fused_norm_gemv.cu,
      fused_add_norm_qkv_gemv.cu, fused_add_norm_gateup_gemv.cu,
      fused_oproj_add_norm_gateup_gemv.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds
from .rms_norm import rms_norm
from .activation import fused_silu_mul


# -------- dense GEMM --------

def gemm_bf16(x, w):
    # x: [M, K], w: [K, N]  ->  [M, N]
    return jnp.einsum("mk,kn->mn", x, w)


def gemm_bf16_trace_spec(shapes, dtype="bf16"):
    m, n, k = shapes["m"], shapes["n"], shapes["k"]
    return (sds((m, k), dtype), sds((k, n), dtype)), {}


def gemv_bf16(x, w):
    # x: [1, K], w: [K, N]
    return jnp.einsum("mk,kn->mn", x, w)


def gemv_bf16_trace_spec(shapes, dtype="bf16"):
    m, n, k = shapes["m"], shapes["n"], shapes["k"]
    return (sds((m, k), dtype), sds((k, n), dtype)), {}


def gemv_fp8(x, w, x_scale, w_scale):
    # TODO: FP8 scaled dot. Lowers once stablehlo scaled-dot / TPU FP8 lands.
    raise NotImplementedError(
        "gemv_fp8: requires StableHLO scaled-dot op for E4M3 tensors."
    )


def gemv_fp8_trace_spec(shapes, dtype="bf16"):
    m, n, k = shapes["m"], shapes["n"], shapes["k"]
    return (
        sds((m, k), dtype),
        sds((k, n), dtype),
        sds((), "f32"),
        sds((), "f32"),
    ), {}


def gemm_fp8_scaled(x, w, x_scale, w_scale, bias=None, residual=None):
    # TODO: FP8 matmul w/ per-tensor (or per-block) scale and optional
    # residual fuse. Same blocker as above.
    raise NotImplementedError(
        "gemm_fp8_scaled: FP8 GEMM not lowerable to StableHLO today."
    )


def gemm_fp8_scaled_trace_spec(shapes, dtype="bf16"):
    m, n, k = shapes["m"], shapes["n"], shapes["k"]
    return (
        sds((m, k), dtype),
        sds((k, n), dtype),
        sds((), "f32"),
        sds((), "f32"),
    ), {}


# -------- INT4 weight-only --------

def gemv_int4(x, w_packed, scales, zeros, group_size: int):
    # w_packed: [K/8, N] uint32 (8 x int4 per word)
    # scales/zeros: [K/group_size, N]. `zeros` contains unsigned zero-points
    # in [0, 15]; symmetric signed-int4 uses zero-point 8.
    k_pack, n = w_packed.shape
    k = k_pack * 8
    if group_size <= 0 or k % group_size:
        raise ValueError("group_size must be positive and divide K")
    expected = (k // group_size, n)
    if scales.shape != expected or zeros.shape != expected:
        raise ValueError(f"scales and zeros must have shape {expected}")

    # unpack 8 nibbles per uint32 into int4 lanes. bitcast i32 -> u32 so the
    # logical right-shift does not sign-extend the high nibble.
    packed_u32 = jax.lax.bitcast_convert_type(w_packed, jnp.uint32)
    shifts = jnp.arange(8, dtype=jnp.uint32) * jnp.uint32(4)
    w_u4 = (packed_u32[..., None] >> shifts) & jnp.uint32(0xF)
    w_u4 = w_u4.transpose(0, 2, 1).reshape(k, n).astype(jnp.int32)

    # per-group dequant
    groups = k // group_size
    w_u4 = w_u4.reshape(groups, group_size, n)
    w_deq = (w_u4 - zeros[:, None, :]) * scales[:, None, :]
    w_deq = w_deq.reshape(k, n).astype(x.dtype)
    return jnp.einsum("mk,kn->mn", x, w_deq)


def gemv_int4_trace_spec(shapes, dtype="bf16"):
    m, n, k, g = shapes["m"], shapes["n"], shapes["k"], shapes["group_size"]
    return (
        sds((m, k), dtype),
        sds((k // 8, n), "i32"),  # packed weights as i32 (u32 not in DTYPE_MAP)
        sds((k // g, n), dtype),
        sds((k // g, n), dtype),
    ), {"group_size": g}


# -------- fused variants (component-level; XLA will re-fuse on TPU) --------

def gemm_qkv_bias(x, w, bias):
    return gemm_bf16(x, w) + bias


def gemm_qkv_bias_trace_spec(shapes, dtype="bf16"):
    m, h, q = shapes["m"], shapes["hidden"], shapes["qkv_out"]
    return (sds((m, h), dtype), sds((h, q), dtype), sds((q,), dtype)), {}


def gemm_gateup_silu(x, w_gateup):
    # w_gateup: [hidden, 2*intermediate]
    y = gemm_bf16(x, w_gateup)
    return fused_silu_mul(y)


def gemm_gateup_silu_trace_spec(shapes, dtype="bf16"):
    m, h, i = shapes["m"], shapes["hidden"], shapes["intermediate"]
    return (sds((m, h), dtype), sds((h, 2 * i), dtype)), {}


def gemm_oproj_residual(x, w_o, residual):
    return gemm_bf16(x, w_o) + residual


def gemm_oproj_residual_trace_spec(shapes, dtype="bf16"):
    m, h = shapes["m"], shapes["hidden"]
    return (sds((m, h), dtype), sds((h, h), dtype), sds((m, h), dtype)), {}


def fused_silu_down(gate_up, w_down):
    y = fused_silu_mul(gate_up)
    return gemm_bf16(y, w_down)


def fused_silu_down_trace_spec(shapes, dtype="bf16"):
    m, h, i = shapes["m"], shapes["hidden"], shapes["intermediate"]
    return (sds((m, 2 * i), dtype), sds((i, h), dtype)), {}


def fused_norm_gemv(x, rms_weight, w, eps: float = 1e-6):
    return gemm_bf16(rms_norm(x, rms_weight, eps), w)


def fused_norm_gemv_trace_spec(shapes, dtype="bf16"):
    m, h, o = shapes["m"], shapes["hidden"], shapes["out"]
    return (sds((m, h), dtype), sds((h,), dtype), sds((h, o), dtype)), {}


def fused_add_norm_qkv_gemv(x, residual, rms_weight, w_qkv, eps: float = 1e-6):
    new_residual = x + residual
    normed = rms_norm(new_residual, rms_weight, eps)
    return gemm_bf16(normed, w_qkv), new_residual


def fused_add_norm_qkv_gemv_trace_spec(shapes, dtype="bf16"):
    m, h, q = shapes["m"], shapes["hidden"], shapes["qkv_out"]
    return (
        sds((m, h), dtype),
        sds((m, h), dtype),
        sds((h,), dtype),
        sds((h, q), dtype),
    ), {}


def fused_add_norm_gateup_gemv(x, residual, rms_weight, w_gateup, eps: float = 1e-6):
    new_residual = x + residual
    normed = rms_norm(new_residual, rms_weight, eps)
    return gemm_gateup_silu(normed, w_gateup), new_residual


def fused_add_norm_gateup_gemv_trace_spec(shapes, dtype="bf16"):
    m, h, i = shapes["m"], shapes["hidden"], shapes["intermediate"]
    return (
        sds((m, h), dtype),
        sds((m, h), dtype),
        sds((h,), dtype),
        sds((h, 2 * i), dtype),
    ), {}


def fused_oproj_add_norm_gateup_gemv(
    attn_out, w_o, residual, rms_weight, w_gateup, eps: float = 1e-6
):
    proj = gemm_bf16(attn_out, w_o)
    new_residual = proj + residual
    normed = rms_norm(new_residual, rms_weight, eps)
    return gemm_gateup_silu(normed, w_gateup), new_residual


def fused_oproj_add_norm_gateup_gemv_trace_spec(shapes, dtype="bf16"):
    m, h, i = shapes["m"], shapes["hidden"], shapes["intermediate"]
    return (
        sds((m, h), dtype),
        sds((h, h), dtype),
        sds((m, h), dtype),
        sds((h,), dtype),
        sds((h, 2 * i), dtype),
    ), {}
