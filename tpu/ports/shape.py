"""Layout/reshape ops.

CUDA: qkv_transpose.cu, deinterleave_qkv.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds


def qkv_transpose(x):
    # [num_tokens, num_heads, head_dim] -> [num_heads, num_tokens, head_dim]
    return jnp.transpose(x, (1, 0, 2))


def qkv_transpose_trace_spec(shapes, dtype="bf16"):
    t, h, d = shapes["num_tokens"], shapes["num_heads"], shapes["head_dim"]
    return (sds((t, h, d), dtype),), {}


def deinterleave_qkv(qkv, num_heads: int, num_kv_heads: int, head_dim: int):
    # Input layout: [num_tokens, (num_heads + 2*num_kv_heads) * head_dim] where
    # Q/K/V are concatenated along the last axis.
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = jax.lax.dynamic_slice_in_dim(qkv, 0, q_size, axis=-1)
    k = jax.lax.dynamic_slice_in_dim(qkv, q_size, kv_size, axis=-1)
    v = jax.lax.dynamic_slice_in_dim(qkv, q_size + kv_size, kv_size, axis=-1)
    q = q.reshape(qkv.shape[0], num_heads, head_dim)
    k = k.reshape(qkv.shape[0], num_kv_heads, head_dim)
    v = v.reshape(qkv.shape[0], num_kv_heads, head_dim)
    return q, k, v


def deinterleave_qkv_trace_spec(shapes, dtype="bf16"):
    t = shapes["num_tokens"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    return (sds((t, (h + 2 * kh) * d), dtype),), {
        "num_heads": h,
        "num_kv_heads": kh,
        "head_dim": d,
    }
