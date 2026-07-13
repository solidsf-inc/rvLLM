"""RoPE (rotary embedding) family.

CUDA: rotary_embedding.cu, rotary_embedding_f16.cu, fused_rope_cache.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds


def _apply_rope(x, cos, sin):
    # x: [..., head_dim]; cos/sin: [..., head_dim/2]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    out = jnp.stack([r0, r1], axis=-1)
    return out.reshape(x.shape)


def rotary_embedding(query, key, cos_cache, sin_cache, positions):
    # query: [num_tokens, num_heads, head_dim]
    # key:   [num_tokens, num_kv_heads, head_dim]
    # cos_cache/sin_cache: [max_pos, head_dim/2]
    # positions: [num_tokens] int32
    cos = cos_cache[positions][:, None, :]  # [T,1,hd/2]
    sin = sin_cache[positions][:, None, :]
    q = _apply_rope(query, cos.astype(query.dtype), sin.astype(query.dtype))
    k = _apply_rope(key, cos.astype(key.dtype), sin.astype(key.dtype))
    return q, k


def rotary_embedding_trace_spec(shapes, dtype="bf16"):
    t = shapes["num_tokens"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    mp = shapes["max_pos"]
    return (
        sds((t, h, d), dtype),
        sds((t, kh, d), dtype),
        sds((mp, d // 2), "f32"),
        sds((mp, d // 2), "f32"),
        sds((t,), "i32"),
    ), {}


def fused_rope_cache(
    query,
    key,
    value,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
):
    # RoPE + reshape_and_cache in one. Returns (rotated_q, updated_k_cache,
    # updated_v_cache). block-size embedded in cache shape.
    q, k = rotary_embedding(query, key, cos_cache, sin_cache, positions)

    # cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    block_idx = slot_mapping // block_size
    slot_idx = slot_mapping % block_size

    # scatter K/V into [block_idx, slot_idx, :, :]
    k_flat = k_cache.reshape(num_blocks * block_size, *k_cache.shape[2:])
    v_flat = v_cache.reshape(num_blocks * block_size, *v_cache.shape[2:])
    k_flat = k_flat.at[slot_mapping].set(k.astype(k_cache.dtype))
    v_flat = v_flat.at[slot_mapping].set(value.astype(v_cache.dtype))
    return q, k_flat.reshape(k_cache.shape), v_flat.reshape(v_cache.shape)


def fused_rope_cache_trace_spec(shapes, dtype="bf16"):
    t = shapes["num_tokens"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    mp = shapes["max_pos"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    return (
        sds((t, h, d), dtype),
        sds((t, kh, d), dtype),
        sds((t, kh, d), dtype),
        sds((mp, d // 2), "f32"),
        sds((mp, d // 2), "f32"),
        sds((t,), "i32"),
        sds((t,), "i32"),
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
    ), {}
