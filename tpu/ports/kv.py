"""KV-cache ops.

CUDA: copy_blocks.cu, copy_blocks_f16.cu, reshape_and_cache.cu,
      reshape_and_cache_f16.cu, fp8_kv.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def copy_blocks(k_cache, v_cache, src_to_dst):
    # src_to_dst: [num_pairs, 2] int32 — (src_block, dst_block)
    src = src_to_dst[:, 0]
    dst = src_to_dst[:, 1]
    k_updates = k_cache[src]
    v_updates = v_cache[src]
    k_cache = k_cache.at[dst].set(k_updates)
    v_cache = v_cache.at[dst].set(v_updates)
    return k_cache, v_cache


def copy_blocks_trace_spec(shapes, dtype="bf16"):
    nb, bs, kh, d = (
        shapes["num_blocks"],
        shapes["block_size"],
        shapes["num_kv_heads"],
        shapes["head_dim"],
    )
    np_ = shapes["num_pairs"]
    return (
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((np_, 2), "i32"),
    ), {}


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping):
    # key/value: [num_tokens, num_kv_heads, head_dim]
    # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # slot_mapping: [num_tokens] flat slot index
    k_flat = k_cache.reshape(-1, *k_cache.shape[2:])
    v_flat = v_cache.reshape(-1, *v_cache.shape[2:])
    k_flat = k_flat.at[slot_mapping].set(key.astype(k_cache.dtype))
    v_flat = v_flat.at[slot_mapping].set(value.astype(v_cache.dtype))
    return k_flat.reshape(k_cache.shape), v_flat.reshape(v_cache.shape)


def reshape_and_cache_trace_spec(shapes, dtype="bf16"):
    t = shapes["num_tokens"]
    kh, d = shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    return (
        sds((t, kh, d), dtype),
        sds((t, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((t,), "i32"),
    ), {}


def fp8_kv(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale):
    # TODO: FP8 E4M3 cache quantisation. StableHLO lacks a native quant op;
    # unblock via stablehlo.custom_call once float8_e4m3fn lands on TPU.
    raise NotImplementedError(
        "fp8_kv: TPU FP8 KV-cache path not expressible in StableHLO yet."
    )


def fp8_kv_trace_spec(shapes, dtype="bf16"):
    t = shapes["num_tokens"]
    kh, d = shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    return (
        sds((t, kh, d), dtype),
        sds((t, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((t,), "i32"),
        sds((), "f32"),
        sds((), "f32"),
    ), {}
