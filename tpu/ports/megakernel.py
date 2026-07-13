"""Persistent / megakernel decode.

CUDA: persistent_layer_decode.cu, persistent_layer_v3.cu, megakernel_decode.cu

These are launch-side optimisations on GPU (one big kernel per decode step
to avoid launch overhead). On TPU the equivalent is a single `jax.jit`
scope holding the full per-layer decode graph; XLA will tile, fuse, and
emit a single HLO module per layer.

The port expresses the decode layer as a straight-line composition of:
    add_norm_qkv_gemv -> RoPE + cache write -> paged_attention ->
    add_norm_gateup_gemv -> fused_silu_down -> residual
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds
from .attention import paged_attention
from .gemm import (
    fused_add_norm_qkv_gemv,
    fused_oproj_add_norm_gateup_gemv,
    gemm_bf16,
)
from .rope import rotary_embedding
from .kv import reshape_and_cache


def decode_layer(
    x,
    residual,
    rms_pre,
    w_qkv,
    w_o,
    rms_post,
    w_gateup,
    w_down,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    eps: float = 1e-6,
):
    qkv, new_residual = fused_add_norm_qkv_gemv(x, residual, rms_pre, w_qkv, eps)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = qkv[:, :q_size].reshape(-1, num_heads, head_dim)
    k = qkv[:, q_size:q_size + kv_size].reshape(-1, num_kv_heads, head_dim)
    v = qkv[:, q_size + kv_size:].reshape(-1, num_kv_heads, head_dim)

    q, k = rotary_embedding(q, k, cos_cache, sin_cache, positions)
    k_cache, v_cache = reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)

    # decode-only attention path expects q: [num_seqs, H, D]
    attn_out = paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)
    attn_out = attn_out.reshape(-1, num_heads * head_dim)

    # fused_oproj_add_norm_gateup_gemv already applies SiLU-mul, so its
    # output is the post-activation tensor at `intermediate` width.
    mlp_act, new_residual = fused_oproj_add_norm_gateup_gemv(
        attn_out, w_o, new_residual, rms_post, w_gateup, eps
    )
    # Return the MLP branch separately. The next layer's fused add-norm adds
    # it to new_residual exactly once.
    mlp_out = gemm_bf16(mlp_act, w_down)
    return mlp_out, new_residual, k_cache, v_cache


def decode_layer_trace_spec(shapes, dtype="bf16"):
    s = shapes["num_seqs"]
    h = shapes["hidden"]
    nh, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    i = shapes["intermediate"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    mb = shapes.get("max_ctx_blocks", 256)
    mp = shapes.get("max_pos", 4096)
    return (
        sds((s, h), dtype),                      # x
        sds((s, h), dtype),                      # residual
        sds((h,), dtype),                        # rms_pre
        sds((h, (nh + 2 * kh) * d), dtype),      # w_qkv
        sds((nh * d, h), dtype),                 # w_o
        sds((h,), dtype),                        # rms_post
        sds((h, 2 * i), dtype),                  # w_gateup
        sds((i, h), dtype),                      # w_down
        sds((mp, d // 2), "f32"),                # cos
        sds((mp, d // 2), "f32"),                # sin
        sds((s,), "i32"),                        # positions
        sds((s,), "i32"),                        # slot_mapping
        sds((nb, bs, kh, d), dtype),             # k_cache
        sds((nb, bs, kh, d), dtype),             # v_cache
        sds((s, mb), "i32"),                     # block_tables
        sds((s,), "i32"),                        # context_lens
    ), {
        "num_heads": nh,
        "num_kv_heads": kh,
        "head_dim": d,
        "scale": 1.0 / (d ** 0.5),
    }
