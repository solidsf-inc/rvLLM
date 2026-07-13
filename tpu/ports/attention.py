"""Attention family.

CUDA: flash_attention.cu, flash_attention_3.cu (+ prefill, v3, sm90 wrapper),
      paged_attention.cu, split_kv_attention.cu
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from ._base import sds


def _gqa_expand(k, num_heads: int):
    # k: [*, num_kv_heads, head_dim] -> [*, num_heads, head_dim] via group repeat
    num_kv_heads = k.shape[-2]
    if num_kv_heads <= 0 or num_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError("num_heads must be positive and divisible by num_kv_heads")
    reps = num_heads // num_kv_heads
    return jnp.repeat(k, reps, axis=-2)


def _flash_reference(q, k, v, scale: float, causal: bool):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 BSHD tensors")
    if k.shape != v.shape:
        raise ValueError("k and v shapes must match")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k batch/head dimensions must match")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    num_heads = q.shape[-2]
    k_exp = _gqa_expand(k, num_heads)
    v_exp = _gqa_expand(v, num_heads)

    scores = jnp.einsum("bshd,bthd->bhst", q, k_exp) * scale  # [B,H,Sq,Sk]
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask = jnp.tril(jnp.ones((sq, sk), dtype=jnp.bool_), k=sk - sq)
        scores = jnp.where(mask, scores, -jnp.inf)
    else:
        mask = jnp.ones(scores.shape[-2:], dtype=jnp.bool_)

    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.where(mask, jnp.exp((scores - m).astype(jnp.float32)), 0.0)
    denominator = jnp.sum(p, axis=-1, keepdims=True)
    p = jnp.where(denominator > 0, p / denominator, 0.0)
    p = p.astype(q.dtype)
    return jnp.einsum("bhst,bthd->bshd", p, v_exp)


def flash_attention(q, k, v, scale: float, causal: bool = True):
    # q: [B, Sq, Hq, D]
    # k,v: [B, Sk, Hkv, D]
    # The StableHLO port is deliberately device-independent. A fused Pallas
    # implementation must be a separate, parity-gated entry rather than a
    # trace-time backend branch.
    return _flash_reference(q, k, v, scale, causal)


def flash_attention_trace_spec(shapes, dtype="bf16"):
    b = shapes["batch"]
    sq, sk = shapes["seq_q"], shapes["seq_k"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    return (
        sds((b, sq, h, d), dtype),
        sds((b, sk, kh, d), dtype),
        sds((b, sk, kh, d), dtype),
    ), {"scale": 1.0 / (d ** 0.5), "causal": True}


def paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale: float):
    # q: [num_seqs, num_heads, head_dim]
    # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # block_tables: [num_seqs, max_ctx_blocks] int32
    # context_lens: [num_seqs] int32 (actual token count per seq)
    if q.ndim != 3:
        raise ValueError("q must have shape [num_seqs, num_heads, head_dim]")
    num_seqs, num_heads, head_dim = q.shape
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have matching rank-4 shapes")
    num_blocks, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if num_blocks <= 0 or block_size <= 0:
        raise ValueError("cache must contain at least one non-empty block")
    if cache_head_dim != head_dim:
        raise ValueError("query and cache head dimensions must match")
    if block_tables.ndim != 2 or block_tables.shape[0] != num_seqs:
        raise ValueError("block_tables must have shape [num_seqs, max_blocks]")
    if context_lens.shape != (num_seqs,):
        raise ValueError("context_lens must have shape [num_seqs]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    max_ctx_blocks = block_tables.shape[1]
    max_ctx = max_ctx_blocks * block_size

    # gather per-sequence K/V: [num_seqs, max_ctx, num_kv_heads, head_dim]
    valid_blocks = (block_tables >= 0) & (block_tables < num_blocks)
    safe_blocks = jnp.clip(block_tables, 0, max(num_blocks - 1, 0))
    k_gathered = k_cache[safe_blocks].reshape(num_seqs, max_ctx, num_kv_heads, head_dim)
    v_gathered = v_cache[safe_blocks].reshape(num_seqs, max_ctx, num_kv_heads, head_dim)

    k_exp = _gqa_expand(k_gathered, num_heads)
    v_exp = _gqa_expand(v_gathered, num_heads)

    # scores: [num_seqs, num_heads, max_ctx]
    scores = jnp.einsum("sHd,stHd->sHt", q, k_exp) * scale

    # mask out positions past context_len
    ar = jnp.arange(max_ctx)[None, :]  # [1, max_ctx]
    block_valid_tokens = jnp.repeat(valid_blocks, block_size, axis=1)
    valid_lengths = (context_lens >= 0) & (context_lens <= max_ctx)
    valid = (ar < context_lens[:, None]) & block_valid_tokens
    scores = jnp.where(valid[:, None, :], scores, -jnp.inf)

    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.where(valid[:, None, :], jnp.exp((scores - m).astype(jnp.float32)), 0.0)
    denominator = jnp.sum(p, axis=-1, keepdims=True)
    p = jnp.where(denominator > 0, p / denominator, 0.0)
    p = p.astype(q.dtype)
    out = jnp.einsum("sHt,stHd->sHd", p, v_exp)
    input_valid = valid_lengths & jnp.all(valid_blocks | ~(
        jnp.arange(max_ctx_blocks)[None, :] * block_size < context_lens[:, None]
    ), axis=1)
    return jnp.where(input_valid[:, None, None], out, jnp.nan)


def paged_attention_trace_spec(shapes, dtype="bf16"):
    s = shapes["num_seqs"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    mb = shapes["max_ctx_blocks"]
    return (
        sds((s, h, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((s, mb), "i32"),
        sds((s,), "i32"),
    ), {"scale": 1.0 / (d ** 0.5)}


def split_kv_attention(
    q, k_cache, v_cache, block_tables, context_lens, split_table, scale: float
):
    # Split-KV is a GPU launch-tiling optimisation; the math is identical to
    # paged_attention. XLA:TPU retiles for the mesh, so the StableHLO port
    # collapses back to plain paged_attention. `split_table` is unused but
    # kept in the signature for binder parity with the CUDA call site.
    del split_table
    return paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)


def split_kv_attention_trace_spec(shapes, dtype="bf16"):
    s = shapes["num_seqs"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    ns = shapes["num_splits"]
    # reuse paged_attention layout for the first 5 args; add split_table last
    (q, k, v, bt, cl), _ = paged_attention_trace_spec(
        {**shapes, "max_ctx_blocks": shapes.get("max_ctx_blocks", 256)}, dtype
    )
    return (q, k, v, bt, cl, sds((s, ns), "i32")), {"scale": 1.0 / (d ** 0.5)}
