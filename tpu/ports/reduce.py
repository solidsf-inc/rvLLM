"""Reductions: argmax and fused lm_head argmax.

CUDA: argmax.cu, argmax_f16.cu, fused_lm_head_argmax.cu,
      fused_lm_head_argmax_f16.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def argmax(x):
    return jnp.argmax(x, axis=-1).astype(jnp.int32)


def argmax_trace_spec(shapes, dtype="bf16"):
    r, c = shapes["rows"], shapes["cols"]
    return (sds((r, c), dtype),), {}


def fused_lm_head_argmax(hidden, lm_head_weight):
    # hidden: [T, hidden], lm_head_weight: [vocab, hidden]
    logits = jnp.einsum("th,vh->tv", hidden.astype(jnp.float32),
                        lm_head_weight.astype(jnp.float32))
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def fused_lm_head_argmax_trace_spec(shapes, dtype="bf16"):
    t, h, v = shapes["num_tokens"], shapes["hidden"], shapes["vocab"]
    return (sds((t, h), dtype), sds((v, h), dtype)), {}
