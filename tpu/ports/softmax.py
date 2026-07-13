"""Softmax family.

CUDA: softmax.cu, softmax_f16.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def softmax(x):
    # numerically stable row-wise softmax over last axis
    m = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp((x - m).astype(jnp.float32))
    return (e / jnp.sum(e, axis=-1, keepdims=True)).astype(x.dtype)


def softmax_trace_spec(shapes, dtype="bf16"):
    r, c = shapes["rows"], shapes["cols"]
    return (sds((r, c), dtype),), {}
