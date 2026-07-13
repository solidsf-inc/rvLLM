"""Bias-add family.

CUDA: add_bias.cu, add_bias_f16.cu, add_bias_broadcast.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def add_bias(x, bias):
    # bias broadcasts over leading axes
    return (x + bias).astype(x.dtype)


def add_bias_trace_spec(shapes, dtype="bf16"):
    r, c = shapes["rows"], shapes["cols"]
    return (sds((r, c), dtype), sds((c,), dtype)), {}
