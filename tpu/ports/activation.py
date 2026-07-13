"""Elementwise activations.

CUDA: activation.cu, activation_f16.cu, silu_mul_interleaved.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds


def silu(x):
    return jax.nn.silu(x)


def silu_trace_spec(shapes, dtype="bf16"):
    return (sds((shapes["n"],), dtype),), {}


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def gelu_trace_spec(shapes, dtype="bf16"):
    return (sds((shapes["n"],), dtype),), {}


def fused_silu_mul(gate_up):
    # gate_up layout matches CUDA silu_mul_interleaved: concatenated [gate, up]
    # along the last axis.
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return jax.nn.silu(gate) * up


def fused_silu_mul_trace_spec(shapes, dtype="bf16"):
    t, i = shapes["num_tokens"], shapes["intermediate"]
    return (sds((t, 2 * i), dtype),), {}
