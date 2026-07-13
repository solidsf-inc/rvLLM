"""Activation quantisation to FP8.

CUDA: quantize_activation_fp8.cu, fused_silu_fp8_quant.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def quantize_activation_fp8(x, inv_scale):
    # TODO: JAX has jnp.float8_e4m3fn on CPU/GPU but XLA:TPU quant path still
    # goes through custom_call. Leave as TODO so the emitter refuses to lower
    # a silent FP32 stand-in.
    raise NotImplementedError(
        "quantize_activation_fp8: requires TPU-side f8e4m3fn StableHLO op."
    )


def quantize_activation_fp8_trace_spec(shapes, dtype="bf16"):
    t, h = shapes["num_tokens"], shapes["hidden"]
    return (sds((t, h), dtype), sds((), "f32")), {}


def fused_silu_fp8_quant(gate_up, inv_scale):
    # TODO: SiLU-mul then FP8 quant in one pass. Same block as above.
    raise NotImplementedError(
        "fused_silu_fp8_quant: FP8 quant not expressible in StableHLO yet."
    )


def fused_silu_fp8_quant_trace_spec(shapes, dtype="bf16"):
    t, i = shapes["num_tokens"], shapes["intermediate"]
    return (sds((t, 2 * i), dtype), sds((), "f32")), {}
