"""Dtype cast.

CUDA: cast_fp.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds, DTYPE_MAP


def cast_fp(x, out_dtype: str = "bf16"):
    return x.astype(DTYPE_MAP[out_dtype])


def cast_fp_trace_spec(shapes, dtype="f32"):
    # Emit the f32->bf16 path by default.
    return (sds((shapes["n"],), dtype),), {"out_dtype": "bf16"}
