"""Shared helpers for ports.

Every port exposes:
  - a pure function `fn(*tensors, **static) -> pytree`
  - a callable `trace_spec(shapes: dict) -> (args, kwargs)` that builds
    `jax.ShapeDtypeStruct` inputs matching the manifest's shape template

The emitter imports `fn` via the manifest's "port" field and calls
`trace_spec` to produce tracing inputs, then lowers to StableHLO.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


DTYPE_MAP = {
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16,
    "i32": jnp.int32,
    "i64": jnp.int64,
}


def sds(shape, dtype):
    """jax.ShapeDtypeStruct shortcut."""
    if isinstance(dtype, str):
        dtype = DTYPE_MAP[dtype]
    return jax.ShapeDtypeStruct(tuple(shape), dtype)
