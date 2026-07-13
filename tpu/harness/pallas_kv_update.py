"""Checked JAX reference for a future fused TPU KV-cache update."""

import jax
import jax.numpy as jnp


def kv_cache_update(k, v, kc, vc, pos):
    """Return caches with one K/V row updated.

    Invalid dynamic positions produce NaN caches so a traced caller cannot
    silently write a clamped row. Cache dtypes must therefore be floating.
    """
    if kc.ndim != 2 or vc.shape != kc.shape:
        raise ValueError("kc and vc must have matching rank-2 shapes")
    if k.shape != (kc.shape[1],) or v.shape != k.shape:
        raise ValueError("k and v must match the cache row width")
    if not jnp.issubdtype(kc.dtype, jnp.floating) or not jnp.issubdtype(
        vc.dtype, jnp.floating
    ):
        raise TypeError("KV caches must use floating dtypes")

    position = jnp.asarray(pos, dtype=jnp.int32)
    valid = (position >= 0) & (position < kc.shape[0])
    safe_position = jnp.clip(position, 0, kc.shape[0] - 1)
    kc_updated = kc.at[safe_position].set(k.astype(kc.dtype))
    vc_updated = vc.at[safe_position].set(v.astype(vc.dtype))
    return jax.lax.cond(
        valid,
        lambda: (kc_updated, vc_updated),
        lambda: (jnp.full_like(kc, jnp.nan), jnp.full_like(vc, jnp.nan)),
    )


def fused_kv_cache_update(*_args, **_kwargs):
    """Fail closed until an alias-safe Pallas implementation passes parity."""
    raise NotImplementedError(
        "fused Pallas KV update is not validated; use kv_cache_update"
    )


if __name__ == "__main__":
    key_cache = jnp.zeros((4, 3), dtype=jnp.bfloat16)
    value_cache = jnp.zeros_like(key_cache)
    key = jnp.array([1, 2, 3], dtype=jnp.bfloat16)
    value = jnp.array([4, 5, 6], dtype=jnp.bfloat16)
    key_cache, value_cache = kv_cache_update(
        key, value, key_cache, value_cache, 2
    )
    assert jnp.array_equal(key_cache[2], key)
    assert jnp.array_equal(value_cache[2], value)
    assert jnp.count_nonzero(key_cache.at[2].set(0)) == 0
