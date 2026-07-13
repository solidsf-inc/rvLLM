import unittest

import jax.numpy as jnp

from ports.attention import flash_attention, paged_attention


class AttentionReferenceTest(unittest.TestCase):
    def test_gqa_divisibility_is_checked(self):
        q = jnp.zeros((1, 1, 3, 2))
        kv = jnp.zeros((1, 1, 2, 2))
        with self.assertRaises(ValueError):
            flash_attention(q, kv, kv, 1.0)

    def test_scale_changes_reference_distribution(self):
        q = jnp.array([[[[1.0]]]])
        k = jnp.array([[[[0.0]], [[2.0]]]])
        v = jnp.array([[[[0.0]], [[1.0]]]])
        low = flash_attention(q, k, v, 0.1, causal=False)
        high = flash_attention(q, k, v, 2.0, causal=False)
        self.assertLess(float(low[0, 0, 0, 0]), float(high[0, 0, 0, 0]))

    def test_paged_attention_masks_padding(self):
        q = jnp.ones((1, 2, 1))
        k = jnp.array([[[[1.0]], [[100.0]]]])
        v = jnp.array([[[[3.0]], [[99.0]]]])
        out = paged_attention(
            q,
            k,
            v,
            jnp.array([[0]], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            1.0,
        )
        self.assertTrue(jnp.allclose(out, 3.0))

    def test_invalid_block_is_visible(self):
        q = jnp.ones((1, 1, 1))
        cache = jnp.ones((1, 1, 1, 1))
        out = paged_attention(
            q,
            cache,
            cache,
            jnp.array([[2]], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            1.0,
        )
        self.assertTrue(jnp.isnan(out).all())
