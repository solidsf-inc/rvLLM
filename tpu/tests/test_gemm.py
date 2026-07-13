import unittest

import jax.numpy as jnp

from ports.gemm import gemv_int4


class Int4DequantizationTest(unittest.TestCase):
    def test_unsigned_and_symmetric_zero_points(self):
        packed = jnp.array([[0x76543210]], dtype=jnp.uint32)
        x = jnp.ones((1, 8), dtype=jnp.float32)
        scales = jnp.ones((1, 1), dtype=jnp.float32)

        unsigned = gemv_int4(x, packed, scales, jnp.zeros((1, 1)), 8)
        symmetric = gemv_int4(x, packed, scales, jnp.full((1, 1), 8), 8)

        self.assertEqual(float(unsigned[0, 0]), 28.0)
        self.assertEqual(float(symmetric[0, 0]), -36.0)

    def test_group_shape_is_checked(self):
        with self.assertRaises(ValueError):
            gemv_int4(
                jnp.ones((1, 8)),
                jnp.zeros((1, 1), dtype=jnp.uint32),
                jnp.ones((1, 1)),
                jnp.zeros((1, 1)),
                3,
            )


if __name__ == "__main__":
    unittest.main()
