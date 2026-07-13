"""Embedding gather.

CUDA: embedding_gather.cu, embedding_gather_f16.cu
"""

from __future__ import annotations

import jax.numpy as jnp

from ._base import sds


def embedding_gather(table, token_ids):
    # table: [vocab, hidden], token_ids: [num_tokens] int32
    return table[token_ids]


def embedding_gather_trace_spec(shapes, dtype="bf16"):
    t, v, h = shapes["num_tokens"], shapes["vocab"], shapes["hidden"]
    return (sds((v, h), dtype), sds((t,), "i32")), {}
