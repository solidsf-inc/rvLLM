"""Emit a parameterized scan-based decode-step StableHLO module.

Inputs: token_ids + stacked weights + stacked KV caches + metadata
Outputs: token_ids + stacked KV caches
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct as sds


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=positive, required=True)
parser.add_argument("--query-heads", type=positive, required=True)
parser.add_argument("--kv-heads", type=positive, required=True)
parser.add_argument("--head-dim", type=positive, required=True)
parser.add_argument("--intermediate-size", type=positive, required=True)
parser.add_argument("--vocab-size", type=positive, required=True)
parser.add_argument("--num-blocks", type=positive, required=True)
parser.add_argument("--block-size", type=positive, required=True)
parser.add_argument("--max-context-blocks", type=positive, required=True)
parser.add_argument("--num-layers", type=positive, required=True)
parser.add_argument("--max-positions", type=positive, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
options = parser.parse_args()

B = options.batch
H = options.query_heads
KVH = options.kv_heads
HD = options.head_dim
INTER = options.intermediate_size
VOCAB = options.vocab_size
NB = options.num_blocks
BS = options.block_size
MCB = options.max_context_blocks
NUM_LAYERS = options.num_layers
if H % KVH:
    parser.error("--query-heads must be divisible by --kv-heads")
if HD % 2:
    parser.error("--head-dim must be even for split-half RoPE")
if MCB > NB:
    parser.error("--max-context-blocks cannot exceed --num-blocks")

SCALE = 1.0 / (HD ** 0.5)
HIDDEN = H * HD
QKV_DIM = H * HD + 2 * KVH * HD


def gqa_expand(k, nh):
    return jnp.repeat(k, nh // k.shape[-2], axis=-2)


def silu(x):
    return x * jax.nn.sigmoid(x)


def rms_norm(x, g, eps=1e-6):
    ms = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(ms + eps).astype(x.dtype)) * g


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x0, x1 = x[..., :half], x[..., half:]
    return jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)


def one_layer(carry, layer_params):
    x, residual, cos, sin, positions, slot_mapping, block_tables, context_lens = carry
    n1g, qkv_w, o_w, n2g, gu_w, dw, kc, vc = layer_params

    xr = x + residual
    normed = rms_norm(xr, n1g)
    qkv = normed @ qkv_w
    q = qkv[:, :H * HD].reshape(B, H, HD)
    k = qkv[:, H * HD:H * HD + KVH * HD].reshape(B, KVH, HD)
    v = qkv[:, H * HD + KVH * HD:].reshape(B, KVH, HD)
    q, k = rope(q, cos, sin), rope(k, cos, sin)

    fk = kc.reshape(-1, KVH, HD).at[slot_mapping].set(k)
    kc = fk.reshape(kc.shape)
    fv = vc.reshape(-1, KVH, HD).at[slot_mapping].set(v)
    vc = fv.reshape(vc.shape)

    kg = gqa_expand(kc[block_tables].reshape(B, MCB * BS, KVH, HD), H)
    vg = gqa_expand(vc[block_tables].reshape(B, MCB * BS, KVH, HD), H)
    sc = jnp.einsum("bhd,bthd->bht", q, kg) * SCALE
    valid = jnp.arange(MCB * BS)[None, :] < context_lens[:, None]
    sc = jnp.where(valid[:, None, :], sc, jnp.finfo(sc.dtype).min)
    m = jnp.max(sc, axis=-1, keepdims=True)
    p = jnp.exp((sc - m).astype(jnp.float32))
    p = (p / jnp.sum(p, axis=-1, keepdims=True)).astype(q.dtype)
    ao = jnp.einsum("bht,bthd->bhd", p, vg).reshape(B, HIDDEN)

    x2 = (ao @ o_w) + xr
    n2 = rms_norm(x2, n2g)
    gu = n2 @ gu_w
    x_out = (silu(gu[:, :INTER]) * gu[:, INTER:]) @ dw + x2
    residual_out = jnp.zeros_like(x_out)

    new_carry = (x_out, residual_out, cos, sin, positions, slot_mapping, block_tables, context_lens)
    return new_carry, (kc, vc)


def full_step(
    token_ids, embed_w, final_norm_g, lm_head_w, rope_cos, rope_sin,
    positions, slot_mapping, block_tables, context_lens,
    all_n1g, all_qkv_w, all_o_w, all_n2g, all_gu_w, all_dw,
    all_kc, all_vc,
):
    x = embed_w[token_ids]
    residual = jnp.zeros_like(x)
    cos = rope_cos[positions][:, None, :]
    sin = rope_sin[positions][:, None, :]

    init_carry = (x, residual, cos, sin, positions, slot_mapping, block_tables, context_lens)
    layer_params = (all_n1g, all_qkv_w, all_o_w, all_n2g, all_gu_w, all_dw, all_kc, all_vc)

    final_carry, (all_kc_out, all_vc_out) = jax.lax.scan(one_layer, init_carry, layer_params)

    x_final = final_carry[0]
    normed_final = rms_norm(x_final, final_norm_g)
    logits = jnp.dot(normed_final.astype(jnp.float32), lm_head_w.astype(jnp.float32).T)
    token_out = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    return token_out, all_kc_out, all_vc_out


args = [
    sds((B,), jnp.int32),                                        # token_ids
    sds((VOCAB, HIDDEN), jnp.bfloat16),                          # embed_w
    sds((HIDDEN,), jnp.bfloat16),                                # final_norm_g
    sds((VOCAB, HIDDEN), jnp.bfloat16),                          # lm_head_w
    sds((options.max_positions, HD // 2), jnp.float32),          # rope_cos
    sds((options.max_positions, HD // 2), jnp.float32),          # rope_sin
    sds((B,), jnp.int32),                                        # positions
    sds((B,), jnp.int32),                                        # slot_mapping
    sds((B, MCB), jnp.int32),                                    # block_tables
    sds((B,), jnp.int32),                                        # context_lens
    # Stacked per-layer weights (leading dim = NUM_LAYERS)
    sds((NUM_LAYERS, HIDDEN), jnp.bfloat16),                     # all_n1g
    sds((NUM_LAYERS, HIDDEN, QKV_DIM), jnp.bfloat16),            # all_qkv_w
    sds((NUM_LAYERS, HIDDEN, HIDDEN), jnp.bfloat16),             # all_o_w
    sds((NUM_LAYERS, HIDDEN), jnp.bfloat16),                     # all_n2g
    sds((NUM_LAYERS, HIDDEN, 2 * INTER), jnp.bfloat16),          # all_gu_w
    sds((NUM_LAYERS, INTER, HIDDEN), jnp.bfloat16),              # all_dw
    # Stacked KV caches
    sds((NUM_LAYERS, NB, BS, KVH, HD), jnp.bfloat16),            # all_kc
    sds((NUM_LAYERS, NB, BS, KVH, HD), jnp.bfloat16),            # all_vc
]

print(f"Tracing scan-based step: MCB={MCB} ctx={MCB * BS} tokens, {len(args)} args...")
lowered = jax.jit(full_step).lower(*args)
ir = str(lowered.compiler_ir(dialect="stablehlo"))
output = options.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as f:
    f.write(ir)
print(f"wrote {output} ({ir.count(chr(10))} lines)")
