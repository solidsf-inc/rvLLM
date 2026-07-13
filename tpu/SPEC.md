# CUDA → StableHLO porting spec

## Op mapping cheat-sheet

| CUDA / CUTLASS pattern             | StableHLO (via JAX)                                                 |
|------------------------------------|----------------------------------------------------------------------|
| elementwise (add, mul, silu, gelu) | `stablehlo.add/multiply/logistic/erf`                                |
| RMSNorm reduction                  | `stablehlo.reduce` (sum) + `rsqrt` + broadcast mul                   |
| softmax (stable)                   | `reduce(max)` → `sub` → `exp` → `reduce(sum)` → `divide`             |
| RoPE pair rotation                 | slice even/odd, two muls, add/sub, concat                            |
| dense GEMM (FP16/BF16)             | `stablehlo.dot_general`                                              |
| paged GEMV / GQA                   | `dot_general` + `gather` on block table                              |
| flash attention                    | `stablehlo.custom_call` to `jax.nn.dot_product_attention` lowering,  |
|                                    | or hand-rolled tiled matmul + softmax                                |
| copy_blocks (KV reshuffle)         | `dynamic_update_slice` in a `fori_loop` or `jax.lax.scan`            |
| reshape_and_cache                  | `scatter` / `dynamic_update_slice`                                   |
| argmax / lm_head argmax            | `stablehlo.reduce` with `argmax` comparator (`jnp.argmax`)           |
| bias add (broadcast)               | `stablehlo.broadcast_in_dim` + `add`                                 |
| cast fp                            | `stablehlo.convert`                                                  |
| FP8 E4M3 scaled GEMM               | **TODO** — no direct StableHLO op; route through `custom_call`       |
|                                    | `"mhlo.stablehlo_scaled_dot"` or lower manually once XLA supports it |
| TMA / WGMMA                        | **N/A on TPU**; kernels collapse into straight `dot_general`         |
| persistent / megakernel launchers  | not portable — expressed as a sequence of `dot_general` + norms      |

## Dtype strategy

- F32 reference ports are written first; F16/BF16 variants re-use the same
  pure function with a different input dtype.
- **TPU preferred compute dtype is BF16.** The CUDA FP16 kernels should
  lower as BF16 on TPU. FP8 is a v5p-only story and stays TODO.
- INT4 (`gemv_int4`) requires a weight-dequant path: `convert(reshape(bit-unpack))`
  before the `dot_general`.

## Fused kernels

The rvLLM `fused_*` kernels (e.g. `fused_oproj_add_norm_gateup_gemv`) are
compositions of simpler ops. We port the **components** separately and rely
on the XLA fuser to re-fuse on TPU. The manifest records the fused kernel
pointing at the component port.

## Attention family

- `flash_attention.cu`, `flash_attention_3.cu`, `fa3_sm90_wrapper.cu`,
  `flash_attention_3_prefill.cu`, `flash_attention_3_v3.cu` all map to the
  same StableHLO: tiled Q·Kᵀ → scaled softmax → A·V.
- `paged_attention.cu` adds a `gather` on the block table before the K/V
  loads.
- `split_kv_attention.cu` is two sequential attentions concatenated on the
  seq-len axis.

## Persistence and fallbacks

No fallbacks. If a port cannot be expressed in StableHLO today, the port
file raises `NotImplementedError("TODO: ...")`. The emitter treats these as
hard failures, not warnings.
