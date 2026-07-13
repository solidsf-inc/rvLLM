# DFlash and DDTree TPU Port Note

Status: design note only. rvLLM does not currently ship a trained DFlash
drafter or an end-to-end DFlash/DDTree TPU implementation.

## Provenance

- DFlash: Jian Chen, Yesheng Liang, and Zhijian Liu, “DFlash: Block Diffusion
  for Flash Speculative Decoding,” [arXiv:2602.06036](https://arxiv.org/abs/2602.06036).
- DDTree: Liran Ringel and Yaniv Romano, “Accelerating Speculative Decoding
  with Block Diffusion Draft Trees,”
  [arXiv:2604.12989](https://arxiv.org/abs/2604.12989).

This document describes a prospective clean-room adaptation of the published
algorithms. It does not claim parity with either paper or reuse their code.

## Algorithm boundary

DFlash replaces sequential autoregressive drafting with a lightweight block
diffusion drafter. A target model verifies the proposed block and accepts a
matching prefix without changing the target distribution.

DDTree consumes the block drafter’s per-position distributions, constructs a
bounded draft tree, and verifies it with an ancestor-only attention mask.

## Proposed rvLLM interface

All model and runtime dimensions must be supplied from a validated public model
configuration. No shape below is implicit:

- hidden size, layer count, attention head counts, and head dimension;
- draft block size and optional DDTree node budget;
- target feature-layer indices;
- context length, KV-cache capacity, and cache slot mapping;
- vocabulary size, mask token, and tokenizer revision;
- drafter checkpoint identity and content digest.

The DFlash draft step would accept an anchor token, masked draft positions, and
selected target features. It would return per-position logits or probabilities
for one block. The DDTree step would return bounded token, parent, and position
arrays suitable for a single target verification call.

## Correctness gates

An implementation is not release-ready until it demonstrates:

1. Exact greedy-prefix acceptance against an independent reference.
2. Distribution-preserving stochastic acceptance for supported sampling modes.
3. Cache parity across rejection, full acceptance, ring wrap, and maximum
   context length.
4. Ancestor-mask parity for every DDTree node and position.
5. Deterministic tree construction for a fixed seed and tie policy.
6. Shape, allocation, token-ID, parent-index, and cache-slot bounds checks.
7. Model, tokenizer, drafter, source, toolchain, and hardware identities in
   machine-readable test output.

Performance must be measured end to end on the released implementation. Paper
results and measurements from other models or accelerators are not rvLLM
throughput claims.
