# EAGLE-3 TPU Adaptation Note

Status: design note only. rvLLM does not currently ship a trained EAGLE-3
drafter or claim end-to-end EAGLE-3 parity on TPU.

## Provenance

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang, “EAGLE-3: Scaling up
Inference Acceleration of Large Language Models via Training-Time Test,”
[arXiv:2503.01840](https://arxiv.org/abs/2503.01840). The authors’ public code is
linked from the paper.

This note describes a prospective clean-room adaptation of the published
method. It does not claim that rvLLM code or weights reproduce the paper.

## Algorithm boundary

EAGLE-3 uses multi-layer target features and direct token prediction to draft a
short sequence. The target model verifies the draft in one forward pass. A
greedy implementation accepts the matching prefix and emits a target-model
correction at the first mismatch; stochastic modes require their corresponding
distribution-preserving acceptance rule.

## Proposed rvLLM interface

The adaptation must obtain every dimension and identity from explicit inputs:

- target model and tokenizer revisions and digests;
- hidden size, layer count, feature-layer indices, and attention geometry;
- draft depth, maximum draft length, and context limit;
- drafter checkpoint revision and digest;
- KV-cache capacity, slot mapping, and position metadata.

The drafter would consume the last accepted token plus selected target features
and return bounded draft token IDs. The verifier would return target token IDs,
the accepted length, and committed cache positions without relying on implicit
rollback or stale cache contents.

## Correctness gates

An implementation is not release-ready until it demonstrates:

1. Independent-reference parity for feature capture, draft inputs, target
   verification, and greedy left-tie behavior.
2. Distribution-preserving parity for every supported sampling mode.
3. Cache parity after zero acceptance, partial acceptance, full acceptance,
   ring wrap, and maximum context length.
4. Bounds checks for draft length, token IDs, positions, and cache slots.
5. Deterministic results for fixed inputs, seed, and tie policy.
6. Machine-readable evidence recording model, tokenizer, drafter, source,
   toolchain, and hardware identities.

Any throughput numbers must come from the released implementation and include
the drafter, verifier, acceptance logic, synchronization, and cache maintenance.
Paper results or measurements on different configurations are not rvLLM claims.
