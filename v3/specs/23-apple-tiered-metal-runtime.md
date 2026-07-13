# Apple tiered Metal runtime

This is a generic residency design for models larger than the preferred Metal
working set. It contains no application-specific model roles or latent bridge.

Weights are divided into immutable layer groups. A bounded manager maps local
files read-only, validates index entries and tensor extents, stages the next
group, and evicts only after the prior command buffer completes. At most the
declared number of groups may be resident. Paging errors, memory pressure, and
missing files fail the request rather than substitute zero weights.

Security requirements include root-contained paths, checked offsets and sizes,
file identity/digest verification, no network fetch, and no secret paths in
logs. Correctness requires resident and tiered logits parity, eviction-fence
tests, truncated/corrupt file rejection, and memory-pressure tests.

Status: unimplemented design. Existing Metal components do not establish this
tiered runtime or end-to-end serving support.
