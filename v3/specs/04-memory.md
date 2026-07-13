# Memory

Every allocation has a checked size, alignment, owner, device, and completion
lifetime. Device buffers referenced by a stream or captured graph cannot move
or free until the final dependent event completes. Pinned host buffers used by
asynchronous copies obey the same rule.

Arena offsets, KV page counts, workspaces, and tensor byte lengths use checked
arithmetic. Allocation failure is explicit; zero-sized or overflowed plans do
not alias valid storage. Concurrent requests may share immutable weights but
not mutable metadata, scratch, RNG, or KV ownership without synchronization.

Unified memory is a separate backend choice with residency and synchronization
costs, not an automatic overflow tier. Real-device stress tests must cover
allocation limits, cancellation, graph destruction, and concurrent teardown.
