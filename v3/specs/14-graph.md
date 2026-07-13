# Graph capture

A captured graph is keyed by a complete launch fingerprint: backend/device,
model and kernel artifact identities, bucket/shape, dtypes, buffer addresses and
sizes, metadata layout version, workspace, and control-flow options. A mismatch
requires recapture or an error.

Capture failure destroys partial graph state. Replay input updates complete
before launch. Graphs, executable instances, modules, and referenced buffers
remain alive until all in-flight replays fence; eviction and shutdown wait on
those fences.

Tests cover fingerprint changes, failed capture cleanup, concurrent replay,
eviction under load, cancellation, and eager/replay output identity.
