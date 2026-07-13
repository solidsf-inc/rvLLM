# Metadata

Device metadata is generated from one validated batch plan and frozen for the
duration of a launch or graph replay. The layout records version, region
offsets, element counts, alignment, dtypes, and total bytes.

All size/offset products and sums are checked. Validation rejects overlap,
misalignment, truncation, unknown versions, inconsistent counts, invalid page
indices, out-of-range sequence lengths, and a total size that differs from the
backing allocation. Uploads cannot partially update an orthogonal view of the
same frozen block.

Round-trip tests, malformed-plan property tests, and graph-fingerprint tests
are required for every layout change.
