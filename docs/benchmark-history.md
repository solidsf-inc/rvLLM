# Benchmark history

[`bench.html`](bench.html) preserves the project-reported TPU v6e-4
measurements from April–June 2026. The old H100 rows were removed after
revalidation found invalid KV-page and block-table geometry.

The retained snapshot does not include raw receipts or immutable model
revisions. New benchmark rows must include the source SHA, model revision,
hardware, driver/toolchain, command, warmup, raw samples, failures, and a
content hash.
