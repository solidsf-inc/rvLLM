# Benchmark history

[`bench.html`](bench.html) preserves the project-reported TPU v6e-4
measurements from April–June 2026. The old H100 rows were removed after
revalidation found invalid KV-page and block-table geometry.

The v0.3.0 H100 replacement uses the corrected geometry and 30 fresh samples
from the exact v0.3.0 kernel ZIP. Its raw public samples and validation receipts are
in [`receipts/h100-2026-07-13/`](receipts/h100-2026-07-13/). The retained TPU
snapshot does not include raw receipts or immutable model revisions.
