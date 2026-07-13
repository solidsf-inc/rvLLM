"""Fail-closed placeholder for the numerical parity verifier.

For each `impl` entry in manifest.toml, the verifier will:
  1. build random inputs from the trace spec
  2. run the JAX port on CPU
  3. run the matching CUDA kernel via the rvLLM runtime (ctypes/FFI into
     ../kernels/*.so) against the same inputs
  4. compare with per-dtype tolerances (bf16: atol=5e-2 rtol=5e-2,
     f32: atol=1e-5 rtol=1e-5)

    Step 3 requires an audited rvLLM runtime shim. Until it exists, this
    command deliberately returns failure so it cannot approve a release.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("verify_all: FAIL — CUDA/StableHLO reference parity is not wired.", file=sys.stderr)
    print("No kernel may be marked verified until identical-input comparisons run.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
