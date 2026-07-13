#!/usr/bin/env bash
set -euo pipefail

readonly TOOLCHAIN=1.95.0
readonly MANIFEST=v3/Cargo.toml

command -v rustup >/dev/null || {
  echo "rustup is required; install it from https://rustup.rs and rerun." >&2
  exit 1
}
command -v cargo >/dev/null || {
  echo "cargo is required." >&2
  exit 1
}

rustup toolchain install "$TOOLCHAIN" --profile minimal --component clippy,rustfmt

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    xcrun -sdk macosx metal --version >/dev/null 2>&1 || {
      echo "Metal builds require Xcode Metal tools; core checks can still run." >&2
    }
    ;;
  Linux-x86_64)
    command -v nvcc >/dev/null || {
      echo "CUDA builds require a supported NVIDIA CUDA toolkit; core checks can still run." >&2
    }
    ;;
  *)
    echo "Unsupported accelerator platform; running portable workspace checks only." >&2
    ;;
esac

cargo +"$TOOLCHAIN" check --manifest-path "$MANIFEST" --workspace --all-targets --locked
echo "rvLLM prerequisites and the locked workspace check passed."
