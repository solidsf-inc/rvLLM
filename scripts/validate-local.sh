#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo fmt --manifest-path v3/Cargo.toml --all -- --check
cargo check --manifest-path v3/Cargo.toml --workspace --all-targets --locked
cargo test --manifest-path v3/Cargo.toml --workspace --locked
cargo check --manifest-path chat-client/Cargo.toml --locked

pycache=$(mktemp -d)
trap 'rm -rf "$pycache"' EXIT
PYTHONPYCACHEPREFIX="$pycache" python3 -m compileall -q \
    deploy scripts tests tpu fp8_precision_check.py
while IFS= read -r script; do bash -n "$script"; done < <(find scripts tests -name '*.sh' -type f -print)

if rg -n '/Users/[^/]+/|/root/\.ssh|StrictHostKeyChecking=no|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|hf_[A-Za-z0-9]{20,}' \
    --glob '!Cargo.lock' \
    --glob '!chat-client/Cargo.lock' \
    --glob '!scripts/validate-local.sh' \
    --glob '!.github/workflows/ci.yml' .; then
    echo "release-private marker found" >&2
    exit 1
fi

echo "rvLLM local validation passed"
