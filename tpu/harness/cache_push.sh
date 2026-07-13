#!/usr/bin/env bash
# Push compiled XLA cache to HF for reuse across deploys.
# Usage: ./cache_push.sh
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-${HOME}/.jax_cache}"
HF_REPO="${HF_REPO:?set HF_REPO to an authorized artifact repository}"
ARTIFACT="${ARTIFACT:-xla-cache}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the full rvLLM commit}"
TARGET="${TARGET:-tpu}"
ALLOW_CACHE_UPLOAD="${ALLOW_CACHE_UPLOAD:-0}"
JAX_VER=$(python3 -c "import jax; print(jax.__version__)")
[ "$ALLOW_CACHE_UPLOAD" = "1" ] || {
    echo "ERROR: set ALLOW_CACHE_UPLOAD=1 after auditing the cache contents" >&2
    exit 1
}

if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR")" ]; then
    echo "no XLA cache at $CACHE_DIR"
    exit 1
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-cache-push.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
TAR="${WORK_DIR}/${ARTIFACT}-${SOURCE_REVISION}.tar.gz"
REMOTE_PATH="xla/${TARGET}/${ARTIFACT}-jax${JAX_VER}-${SOURCE_REVISION}.tar.gz"
echo "packaging cache: $(du -sh "$CACHE_DIR" | cut -f1)"
tar -czf "$TAR" -C "$(dirname "$CACHE_DIR")" "$(basename "$CACHE_DIR")"
tar -tzf "$TAR" >/dev/null
shasum -a 256 "$TAR" | awk '{print $1}' >"${TAR}.sha256"
echo "uploading audited artifact to $HF_REPO"
huggingface-cli upload "$HF_REPO" "$TAR" "$REMOTE_PATH"
huggingface-cli upload "$HF_REPO" "${TAR}.sha256" "${REMOTE_PATH}.sha256"
echo "done: ${REMOTE_PATH}"
