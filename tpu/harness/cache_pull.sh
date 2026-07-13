#!/usr/bin/env bash
# Pull compiled XLA cache from HF to skip cold compilation.
# Usage: ./cache_pull.sh [artifact-name]
set -euo pipefail

HF_REPO="${HF_REPO:?set HF_REPO to an authorized artifact repository}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.jax_cache}"

if [ -n "${1:-}" ]; then
    ARTIFACT="$1"
else
    echo "ERROR: pass an exact artifact path; latest-artifact selection is not reproducible" >&2
    exit 1
fi

echo "downloading $ARTIFACT"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-cache-pull.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
huggingface-cli download "$HF_REPO" "$ARTIFACT" "${ARTIFACT}.sha256" --local-dir "$WORK_DIR"
TAR="${WORK_DIR}/${ARTIFACT}"
CHECKSUM="${WORK_DIR}/${ARTIFACT}.sha256"
[ -f "$TAR" ] && [ -f "$CHECKSUM" ] || { echo "ERROR: artifact or checksum missing" >&2; exit 1; }
EXPECTED=$(tr -d '[:space:]' <"$CHECKSUM")
[[ "$EXPECTED" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: malformed checksum" >&2; exit 1; }
ACTUAL=$(shasum -a 256 "$TAR" | awk '{print $1}')
[ "$ACTUAL" = "$EXPECTED" ] || { echo "ERROR: artifact checksum mismatch" >&2; exit 1; }
python3 - "$TAR" "$(basename "$CACHE_DIR")" <<'PY'
import pathlib
import sys
import tarfile

archive, expected_root = sys.argv[1:]
with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("ERROR: empty cache archive")
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_root:
            raise SystemExit(f"ERROR: unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"ERROR: unsupported archive member: {member.name!r}")
PY
[ ! -e "$CACHE_DIR" ] || { echo "ERROR: cache destination already exists: $CACHE_DIR" >&2; exit 1; }
EXTRACT_DIR="${WORK_DIR}/extract"
mkdir -p "$EXTRACT_DIR"
tar -xzf "$TAR" --no-same-owner -C "$EXTRACT_DIR"
EXTRACTED_CACHE="${EXTRACT_DIR}/$(basename "$CACHE_DIR")"
[ -d "$EXTRACTED_CACHE" ] || { echo "ERROR: archive does not contain $(basename "$CACHE_DIR")" >&2; exit 1; }
mkdir -p "$(dirname "$CACHE_DIR")"
mv "$EXTRACTED_CACHE" "$CACHE_DIR"
echo "restored XLA cache to $CACHE_DIR ($(du -sh "$CACHE_DIR" | cut -f1))"
