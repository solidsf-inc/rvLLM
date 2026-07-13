#!/usr/bin/env bash
# rvLLM benchmark wrapper for GB10 / DGX Spark (sm_121).
#
# Usage:
#   ./scripts/bench_sm121.sh [batch] [iters]
#     batch   default 32   (decode batch size)
#     iters   default 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V3_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V3_DIR/.." && pwd)"

BATCH=${1:-32}
ITERS=${2:-50}
WARMUP=${WARMUP:-10}
for value in "$BATCH" "$ITERS" "$WARMUP"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: batch/iters/warmup must be positive integers" >&2; exit 1; }
done

BIN="$V3_DIR/target/release/rvllm-bench"
if [ ! -x "$BIN" ]; then
    echo "binary not found — build with:"
    echo "  cargo build --release -p rvllm-bench --features gb10"
    exit 1
fi

KERNELS="$REPO_ROOT/kernels"
SM120_SO="$KERNELS/sm_121/libcutlass_sm120.so"
if [ ! -f "$SM120_SO" ]; then
    echo "missing: $SM120_SO"
    echo "build with: $REPO_ROOT/kernels/build_cutlass_sm120_so.sh sm_121a"
    exit 1
fi

MANIFEST="$KERNELS/sm_121/manifest.json"
MANIFEST_ROOT="$KERNELS/sm_121/manifest.sha256"
MODEL="${RVLLM_MODEL_DIR:?set RVLLM_MODEL_DIR to an authorized local model}"
MODEL_REVISION="${MODEL_REVISION:?set MODEL_REVISION to the exact model revision}"
MODEL_SHA256="${RVLLM_MODEL_SHA256:?set RVLLM_MODEL_SHA256 to the exact model artifact digest}"
RUN_ID="${RVLLM_RUN_ID:?set RVLLM_RUN_ID}"
RESULT_DIR="${RVLLM_RESULT_DIR:-${REPO_ROOT}/bench-results/${RUN_ID}}"
for artifact in "$SM120_SO" "$MANIFEST" "$MANIFEST_ROOT"; do
    [ -f "$artifact" ] || { echo "ERROR: missing artifact: $artifact" >&2; exit 1; }
done
(cd "$(dirname "$MANIFEST")" && shasum -a 256 -c "$(basename "$MANIFEST_ROOT")")
[ "$(cat "$MODEL/REVISION" 2>/dev/null)" = "$MODEL_REVISION" ] || {
    echo "ERROR: model revision mismatch" >&2; exit 1; }

verify_hash() {
    local file="$1" expected="$2"
    [ -n "$expected" ] || { echo "ERROR: expected hash missing for $file" >&2; exit 1; }
    [ "$(shasum -a 256 "$file" | awk '{print $1}')" = "$expected" ] || {
        echo "ERROR: artifact hash mismatch: $file" >&2; exit 1; }
}
verify_hash "$SM120_SO" "${CUTLASS_SO_SHA256:?set CUTLASS_SO_SHA256}"
RELEASE_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
MANIFEST_SHA256="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_KERNELS_DIR="$KERNELS"
export RVLLM_CUTLASS_SO="$SM120_SO"
export RVLLM_RELEASE_REVISION="$RELEASE_REVISION"
export RVLLM_KERNEL_ARCH=sm_121
export RVLLM_KERNEL_MANIFEST_SHA256="$MANIFEST_SHA256"
export RVLLM_SOURCE_SHA="$RELEASE_REVISION"
export RVLLM_MODEL_ID="${RVLLM_MODEL_ID:-gemma-4@${MODEL_REVISION}}"
export RVLLM_MODEL_SHA256="$MODEL_SHA256"
export RVLLM_HARDWARE="${RVLLM_HARDWARE:-$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)}"
export RVLLM_DRIVER="${RVLLM_DRIVER:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)}"
export RVLLM_TOOLCHAIN="${RVLLM_TOOLCHAIN:-$(rustc --version)}"
export RVLLM_BATCH="$BATCH"
export RVLLM_ITERS="$ITERS"
export RVLLM_WARMUP="$WARMUP"
export RVLLM_CUTLASS_SM120_SO="$SM120_SO"
export RVLLM_FP8_GEMM_CUTLASS_SM120=1    # opt-in the CUTLASS blockwise path
export RVLLM_ARENA_GB="${RVLLM_ARENA_GB:-40}"
# Gemma 4 global attention has head_dim=512; the FA2 f16-KV route supports
# at most 256, so this benchmark must use the FP8-KV decode kernel.
export RVLLM_F16_KV="${RVLLM_F16_KV:-0}"

mkdir -p "$RESULT_DIR"
export RUN_ID RESULT_DIR MODEL_REVISION
python3 - <<'PY' >"${RESULT_DIR}/metadata.json"
import json, os, pathlib, platform, subprocess
root = pathlib.Path(os.environ["RVLLM_KERNELS_DIR"]).parent
def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
metadata = {
    "schema": "rvllm-benchmark-metadata-v1",
    "run_id": os.environ["RUN_ID"],
    "revision": output(["git", "-C", str(root), "rev-parse", "HEAD"]),
    "model_revision": os.environ["MODEL_REVISION"],
    "gpu": output(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
    "platform": platform.platform(),
    "batch": int(os.environ["RVLLM_BATCH"]),
    "iters": int(os.environ["RVLLM_ITERS"]),
    "warmup": int(os.environ["RVLLM_WARMUP"]),
}
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

echo "== rvLLM benchmark (sm_121 / CUTLASS blockwise) =="
echo "  batch=$BATCH iters=$ITERS warmup=$WARMUP"
echo "  model=$MODEL"
echo "  metadata=${RESULT_DIR}/metadata.json"
echo

"$BIN" | tee "${RESULT_DIR}/result.jsonl"
