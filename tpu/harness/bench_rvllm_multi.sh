#!/bin/bash
set -euo pipefail

RVLLM_ROOT="${RVLLM_ROOT:?set RVLLM_ROOT to the rvLLM checkout}"
RVLLM_RUN_ID="${RVLLM_RUN_ID:?set RVLLM_RUN_ID for reproducible output naming}"
RVLLM_ARCH="${RVLLM_ARCH:-sm_90}"
RVLLM_MODEL_ROOT="${RVLLM_MODEL_ROOT:?set RVLLM_MODEL_ROOT to authorized local models}"
: "${RVLLM_BENCH_MODELS:?set RVLLM_BENCH_MODELS to space-separated supported model directory names}"
read -r -a RVLLM_BENCH_MODEL_LIST <<< "$RVLLM_BENCH_MODELS"
((${#RVLLM_BENCH_MODEL_LIST[@]} > 0)) || {
  echo "ERROR: RVLLM_BENCH_MODELS must name at least one model" >&2
  exit 1
}
export RVLLM_KERNELS_DIR="${RVLLM_KERNELS_DIR:-${RVLLM_ROOT}/kernels}"
RVLLM_KERNEL_ARCH_DIR="${RVLLM_KERNELS_DIR}/${RVLLM_ARCH}"
export RVLLM_CUTLASS_SO="${RVLLM_CUTLASS_SO:-${RVLLM_KERNEL_ARCH_DIR}/libcutlass_kernels.so}"
export RVLLM_FA3_SO="${RVLLM_FA3_SO:-${RVLLM_KERNEL_ARCH_DIR}/libfa3_kernels.so}"
export RVLLM_POLICY="${RVLLM_POLICY:-${RVLLM_KERNEL_ARCH_DIR}/policy.json}"
export RVLLM_ITERS="${RVLLM_ITERS:-512}"
export RVLLM_WARMUP="${RVLLM_WARMUP:-8}"
export RVLLM_REAL_PREFILL="${RVLLM_REAL_PREFILL:-1}"
export RVLLM_PREFILL_LEN="${RVLLM_PREFILL_LEN:-16}"
export RVLLM_TTFT="${RVLLM_TTFT:-1}"
export RVLLM_ARENA_GB="${RVLLM_ARENA_GB:-70}"
export RVLLM_BLOCK_SIZE="${RVLLM_BLOCK_SIZE:-256}"
export RVLLM_NAN_CHECK="${RVLLM_NAN_CHECK:-1}"

export RVLLM_RELEASE_REVISION="${RVLLM_RELEASE_REVISION:-$(git -C "$RVLLM_ROOT" rev-parse HEAD)}"
[[ "$RVLLM_RELEASE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ERROR: RVLLM_RELEASE_REVISION must be a 40-character lowercase commit" >&2
  exit 1
}
export RVLLM_SOURCE_SHA="${RVLLM_SOURCE_SHA:-$RVLLM_RELEASE_REVISION}"
export RVLLM_KERNEL_ARCH="${RVLLM_KERNEL_ARCH:-$RVLLM_ARCH}"
export RVLLM_KERNEL_MANIFEST_SHA256="${RVLLM_KERNEL_MANIFEST_SHA256:-$(shasum -a 256 "$RVLLM_KERNEL_ARCH_DIR/manifest.json" | awk '{print $1}')}"
export RVLLM_POLICY_SHA256="${RVLLM_POLICY_SHA256:-$(shasum -a 256 "$RVLLM_POLICY" | awk '{print $1}')}"
if [ -z "${RVLLM_HARDWARE:-}" ] || [ -z "${RVLLM_DRIVER:-}" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
fi
export RVLLM_HARDWARE="${RVLLM_HARDWARE:-$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)}"
export RVLLM_DRIVER="${RVLLM_DRIVER:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)}"
export RVLLM_TOOLCHAIN="${RVLLM_TOOLCHAIN:-$(rustc --version)}"

BENCH="${RVLLM_BENCH:-${RVLLM_ROOT}/v3/target/release/rvllm-bench}"
[ -x "$BENCH" ] || { echo "ERROR: benchmark binary is not executable: $BENCH" >&2; exit 1; }
for artifact in "$RVLLM_CUTLASS_SO" "$RVLLM_FA3_SO" "$RVLLM_POLICY"; do
  [ -f "$artifact" ] || { echo "ERROR: required artifact missing: $artifact" >&2; exit 1; }
done
printf 'run_id=%s revision=%s arch=%s\n' \
  "$RVLLM_RUN_ID" "$RVLLM_RELEASE_REVISION" "$RVLLM_ARCH"

for MODEL_NAME in "${RVLLM_BENCH_MODEL_LIST[@]}"; do
  [[ "$MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid model directory name: $MODEL_NAME" >&2
    exit 1
  }
  echo "=== $MODEL_NAME ==="
  export RVLLM_MODEL_DIR="${RVLLM_MODEL_ROOT}/${MODEL_NAME}"
  [ -d "$RVLLM_MODEL_DIR" ] || { echo "ERROR: model missing: $RVLLM_MODEL_DIR" >&2; exit 1; }
  export RVLLM_MODEL_ID="${MODEL_NAME}@${RVLLM_MODEL_REVISION:-local}"
  DIGEST_SUFFIX=$(printf '%s' "$MODEL_NAME" | LC_ALL=C tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9_' '_')
  DIGEST_VAR="RVLLM_MODEL_SHA256_${DIGEST_SUFFIX}"
  MODEL_DIGEST="${!DIGEST_VAR:-}"
  [ -n "$MODEL_DIGEST" ] || {
    echo "ERROR: set $DIGEST_VAR to the immutable model digest" >&2
    exit 1
  }
  [[ "$MODEL_DIGEST" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: $DIGEST_VAR must be 64 lowercase hex characters" >&2
    exit 1
  }
  export RVLLM_MODEL_SHA256="$MODEL_DIGEST"
  for N in 1 8 16 64 128 256 512; do
    export RVLLM_BATCH=$N
    RESULT=$("$BENCH" 2>&1)
    TOKS=$(printf '%s\n' "$RESULT" | sed -n 's/.*"tok_per_sec":\([0-9.][0-9.]*\).*/\1/p' | tail -1)
    NAN_LINES=$(printf '%s\n' "$RESULT" | grep -c '\[NaN\]' || true)
    [ -n "$TOKS" ] || { echo "ERROR: benchmark emitted no tok_per_sec" >&2; exit 1; }
    echo "N=$N  toks=${TOKS:-0}  nan_layers=$NAN_LINES"
  done
done
echo "=== ALL DONE ==="
