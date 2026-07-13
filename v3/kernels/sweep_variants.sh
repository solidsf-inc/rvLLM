#!/usr/bin/env bash
# Reproducible CUTLASS FP8 GEMM variant sweep.
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BENCH:-"$ROOT/target/release/rvllm-bench"}
MK=${POLICY_GENERATOR:-"$ROOT/kernels/make_policy.py"}
OUT_DIR=${OUT_DIR:?set OUT_DIR to a new artifact directory}
NONRES_CANDIDATES=${NONRES_CANDIDATES:-"0 1 2 3 4"}
RES_CANDIDATES=${RES_CANDIDATES:-"100"}

: "${RVLLM_MODEL_DIR:?}"
: "${RVLLM_KERNELS_DIR:?}"
: "${RVLLM_CUTLASS_SO:?}"
: "${RVLLM_FA3_SO:?}"
: "${RVLLM_KERNEL_MANIFEST_SHA256:?}"
: "${RVLLM_MODEL_ID:?}"
: "${RVLLM_MODEL_SHA256:?}"
: "${RVLLM_SOURCE_SHA:?}"
: "${RVLLM_HARDWARE:?}"
: "${RVLLM_DRIVER:?}"
: "${RVLLM_TOOLCHAIN:?}"
: "${POLICY_REVISION:?full 40-hex source revision required}"
: "${GENERATOR_REVISION:?full 40-hex generator revision required}"
: "${POLICY_ARCH:?}"
: "${POLICY_HIDDEN:?}"
: "${POLICY_Q_HEADS:?}"
: "${POLICY_KV_HEADS:?}"
: "${POLICY_HEAD_DIM:?}"
: "${POLICY_INTERMEDIATE:?}"
: "${POLICY_VOCAB:?}"
: "${POLICY_BUCKETS:?comma-separated positive integers required}"
: "${POLICY_WORKSPACE_BYTES:?}"

[[ -x "$BIN" && -f "$MK" && -d "$RVLLM_MODEL_DIR" && -d "$RVLLM_KERNELS_DIR" ]]
[[ -f "$RVLLM_CUTLASS_SO" && -f "$RVLLM_FA3_SO" ]]
[[ -f "$RVLLM_KERNELS_DIR/$POLICY_ARCH/manifest.json" ]]
[[ ! -e "$OUT_DIR" ]]
mkdir -p "$OUT_DIR"

[[ "$POLICY_REVISION" =~ ^[0-9a-f]{40}$ && "$GENERATOR_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$RVLLM_SOURCE_SHA" = "$POLICY_REVISION" ]]
[[ "$RVLLM_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$RVLLM_KERNEL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$POLICY_ARCH" =~ ^sm_(80|89|90|100|121)$ ]]

for value in ${RVLLM_BATCH:-128} ${RVLLM_ITERS:-30} ${RVLLM_WARMUP:-5} \
  $POLICY_HIDDEN $POLICY_Q_HEADS $POLICY_KV_HEADS $POLICY_HEAD_DIM \
  $POLICY_INTERMEDIATE $POLICY_VOCAB $POLICY_WORKSPACE_BYTES; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "invalid positive integer: $value" >&2; exit 2; }
done
for value in $NONRES_CANDIDATES $RES_CANDIDATES; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "invalid variant: $value" >&2; exit 2; }
done
for value in $NONRES_CANDIDATES; do
  [[ " 0 1 2 3 4 " == *" $value "* ]] || { echo "invalid non-residual variant: $value" >&2; exit 2; }
done
for value in $RES_CANDIDATES; do
  [[ "$value" = 100 ]] || { echo "invalid residual variant: $value" >&2; exit 2; }
done

{
  printf 'bench_sha256\t'; shasum -a 256 "$BIN" | awk '{print $1}'
  printf 'cutlass_sha256\t'; shasum -a 256 "$RVLLM_CUTLASS_SO" | awk '{print $1}'
  printf 'fa3_sha256\t'; shasum -a 256 "$RVLLM_FA3_SO" | awk '{print $1}'
  printf 'source_revision\t%s\n' "$POLICY_REVISION"
  printf 'generator_revision\t%s\n' "$GENERATOR_REVISION"
  printf 'kernel_manifest_sha256\t%s\n' "$RVLLM_KERNEL_MANIFEST_SHA256"
  printf 'system\t%s\n' "$(uname -srm)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
} >"$OUT_DIR/environment.tsv"

: >"$OUT_DIR/results.tsv"
for nr in $NONRES_CANDIDATES; do
  for res in $RES_CANDIDATES; do
    policy="$OUT_DIR/policy-${nr}-${res}.json"
    python3 "$MK" "$policy" "$POLICY_REVISION" \
      --arch "$POLICY_ARCH" --hidden "$POLICY_HIDDEN" --q-heads "$POLICY_Q_HEADS" \
      --kv-heads "$POLICY_KV_HEADS" --head-dim "$POLICY_HEAD_DIM" \
      --intermediate "$POLICY_INTERMEDIATE" --vocab "$POLICY_VOCAB" \
      --buckets "$POLICY_BUCKETS" --nonres-variant "$nr" --residual-variant "$res" \
      --workspace-bytes "$POLICY_WORKSPACE_BYTES" --generator-revision "$GENERATOR_REVISION"
    log="$OUT_DIR/bench-${nr}-${res}"
    policy_sha256=$(shasum -a 256 "$policy" | awk '{print $1}')
    RVLLM_POLICY="$policy" RVLLM_POLICY_SHA256="$policy_sha256" \
      RVLLM_RELEASE_REVISION="$POLICY_REVISION" RVLLM_KERNEL_ARCH="$POLICY_ARCH" \
      RVLLM_BATCH=${RVLLM_BATCH:-128} RVLLM_ITERS=${RVLLM_ITERS:-30} \
      RVLLM_WARMUP=${RVLLM_WARMUP:-5} \
      "$BIN" >"$log.stdout" 2>"$log.stderr"
    result=$(tail -n 1 "$log.stdout")
    printf '%s\t%s\t%s\t%s\n' "$nr" "$res" "$policy_sha256" "$result" | tee -a "$OUT_DIR/results.tsv"
  done
done
