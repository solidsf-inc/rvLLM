#!/usr/bin/env bash
# Reproducible decode and generation spread measurement.
set -euo pipefail
umask 077

BENCH=${BENCH:-./rvllm-bench}
BATCHES=${BATCHES:-"1 2 4 8 16 32 64 128 256"}
ITERS=${ITERS:-40}
WARMUP=${WARMUP:-8}
OUT_DIR=${OUT_DIR:?set OUT_DIR to a new artifact directory}

[[ -x "$BENCH" && ! -e "$OUT_DIR" ]]
[[ "$ITERS" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[0-9]+$ ]]
read -r -a batch_values <<<"$BATCHES"
((${#batch_values[@]} > 0))
for batch in "${batch_values[@]}"; do
  [[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch: $batch" >&2; exit 2; }
done
if [[ -n ${PROMPT_FILE:-} ]]; then [[ -f "$PROMPT_FILE" && ! -L "$PROMPT_FILE" ]]; fi
mkdir -p "$OUT_DIR"

{
  printf 'bench_sha256\t'; shasum -a 256 "$BENCH" | awk '{print $1}'
  printf 'system\t%s\n' "$(uname -srm)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
} >"$OUT_DIR/environment.tsv"
: >"$OUT_DIR/commands.tsv"

run_decode() {
  local route=$1 batch=$2 prefix="$OUT_DIR/decode-${1}-${2}"
  local -a route_env
  if [[ "$route" == cutlass ]]; then route_env=(RVLLM_FP8_GEMM_LT_M1=0); else route_env=(RVLLM_FP8_GEMM_LT_MAX_M=512); fi
  printf '%s\t%s\t%s\t%s\n' decode "$route" "$batch" "${route_env[*]}" >>"$OUT_DIR/commands.tsv"
  env -u RVLLM_FP8_GEMM_LT_M1 -u RVLLM_FP8_GEMM_LT_MAX_M "${route_env[@]}" \
    RVLLM_BATCH="$batch" RVLLM_ITERS="$ITERS" RVLLM_WARMUP="$WARMUP" \
    "$BENCH" >"$prefix.stdout" 2>"$prefix.stderr"
  printf 'B=%s %s: %s\n' "$batch" "$route" "$(tail -n 1 "$prefix.stdout")"
}

for batch in "${batch_values[@]}"; do
  run_decode cutlass "$batch"
  run_decode cublaslt "$batch"
done

run_generate() {
  local name=$1; shift
  [[ "$name" =~ ^[a-z0-9_-]+$ ]]
  printf '%s\t%s\t%s\n' generate "$name" "$*" >>"$OUT_DIR/commands.tsv"
  env RVLLM_DECODE_GRAPH=1 "$@" "$BENCH" --generate \
    >"$OUT_DIR/generate-${name}.stdout" 2>"$OUT_DIR/generate-${name}.stderr"
  tail -n 1 "$OUT_DIR/generate-${name}.stdout"
  grep -E 'decoded in|prefill|^\[spec\]' "$OUT_DIR/generate-${name}.stderr" | tail -n 3 || true
}

run_generate fp8_short RVLLM_F16_KV=0 RVLLM_GEN_TOKENS=400 RVLLM_GEN_PROMPT=32
run_generate f16_short RVLLM_GEN_TOKENS=400 RVLLM_GEN_PROMPT=32
run_generate spec_short RVLLM_F16_KV=0 RVLLM_SPEC_DECODE=1 RVLLM_SPEC_K=4 RVLLM_GEN_TOKENS=400 RVLLM_GEN_PROMPT=32
if [[ -n ${PROMPT_FILE:-} ]]; then
  run_generate fp8_long RVLLM_F16_KV=0 RVLLM_GEN_TOKENS=300 RVLLM_GEN_PROMPT_FILE="$PROMPT_FILE"
  run_generate spec_long RVLLM_F16_KV=0 RVLLM_SPEC_DECODE=1 RVLLM_SPEC_K=4 RVLLM_GEN_TOKENS=300 RVLLM_GEN_PROMPT_FILE="$PROMPT_FILE"
fi

(
  cd "$OUT_DIR"
  checksum_tmp=$(mktemp .SHA256SUMS.XXXXXX)
  trap 'rm -f "$checksum_tmp"' EXIT
  find . -type f ! -name SHA256SUMS ! -name '.SHA256SUMS.*' -print0 |
    LC_ALL=C sort -z | xargs -0 shasum -a 256 >"$checksum_tmp"
  mv "$checksum_tmp" SHA256SUMS
  trap - EXIT
)
