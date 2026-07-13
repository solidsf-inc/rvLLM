#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${RVLLM_MODEL_DIR:?set RVLLM_MODEL_DIR}"
: "${RVLLM_MODEL_ID:?set RVLLM_MODEL_ID}"
: "${RVLLM_MODEL_SHA256:?set RVLLM_MODEL_SHA256}"
: "${RVLLM_SOURCE_SHA:?set RVLLM_SOURCE_SHA}"
: "${RVLLM_HARDWARE:?set RVLLM_HARDWARE}"
: "${RVLLM_DRIVER:?set RVLLM_DRIVER}"
: "${RVLLM_TOOLCHAIN:?set RVLLM_TOOLCHAIN}"
[[ "$RVLLM_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid RVLLM_SOURCE_SHA" >&2; exit 2; }
[[ "$RVLLM_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid RVLLM_MODEL_SHA256" >&2; exit 2; }
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }

batch="${RVLLM_PROFILE_BATCH:-1}"
output="${RVLLM_PROFILE_OUTPUT:-results/ncu/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output"
cargo build --manifest-path v3/Cargo.toml --release --locked --features cuda -p rvllm-bench
binary="v3/target/release/rvllm-bench"
ncu --set "${RVLLM_NCU_SET:-roofline}" --target-processes all --force-overwrite \
    --export "$output/rvllm_batch_${batch}" env RVLLM_BATCH="$batch" \
    RVLLM_ITERS="${RVLLM_PROFILE_ITERS:-20}" RVLLM_WARMUP="${RVLLM_PROFILE_WARMUP:-5}" \
    "$binary" > "$output/benchmark.jsonl" 2> "$output/benchmark.log"

python3 - "$output/receipt.json" "$RVLLM_SOURCE_SHA" "$RVLLM_HARDWARE" \
    "$RVLLM_MODEL_ID" "$RVLLM_MODEL_SHA256" "$RVLLM_DRIVER" "$RVLLM_TOOLCHAIN" \
    "$batch" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).parent
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.iterdir()) if p.is_file() and p.name != "receipt.json"}
receipt = {"schema": "rvllm.ncu.v1", "source_sha": sys.argv[2],
           "hardware": sys.argv[3], "model": sys.argv[4],
           "model_sha256": sys.argv[5], "driver": sys.argv[6],
           "toolchain": sys.argv[7], "batch": int(sys.argv[8]), "files": files}
raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
(root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
echo "$output"
