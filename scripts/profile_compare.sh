#!/usr/bin/env bash
# Legacy entry point for an rvLLM-only batch/profile sweep.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${RVLLM_MODEL_DIR:?set RVLLM_MODEL_DIR}"
: "${RVLLM_MODEL_ID:?set RVLLM_MODEL_ID to a public model identifier}"
: "${RVLLM_MODEL_SHA256:?set RVLLM_MODEL_SHA256 to the model artifact digest}"
: "${RVLLM_SOURCE_SHA:?set RVLLM_SOURCE_SHA to the tested 40-character commit SHA}"
: "${RVLLM_HARDWARE:?set RVLLM_HARDWARE to the exact accelerator description}"
: "${RVLLM_DRIVER:?set RVLLM_DRIVER to the exact driver version}"
: "${RVLLM_TOOLCHAIN:?set RVLLM_TOOLCHAIN to the exact compiler toolchain}"
[[ "$RVLLM_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid RVLLM_SOURCE_SHA" >&2; exit 2; }
[[ "$RVLLM_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid RVLLM_MODEL_SHA256" >&2; exit 2; }

batches="1,8,32"
iters=100
warmup=10
output="results/profile/$(date -u +%Y%m%dT%H%M%SZ)"
run_nsys=0
while (($#)); do
    case "$1" in
        --batches) batches="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --nsys) run_nsys=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

cargo build --manifest-path v3/Cargo.toml --release --locked --features cuda -p rvllm-bench
binary="v3/target/release/rvllm-bench"
[[ -x "$binary" ]] || { echo "missing $binary" >&2; exit 1; }
mkdir -p "$output"

IFS=',' read -r -a batch_values <<< "$batches"
for batch in "${batch_values[@]}"; do
    [[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch: $batch" >&2; exit 2; }
    RVLLM_BATCH="$batch" RVLLM_ITERS="$iters" RVLLM_WARMUP="$warmup" \
        "$binary" > "$output/batch_${batch}.jsonl" 2> "$output/batch_${batch}.log"
    if ((run_nsys)); then
        command -v nsys >/dev/null || { echo "nsys not found" >&2; exit 1; }
        nsys profile --force-overwrite=true --output "$output/batch_${batch}" \
            env RVLLM_BATCH="$batch" RVLLM_ITERS="$iters" RVLLM_WARMUP="$warmup" "$binary"
    fi
done

python3 - "$output/receipt.json" "$RVLLM_SOURCE_SHA" "$RVLLM_HARDWARE" \
    "$RVLLM_MODEL_ID" "$RVLLM_MODEL_SHA256" "$RVLLM_DRIVER" "$RVLLM_TOOLCHAIN" \
    "$batches" "$iters" "$warmup" "$run_nsys" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
files = {}
for item in sorted(path.parent.iterdir()):
    if item.is_file() and item.name != path.name:
        files[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
receipt = {
    "schema": "rvllm.profile.v1",
    "source_sha": sys.argv[2],
    "hardware": sys.argv[3],
    "model": sys.argv[4],
    "model_sha256": sys.argv[5],
    "driver": sys.argv[6],
    "toolchain": sys.argv[7],
    "batches": [int(v) for v in sys.argv[8].split(",")],
    "iters": int(sys.argv[9]),
    "warmup": int(sys.argv[10]),
    "nsys": sys.argv[11] == "1",
    "files": files,
}
canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
echo "$output"
