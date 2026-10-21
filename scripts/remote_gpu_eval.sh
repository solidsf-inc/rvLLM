#!/usr/bin/env bash
# Remote GPU evaluation runner invoked on leased Vast.ai instances by scripts/vast_ptx_tuner.py.
# 1. Probes GPU attributes via nvidia-smi.
# 2. Compiles the synthesized .cu source with nvcc -O3 --ptxas-options=-v and verifies 0 spills.
# 3. Writes gpu_eval_receipt.json.
set -euo pipefail

CU_FILE="${1:?usage: remote_gpu_eval.sh <kernel.cu> <ptx_target> <repeats>}"
PTX_TARGET="${2:-sm_90a}"
REPEATS="${3:-5}"

GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader | head -n 1)"

nvcc -O3 -std=c++17 "-arch=${PTX_TARGET}" --ptxas-options=-v \
  -Xcompiler -fPIC -shared "${CU_FILE}" -o kernel_eval.so 2> ptxas_build.log

if grep -E "[1-9][0-9]* bytes spill" ptxas_build.log; then
  echo "ERROR: register spills detected in ptxas_build.log" >&2
  cat ptxas_build.log >&2
  exit 3
fi

python3 - "${CU_FILE}" "${PTX_TARGET}" "${GPU_INFO}" "${REPEATS}" <<'PY'
import json
import pathlib
import sys

cu_file, ptx_target, gpu_info, repeats = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ptxas_log = pathlib.Path("ptxas_build.log").read_text(encoding="utf-8", errors="replace")

receipt = {
    "schema": "rvllm.ptx_tuner.gpu_eval_receipt.v1",
    "cu_file": cu_file,
    "ptx_target": ptx_target,
    "gpu_info": gpu_info.strip(),
    "zero_spills_verified": True,
    "ptxas_log": ptxas_log.strip(),
    "repeats": repeats,
}
pathlib.Path("gpu_eval_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
PY
