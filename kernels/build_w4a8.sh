#!/usr/bin/env bash
# Compile rvLLM W4A8 GEMM .so on an H100-class system.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-${SCRIPT_DIR}/../cutlass}"
CUTLASS_REVISION="${CUTLASS_REVISION:-da5e086dab31d63815acafdac9a9c5893b1c69e2}"
OUT_DIR="${SCRIPT_DIR}/sm_90"
mkdir -p "${OUT_DIR}"

command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found" >&2; exit 1; }
command -v nm >/dev/null 2>&1 || { echo "ERROR: nm not found" >&2; exit 1; }
[ -d "${CUTLASS_DIR}/include/cutlass" ] || { echo "ERROR: CUTLASS missing at ${CUTLASS_DIR}" >&2; exit 1; }
ACTUAL_CUTLASS_REVISION="$(git -C "${CUTLASS_DIR}" rev-parse HEAD 2>/dev/null || true)"
[ "${ACTUAL_CUTLASS_REVISION}" = "${CUTLASS_REVISION}" ] || {
    echo "ERROR: CUTLASS revision is ${ACTUAL_CUTLASS_REVISION:-unknown}; expected ${CUTLASS_REVISION}" >&2
    exit 1
}

NVCC_FLAGS=(
    -std=c++17
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -fPIC
    -shared
    -O3
    -DNDEBUG
    -I"${CUTLASS_DIR}/include"
    -I"${CUTLASS_DIR}/tools/util/include"
    -I"${CUTLASS_DIR}/examples/55_hopper_mixed_dtype_gemm"
    -lineinfo
)

echo "=== Building rvllm_w4a8 GEMM .so ==="
LIB_TMP="${OUT_DIR}/libw4a8_gemm.so.tmp"
nvcc "${NVCC_FLAGS[@]}" \
    -o "${LIB_TMP}" \
    "${SCRIPT_DIR}/cutlass_w4a8_wrapper.cu"

ALL_EXPORTS=$(nm -D --defined-only "${LIB_TMP}" | awk '{print $3}')
for sym in rvllm_w4a8_gemm_workspace_size rvllm_w4a8_gemm_run rvllm_w4a8_encode_weight_fp16; do
    grep -Fxq "${sym}" <<< "${ALL_EXPORTS}" || {
        echo "ERROR: missing exported symbol ${sym}" >&2
        exit 1
    }
done
mv "${LIB_TMP}" "${OUT_DIR}/libw4a8_gemm.so"

echo "=== Done ==="
ls -la "${OUT_DIR}/libw4a8_gemm.so"
nm -D --defined-only "${OUT_DIR}/libw4a8_gemm.so" | grep ' rvllm_w4a8'
REVISION="${REVISION:-$(git -C "${SCRIPT_DIR}/.." rev-parse HEAD 2>/dev/null || true)}"
"${SCRIPT_DIR}/gen_manifest.sh" "${OUT_DIR}" "${REVISION}" cuda-ptx-v1 libw4a8_gemm.so
