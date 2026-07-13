#!/bin/bash
# Build the fallback paged-attention library from the rvLLM split-KV source.
# Source provenance: v3/kernels/fp8_decode_v2.cu in the same rvLLM revision.
set -euo pipefail
ARCH="${1:-sm_90a}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ARCH="${ARCH%a}"
OUT="${DIR}/${OUT_ARCH}"
mkdir -p "${OUT}"
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found" >&2; exit 1; }
command -v nm >/dev/null 2>&1 || { echo "ERROR: nm not found" >&2; exit 1; }
LIB_TMP="${OUT}/libfa_sm89_kernels.so.tmp"
nvcc -shared -o "${LIB_TMP}" \
    "${DIR}/../v3/kernels/fp8_decode_v2.cu" \
    -arch="${ARCH}" -O3 --use_fast_math -Xcompiler -fPIC
ALL_EXPORTS=$(nm -D --defined-only "${LIB_TMP}" | awk '{print $3}')
for sym in rvllm_fa_sm89_abi_version fa_sm89_fp8_output_dtype fa_sm89_fp8_output_element_size fa_sm89_decode_workspace_size fa_sm89_paged_decode fa_sm89_paged_decode_fp8 fa_sm89_paged_prefill_fp8; do
    grep -Fxq "${sym}" <<< "${ALL_EXPORTS}" || {
        echo "ERROR: missing exported symbol ${sym}" >&2
        exit 1
    }
done
mv "${LIB_TMP}" "${OUT}/libfa_sm89_kernels.so"
echo "built ${OUT}/libfa_sm89_kernels.so (${ARCH})"
REVISION="${REVISION:-$(git -C "${DIR}/.." rev-parse HEAD 2>/dev/null || true)}"
"${DIR}/gen_manifest.sh" "${OUT}" "${REVISION}" cuda-ptx-v1 libfa_sm89_kernels.so
