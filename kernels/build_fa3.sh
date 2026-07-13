#!/usr/bin/env bash
# Build FA3 SM90 paged attention as libfa3_kernels.so.
# Supported head dimensions: 128 and 256. Other dimensions must use a
# separately validated fallback.
#
# FlashAttention source: Dao-AILab/flash-attention at FA3_REVISION below,
# BSD-3-Clause license. FA3_DIR must name that checkout's hopper directory.
#
# Parallel build: each .cu compiles to .o independently, then links once.
# Cuts wall time from ~5min to ~1-2min on a 96-core box.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FA3_DIR="${FA3_DIR:-${SCRIPT_DIR}/../flash-attention/hopper}"
CUTLASS_DIR="${CUTLASS_DIR:-${SCRIPT_DIR}/../cutlass}"
FA3_REVISION="${FA3_REVISION:-1233b73b6c95340c65c9edfe929611838354fc6e}"
CUTLASS_REVISION="${CUTLASS_REVISION:-da5e086dab31d63815acafdac9a9c5893b1c69e2}"
OUT_DIR="${SCRIPT_DIR}/sm_90"
OBJ_DIR="${OUT_DIR}/.fa3_obj"
# Core count: nproc on Linux (the H100 box), sysctl on macOS, else 1.
if [ -z "${JOBS:-}" ]; then
    JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi

# --- Preflight: fail loud before launching any nvcc (deploy hygiene) --------
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not on PATH"; exit 1; }
command -v nm >/dev/null 2>&1 || { echo "ERROR: nm not on PATH"; exit 1; }
[ -d "${CUTLASS_DIR}/include" ] || {
    echo "ERROR: CUTLASS headers not found at ${CUTLASS_DIR}/include (set CUTLASS_DIR)"; exit 1; }
[ -d "${FA3_DIR}/instantiations" ] || {
    echo "ERROR: FA3 Hopper source not found at ${FA3_DIR} (set FA3_DIR)"; exit 1; }
FA3_ROOT="$(cd "${FA3_DIR}/.." && pwd)"
ACTUAL_FA3_REVISION="$(git -C "${FA3_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[ "${ACTUAL_FA3_REVISION}" = "${FA3_REVISION}" ] || {
    echo "ERROR: FlashAttention revision is ${ACTUAL_FA3_REVISION:-unknown}; expected ${FA3_REVISION}" >&2
    exit 1
}
ACTUAL_CUTLASS_REVISION="$(git -C "${CUTLASS_DIR}" rev-parse HEAD 2>/dev/null || true)"
[ "${ACTUAL_CUTLASS_REVISION}" = "${CUTLASS_REVISION}" ] || {
    echo "ERROR: CUTLASS revision is ${ACTUAL_CUTLASS_REVISION:-unknown}; expected ${CUTLASS_REVISION}" >&2
    exit 1
}
grep -Fq 'using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;' \
    "${FA3_DIR}/flash_fwd_launch_template.h" || {
    echo "ERROR: pinned FA3 FP8 output type contract is no longer BF16" >&2
    exit 1
}

mkdir -p "${OUT_DIR}" "${OBJ_DIR}"

NVCC_FLAGS=(
    -std=c++17
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -fPIC
    -O3
    -DNDEBUG
    -I"${CUTLASS_DIR}/include"
    -I"${FA3_DIR}"
    -I"${SCRIPT_DIR}"
    -lineinfo
    -DRVLLM_FA3_ABI_VERSION=2
)

echo "=== Building FA3 SM90 (head_dim=128,256) — ${JOBS} parallel jobs ==="

# Translation units. The 8 instantiation .cu must match, one-for-one, the
# run_mha_fwd_<90,{half_t|e4m3_t},{128|256},..,{Split},PagedKV,..> templates
# declared in fa3_sm90_wrapper.cu (paged, PackGQA=true, fp16 + e4m3, split +
# non-split). fa3_combine_hdim256.cu supplies the half/BF16 kBlockK=256
# combine symbols that upstream flash_fwd_combine.cu omits (it ships only 64/128).
SRCS=(
    "${SCRIPT_DIR}/fa3_sm90_wrapper.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_e4m3_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_e4m3_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_fp16_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_fp16_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_e4m3_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_e4m3_paged_split_sm90.cu"
    "${FA3_DIR}/flash_fwd_combine.cu"
    "${SCRIPT_DIR}/fa3_combine_hdim256.cu"
    "${FA3_DIR}/flash_prepare_scheduler.cu"
)

# Verify every source exists before compiling — a missing instantiation must
# abort here, not surface as a confusing link-time undefined symbol.
MISSING=0
for src in "${SRCS[@]}"; do
    [ -f "${src}" ] || { echo "ERROR: missing source ${src}"; MISSING=1; }
done
[ "${MISSING}" -eq 0 ] || { echo "ERROR: required FA3 sources missing — aborting"; exit 1; }

# Record the FA source revision so the .so provenance is reconstructible.
echo "FA3 source: Dao-AILab/flash-attention @ ${ACTUAL_FA3_REVISION} (BSD-3-Clause)"
echo "CUTLASS:    NVIDIA/cutlass @ ${ACTUAL_CUTLASS_REVISION} (BSD-3-Clause)"

# Phase 1: compile each .cu to .o in parallel. Track pid->source so a failure
# names the offending translation unit and its captured log.
echo "Compiling ${#SRCS[@]} translation units (${JOBS} parallel)..."
OBJS=()
PIDS=()
declare -A PID_SRC=()
declare -A PID_LOG=()
FAIL=0
T0=$(date +%s)

wait_batch() {
    local pid
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            FAIL=1
            echo "ERROR: nvcc failed on ${PID_SRC[$pid]}:" >&2
            sed 's/^/    /' "${PID_LOG[$pid]}" >&2 || true
        fi
    done
    PIDS=()
}

for src in "${SRCS[@]}"; do
    base=$(basename "${src}" .cu)
    obj="${OBJ_DIR}/${base}.o"
    log="${OBJ_DIR}/${base}.log"
    OBJS+=("${obj}")
    nvcc "${NVCC_FLAGS[@]}" -c "${src}" -o "${obj}" >"${log}" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    PID_SRC[$pid]="${base}"
    PID_LOG[$pid]="${log}"
    if [ "${#PIDS[@]}" -ge "${JOBS}" ]; then
        wait_batch
    fi
done

# Wait for the final partial batch.
wait_batch
if [ "${FAIL}" -ne 0 ]; then
    echo "ERROR: one or more nvcc compilations failed"
    exit 1
fi

T1=$(date +%s)
echo "Compilation: $((T1 - T0))s"

# Phase 2: link .o files into shared library
echo "Linking..."
nvcc -shared -arch=sm_90a -o "${OUT_DIR}/libfa3_kernels.so" "${OBJS[@]}"
T2=$(date +%s)
echo "Link: $((T2 - T1))s"

SZ=$(stat -c%s "${OUT_DIR}/libfa3_kernels.so" 2>/dev/null || stat -f%z "${OUT_DIR}/libfa3_kernels.so")
echo ""
echo "=== FA3 build complete ($((T2 - T0))s total) ==="
echo "  Size: ${SZ} bytes"
echo "  Path: ${OUT_DIR}/libfa3_kernels.so"

if [ "${SZ}" -lt 1000000 ]; then
    echo "ERROR: .so is too small (<1MB) to be a complete FA3 build" >&2
    exit 1
fi

# Verify the three extern "C" entry points the Rust loader resolves
# (rvllm-attention/src/lib.rs) are actually exported.
echo ""
echo "Exported entry points:"
ALL_EXPORTS=$(nm -D --defined-only "${OUT_DIR}/libfa3_kernels.so" | awk '{print $3}')
EXPORTS=$(printf '%s\n' "${ALL_EXPORTS}" | grep -E '^(rvllm_fa3_|fa3_sm90_)' || true)
if [ -z "${EXPORTS}" ]; then
    echo "  (none found — check build)"
    exit 1
fi
while IFS= read -r line; do printf '  %s\n' "${line}"; done <<<"${EXPORTS}"
for sym in rvllm_fa3_abi_version rvllm_fa3_upstream_revision fa3_sm90_fp8_output_dtype fa3_sm90_fp8_output_element_size fa3_sm90_decode_workspace_size fa3_sm90_prefill_workspace_size fa3_sm90_paged_decode fa3_sm90_paged_decode_fp8 fa3_sm90_paged_prefill_fp8; do
    grep -Fxq "${sym}" <<< "${ALL_EXPORTS}" || { echo "ERROR: missing exported symbol ${sym}"; exit 1; }
done

rm -rf "${OBJ_DIR}"
REVISION="${REVISION:-$(git -C "${SCRIPT_DIR}/.." rev-parse HEAD 2>/dev/null || true)}"
"${SCRIPT_DIR}/gen_manifest.sh" "${OUT_DIR}" "${REVISION}" cuda-ptx-v1 libfa3_kernels.so
