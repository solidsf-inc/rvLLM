#!/bin/bash
# Compile CUTLASS kernels into a shared library (.so) for FFI from Rust.
# CUTLASS kernels are self-launching -- they compute their own grid dims internally.
# You can't launch them as raw PTX via cuLaunchKernel. They need their C++ wrapper
# function called from the host, so we compile to .so and dlopen from Rust.
#
# Requires the repository-pinned CUTLASS submodule by default.
# Usage: ./kernels/build_cutlass_so.sh [arch] [cutlass_dir]

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH=${1:-sm_90}
CUTLASS_DIR=${2:-${CUTLASS_DIR:-"$DIR/../cutlass"}}
CUTLASS_REVISION="${CUTLASS_REVISION:-da5e086dab31d63815acafdac9a9c5893b1c69e2}"
[ -d "$CUTLASS_DIR/include/cutlass" ] || {
    echo "ERROR: pinned CUTLASS submodule is missing at $CUTLASS_DIR" >&2
    exit 1
}
ACTUAL_CUTLASS_REVISION="$(git -C "$CUTLASS_DIR" rev-parse HEAD 2>/dev/null || true)"
[ "$ACTUAL_CUTLASS_REVISION" = "$CUTLASS_REVISION" ] || {
    echo "ERROR: CUTLASS revision is ${ACTUAL_CUTLASS_REVISION:-unknown}; expected $CUTLASS_REVISION" >&2
    exit 1
}

cd "$DIR"
mkdir -p "$ARCH"
OBJ_DIR="$ARCH/obj"
mkdir -p "$OBJ_DIR"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null 2>&1 || { echo "ERROR: nvcc not found: $NVCC" >&2; exit 1; }
[ "$ARCH" = "sm_90" ] || {
    echo "ERROR: libcutlass_kernels.so contains Hopper SM90 schedules; expected arch sm_90" >&2
    exit 1
}
NVCC_ARCH="sm_90a"
NVCC_FLAGS=(
    -std=c++17 "-arch=${NVCC_ARCH}" --expt-relaxed-constexpr -O3 --use_fast_math
    "-I${CUTLASS_DIR}/include"
    "-I${CUTLASS_DIR}/tools/util/include"
    "-I${CUTLASS_DIR}/examples/45_dual_gemm"
    --compiler-options -fPIC
)

GATE_DEFINES=""
for var in \
    RVLLM_CUTLASS_GATE_TILE_M \
    RVLLM_CUTLASS_GATE_TILE_N \
    RVLLM_CUTLASS_GATE_TILE_K \
    RVLLM_CUTLASS_GATE_CLUSTER_M \
    RVLLM_CUTLASS_GATE_CLUSTER_N \
    RVLLM_CUTLASS_GATE_CLUSTER_K \
    RVLLM_CUTLASS_GATE_SCHEDULE; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        GATE_DEFINES="$GATE_DEFINES -D${var}=${val}"
    fi
done

GATE_SOURCE_GEN=${RVLLM_CUTLASS_GATE_SOURCE_GEN:-0}

generate_gate_source() {
    local src="$1"
    local out="$2"
    cp "$src" "$out"
    perl -0pi -e 's@\n#ifndef RVLLM_CUTLASS_GATE_TILE_M.*?#define RVLLM_CUTE_INT\(x\) RVLLM_CUTE_INT_\(x\)\n@@s' "$out"
    perl -0pi -e "s@using GateTileShape = Shape<.*?>;@using GateTileShape = Shape<_${RVLLM_CUTLASS_GATE_TILE_M}, _${RVLLM_CUTLASS_GATE_TILE_N}, _${RVLLM_CUTLASS_GATE_TILE_K}>;@s" "$out"
    perl -0pi -e "s@using GateClusterShape = Shape<.*?>;@using GateClusterShape = Shape<_${RVLLM_CUTLASS_GATE_CLUSTER_M}, _${RVLLM_CUTLASS_GATE_CLUSTER_N}, _${RVLLM_CUTLASS_GATE_CLUSTER_K}>;@s" "$out"
    if [ "${RVLLM_CUTLASS_GATE_SCHEDULE:-0}" = "1" ]; then
        perl -0pi -e 's@#if RVLLM_CUTLASS_GATE_SCHEDULE == 1\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;\n#else\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;\n#endif@using GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;@s' "$out"
    else
        perl -0pi -e 's@#if RVLLM_CUTLASS_GATE_SCHEDULE == 1\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;\n#else\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;\n#endif@using GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;@s' "$out"
    fi
}

echo "Building CUTLASS shared library for $ARCH..."
if [ -n "$GATE_DEFINES" ]; then
    echo "  cutlass_gateup_silu.cu defines:$GATE_DEFINES"
fi

# Compile each .cu to an object file separately to avoid template conflicts.
OK=0
OBJS=()

SOURCES=(
    cutlass_qkv_bias.cu
    cutlass_oproj_residual.cu
    cutlass_gateup_silu.cu
    cutlass_gemm.cu
    cutlass_fp8_gemm.cu
    cutlass_hgemm_autotune.cu
    cutlass_oproj_residual_autotune.cu
    cutlass_gateup_silu_autotune.cu
    cutlass_fp8_gemm_autotune.cu
    cutlass_fp8_gemm_residual.cu
    cutlass_fp8_gemm_channelscale.cu
)

for f in "${SOURCES[@]}"; do
    [ -f "$f" ] || { echo "ERROR: required source is missing: $f" >&2; exit 1; }
    stem=${f%.cu}
    obj="$OBJ_DIR/${stem}.o"
    EXTRA_FLAGS=()
    SOURCE_FILE="$f"
    if [ "$f" = "cutlass_gateup_silu.cu" ]; then
        read -r -a EXTRA_FLAGS <<<"$GATE_DEFINES"
        if [ "$GATE_SOURCE_GEN" = "1" ]; then
            SOURCE_FILE="$ARCH/generated_${stem}.cu"
            generate_gate_source "$f" "$SOURCE_FILE"
            EXTRA_FLAGS=()
        fi
    fi
    echo -n "  $f -> ${stem}.o ... "
    if "$NVCC" -c "${NVCC_FLAGS[@]}" "${EXTRA_FLAGS[@]}" -o "$obj" "$SOURCE_FILE" >"$OBJ_DIR/${stem}.log" 2>&1; then
        echo "ok"
        OBJS+=("$obj")
        OK=$((OK + 1))
    else
        echo "FAILED" >&2
        cat "$OBJ_DIR/${stem}.log" >&2
        exit 1
    fi
done

if [ "${#OBJS[@]}" -ne "${#SOURCES[@]}" ]; then
    echo "No objects compiled, cannot link."
    exit 1
fi

# Link into shared library
echo -n "  Linking libcutlass_kernels.so ... "
LIB_TMP="$ARCH/libcutlass_kernels.so.tmp"
if "$NVCC" -shared -o "$LIB_TMP" "${OBJS[@]}" -lcudart >"$OBJ_DIR/link.log" 2>&1; then
    echo "ok"
else
    echo "FAILED"
    cat "$OBJ_DIR/link.log" >&2
    exit 1
fi

REQUIRED_SYMBOLS=(
    cutlass_qkv_bias_gemm
    cutlass_oproj_residual_gemm
    cutlass_gateup_silu
    cutlass_hgemm
    cutlass_fp8_gemm
    cutlass_fp8_gemm_channelscale
    cutlass_hgemm_v0
    cutlass_oproj_residual_v0
    cutlass_gateup_silu_v0
    cutlass_fp8_gemm_v0
    cutlass_fp8_gemm_residual_v0
)
ALL_EXPORTS=$(nm -D --defined-only "$LIB_TMP" | awk '{print $3}')
for sym in "${REQUIRED_SYMBOLS[@]}"; do
    grep -Fxq "$sym" <<< "$ALL_EXPORTS" || {
        echo "ERROR: missing exported symbol $sym" >&2
        exit 1
    }
done
mv "$LIB_TMP" "$ARCH/libcutlass_kernels.so"

echo ""
echo "CUTLASS shared library: $DIR/$ARCH/libcutlass_kernels.so"
echo "Compiled $OK kernel files"
REVISION="${REVISION:-$(git -C "$DIR/.." rev-parse HEAD 2>/dev/null || true)}"
"$DIR/gen_manifest.sh" "$DIR/$ARCH" "$REVISION" cuda-ptx-v1 libcutlass_kernels.so
