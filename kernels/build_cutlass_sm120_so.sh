#!/bin/bash
# Compile sm_120 CUTLASS FP8 blockwise GEMM into a shared library for
# FFI from Rust on Blackwell-Geforce / DGX Spark (sm_121 = family-compat
# with sm_120).
#
# Produces: kernels/sm_121/libcutlass_sm120.so
#
# This is a SEPARATE .so from `build_cutlass_so.sh` (SM90) — the
# SM120 build uses different CUTLASS schedules + arch targets, and
# we keep it split so architecture-specific ABIs remain isolated.
#
# Requires: CUTLASS at $CUTLASS_DIR (defaults to the vendored
# submodule at <repo>/cutlass — initialise with
# `git submodule update --init cutlass`).
#
# Usage:
#   ./kernels/build_cutlass_sm120_so.sh           # defaults below
#   ./kernels/build_cutlass_sm120_so.sh sm_121a   # override arch
#   CUTLASS_DIR=/path/to/cutlass ./kernels/build_cutlass_sm120_so.sh

set -euo pipefail

ARCH=${1:-sm_121a}
[ "$ARCH" = "sm_121a" ] || {
    echo "unsupported architecture: $ARCH" >&2
    exit 1
}
OUT_SUBDIR=${ARCH%a}   # sm_120a -> sm_120 for the per-arch kernel dir

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
CUTLASS_DIR=${CUTLASS_DIR:-$REPO/cutlass}
CUTLASS_REVISION="${CUTLASS_REVISION:-da5e086dab31d63815acafdac9a9c5893b1c69e2}"

if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "CUTLASS not found at $CUTLASS_DIR"
    echo "  (submodule init:  git submodule update --init cutlass)"
    exit 1
fi
ACTUAL_CUTLASS_REVISION="$(git -C "$CUTLASS_DIR" rev-parse HEAD 2>/dev/null || true)"
[ "$ACTUAL_CUTLASS_REVISION" = "$CUTLASS_REVISION" ] || {
    echo "CUTLASS revision is ${ACTUAL_CUTLASS_REVISION:-unknown}; expected $CUTLASS_REVISION" >&2
    exit 1
}

cd "$DIR"
mkdir -p "$OUT_SUBDIR"
OBJ_DIR="$OUT_SUBDIR/obj"
mkdir -p "$OBJ_DIR"
LOG_DIR="$OBJ_DIR/logs"
mkdir -p "$LOG_DIR"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null 2>&1 || { echo "nvcc not found: $NVCC" >&2; exit 1; }
NVCC_FLAGS=(
    -std=c++17 "-arch=$ARCH" --expt-relaxed-constexpr -O3 --use_fast_math
    "-I$CUTLASS_DIR/include" "-I$CUTLASS_DIR/tools/util/include"
    --compiler-options -fPIC
)
if [ -n "${EXTRA_NVCC_FLAGS:-}" ]; then
    read -r -a EXTRA_FLAGS <<< "$EXTRA_NVCC_FLAGS"
    NVCC_FLAGS+=("${EXTRA_FLAGS[@]}")
fi

echo "Building CUTLASS sm_120 shared library ($ARCH)..."

OBJS=()

# Sources that build the Blackwell-Geforce FP8 GEMM .so. Start with
# one — add more (autotune variants, nvfp4, etc.) here as they land.
SOURCES=(
    cutlass_fp8_gemm_blockscale_sm120.cu
)

for f in "${SOURCES[@]}"; do
    [ -f "$f" ] || { echo "missing required source: $f" >&2; exit 1; }
    stem=${f%.cu}
    obj="$OBJ_DIR/${stem}.o"
    log="$LOG_DIR/${stem}.log"
    echo -n "  $f -> ${stem}.o ... "
    if "$NVCC" -c "${NVCC_FLAGS[@]}" -o "$obj" "$f" 2>"$log"; then
        echo "ok"
        OBJS+=("$obj")
    else
        echo "FAILED  (see $DIR/$log)" >&2
        tail -10 "$log" >&2 || true
        exit 1
    fi
done

if [ "${#OBJS[@]}" -eq 0 ]; then
    echo "no objects compiled, cannot link"
    exit 1
fi

SO_PATH="$OUT_SUBDIR/libcutlass_sm120.so"
LINK_LOG="$LOG_DIR/link.log"
echo -n "  linking $SO_PATH ... "
if "$NVCC" -shared -o "$SO_PATH" "${OBJS[@]}" -lcudart 2>"$LINK_LOG"; then
    echo "ok"
else
    echo "FAILED" >&2
    tail -10 "$LINK_LOG" >&2 || true
    exit 1
fi
[ -s "$SO_PATH" ] || { echo "linked library is empty: $SO_PATH" >&2; exit 1; }
command -v nm >/dev/null 2>&1 || { echo "nm is required to verify the ABI" >&2; exit 1; }
SYMBOLS=$(nm -D --defined-only "$SO_PATH" | awk '{print $3}')
for symbol in \
    cutlass_fp8_gemm_blockscale_sm120 \
    cutlass_fp8_gemm_blockscale_sm120_prep_sfa \
    cutlass_fp8_gemm_blockscale_sm120_prep_sfb \
    cutlass_fp8_gemm_blockscale_sm120_workspace \
    cutlass_fp8_gemm_blockscale_sm120_sfa_bytes \
    cutlass_fp8_gemm_blockscale_sm120_sfb_bytes; do
    grep -Fxq "$symbol" <<< "$SYMBOLS" || { echo "missing exported symbol: $symbol" >&2; exit 1; }
done

echo ""
echo "CUTLASS sm_120 library: $DIR/$SO_PATH"
echo "Compiled ${#OBJS[@]} sources"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)}"
"$DIR/gen_manifest.sh" "$DIR/$OUT_SUBDIR" "$REVISION" cuda-ptx-v1 libcutlass_sm120.so
