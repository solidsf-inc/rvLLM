#!/bin/bash
# Compile CUTLASS kernels for rvLLM.
# Requires the repository-pinned CUTLASS submodule by default.
# Usage: ./kernels/build_cutlass.sh [arch] [cutlass_dir]

set -euo pipefail

ARCH=${1:-sm_90}
DIR="$(cd "$(dirname "$0")" && pwd)"
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

NVCC=${NVCC:-nvcc}
OK=0
FAIL=0
TOTAL=0

for f in cutlass_*.cu; do
    [ -f "$f" ] || continue
    TOTAL=$((TOTAL + 1))
    stem=${f%.cu}
    echo -n "  $f -> $ARCH/${stem}.ptx ... "
    if "$NVCC" --ptx -arch="$ARCH" -O3 --use_fast_math \
        -I"$CUTLASS_DIR/include" \
        -I"$CUTLASS_DIR/tools/util/include" \
        -o "$ARCH/${stem}.ptx" "$f" >"$ARCH/${stem}.log" 2>&1; then
        echo "ok"
        OK=$((OK + 1))
    else
        echo "FAILED"
        FAIL=$((FAIL + 1))
        cat "$ARCH/${stem}.log" >&2
    fi
done

echo ""
echo "CUTLASS kernels: ${OK}/${TOTAL} compiled (${FAIL} failed)"
echo "PTX output: $DIR/$ARCH/cutlass_*.ptx"
[ "$TOTAL" -gt 0 ] && [ "$FAIL" -eq 0 ] || exit 1
