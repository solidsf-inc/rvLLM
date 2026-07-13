#!/usr/bin/env bash
# Compile every CUDA artifact shipped for one release architecture.
# This is a compile/link gate only; it does not require a visible GPU.
set -euo pipefail

TARGET="${1:?usage: $0 <sm90|sm121> [flash-attention-hopper-dir]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$TARGET" in
    sm90)
        FA3_DIR="${2:?sm90 requires the pinned flash-attention hopper directory}"
        ./kernels/build.sh sm_90
        ./kernels/build_fa_sm89_so.sh sm_90a
        ./kernels/build_cutlass_so.sh sm_90
        ./kernels/build_w4a8.sh
        FA3_DIR="$FA3_DIR" ./kernels/build_fa3.sh
        ;;
    sm121)
        ./kernels/build.sh sm_121
        ./kernels/build_cutlass_sm120_so.sh sm_121a
        ;;
    *)
        echo "unsupported release target: $TARGET" >&2
        exit 1
        ;;
esac
