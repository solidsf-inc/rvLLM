#!/bin/bash
# Compile CUDA kernels to PTX for runtime loading via cuModuleLoadData.
#
# Usage: ./build.sh [arch]
#   ./build.sh              # compile for all supported architectures
#   ./build.sh sm_80        # compile for A100 only
#   CUDA_ARCH=sm_90 ./build.sh  # compile for H100 only
#
# Environment variables:
#   NVCC       - path to nvcc (default: nvcc)
#   CUDA_ARCH  - target architecture (overrides default multi-arch build)

set -euo pipefail

NVCC=${NVCC:-nvcc}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
command -v "$NVCC" >/dev/null 2>&1 || { echo "ERROR: nvcc not found: $NVCC" >&2; exit 1; }

# Supported compute capabilities
# sm_80  = A100, A30
# sm_89  = RTX 4090, L40S
# sm_90  = H100, H200
# sm_100 = B100, B200
# sm_121 = GB10 (Project DIGITS / DGX Spark — Grace+Blackwell consumer)
ALL_ARCHS="sm_80 sm_89 sm_90"

# Check nvcc version for sm_100 support (CUDA 12.8+)
NVCC_VERSION=$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
NVCC_VERSION=${NVCC_VERSION:-0.0}
NVCC_MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
NVCC_MINOR=$(echo "$NVCC_VERSION" | cut -d. -f2)
if [ "$NVCC_MAJOR" -ge 13 ] || { [ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -ge 8 ]; }; then
    ALL_ARCHS="$ALL_ARCHS sm_100"
fi

# Check nvcc version for sm_121 support (CUDA 13.0+)
if [ "$NVCC_MAJOR" -ge 13 ]; then
    ALL_ARCHS="$ALL_ARCHS sm_121"
fi

# Use specific arch if provided
if [ -n "${1:-}" ]; then
    ARCHS="$1"
elif [ -n "${CUDA_ARCH:-}" ]; then
    ARCHS="$CUDA_ARCH"
else
    ARCHS="$ALL_ARCHS"
fi

echo "NVCC: $("$NVCC" --version | tail -1)"
echo "Target architectures: $ARCHS"
echo ""

compile_kernel() {
    local cu="$1" arch="$2" ptx="$3"
    local base
    base=$(basename "$cu" .cu)
    # FP8 / NVFP4 tensor-core MMA kernels need the arch-specific
    # feature set (`sm_121a` / `sm_120a`) because `.kind::f8f6f4`
    # and `mma.kind::*f8*` live in the CUDA family-specific PTX
    # feature set. Plain `sm_121` rejects them at ptxas time even
    # though nvcc -ptx emits the instruction successfully.
    if grep -q 'kind::f8f6f4\|mma.sync.*e4m3\|mma.sync.*e2m1\|fp8_mma_frag_pack\|mma_m16n8k32' "$cu" 2>/dev/null; then
        case "$arch" in
            sm_100) arch="sm_100a" ;;
            sm_120) arch="sm_120a" ;;
            sm_121) arch="sm_121a" ;;
            sm_122) arch="sm_122a" ;;
        esac
    fi
    "$NVCC" -ptx -arch="$arch" -O3 -lineinfo -o "$ptx" "$cu"
}

# Revision pinned into the generated manifest.json. Precedence:
#   1. $REVISION env var (tarball / CI shallow-clone builds where
#      git isn't available or history is detached)
#   2. git HEAD
REVISION="${REVISION:-$(git -C "$DIR" rev-parse HEAD 2>/dev/null || true)}"
[ -n "$REVISION" ] || {
    echo "ERROR: set REVISION when building outside a committed checkout" >&2
    exit 1
}
[[ "$REVISION" =~ ^[0-9a-f]{7,64}$ ]] || {
    echo "ERROR: REVISION must be a lowercase 7-64 character hexadecimal commit ID" >&2
    exit 1
}
KERNEL_ABI="${KERNEL_ABI:-cuda-ptx-v1}"
[ "$KERNEL_ABI" = "cuda-ptx-v1" ] || {
    echo "ERROR: KERNEL_ABI must be cuda-ptx-v1" >&2
    exit 1
}

V3_KERNELS_DIR="$DIR/../v3/kernels"
[ -d "$V3_KERNELS_DIR" ] || { echo "ERROR: required kernel tree is missing: $V3_KERNELS_DIR" >&2; exit 1; }

# Only runtime-loadable PTX sources belong here. CUTLASS and FA3 host wrappers
# have dedicated shared-library builds; probes and standalone benchmarks are
# deliberately excluded from release manifests.
TOP_LEVEL_SOURCES=(
    add_bias_f16.cu
    argmax.cu
    cast_fp.cu
    embedding_gather_f16.cu
    flash_attention.cu
    fp8_gemv.cu
    fused_rmsnorm_fp8_quant.cu
    fused_silu_fp8_quant.cu
)
V3_SOURCES=(
    bf16_to_f16_sat.cu compute_qkv_scales.cu f32_to_bf16.cu f32_to_f16_sat.cu
    fp8_channelscale_gemv_ktiled.cu fp8_channelscale_gemv_splitk.cu fp8_decode_v2.cu
    fp8_e4m3_gemv.cu fused_gelu_mul_f16.cu fused_gelu_mul_fp8_quant.cu
    fused_norm_add_residual.cu fused_norm_add_residual_f16.cu fused_qk_rmsnorm.cu
    fused_qkv_rmsnorm.cu fused_rope_cache_fp8kv.cu fused_rope_partial_f16kv.cu
    fused_rope_partial_fp8kv.cu gemma4_ple_gate.cu lmhead_prune_argmax.cu
    logit_softcap.cu map_token_id.cu paged_attention_sm89.cu ple_gelu_mul_f16.cu
    ple_project_combine.cu residual_scale_bf16s_f16.cu residual_scale_f16.cu
    rmsnorm_inplace_bf16.cu rmsnorm_inplace_f16.cu sample_topk_f32.cu
    scale_cols_f16.cu scale_cols_f32.cu scale_rows_f16_pertoken.cu
    scale_rows_f32_ratio.cu vector_add_bf16_to_f16.cu vector_add_f16.cu vnorm_f16.cu
)

declare -A OUTPUT_OWNER=()
for source in "${TOP_LEVEL_SOURCES[@]}"; do
    [ -f "$DIR/$source" ] || { echo "ERROR: required source missing: kernels/$source" >&2; exit 1; }
    OUTPUT_OWNER["${source%.cu}"]="kernels/$source"
done
for source in "${V3_SOURCES[@]}"; do
    [ -f "$V3_KERNELS_DIR/$source" ] || { echo "ERROR: required source missing: v3/kernels/$source" >&2; exit 1; }
    stem="${source%.cu}"
    if [ -n "${OUTPUT_OWNER[$stem]:-}" ]; then
        echo "ERROR: output collision for ${stem}.ptx: ${OUTPUT_OWNER[$stem]} and v3/kernels/$source" >&2
        exit 1
    fi
    OUTPUT_OWNER[$stem]="v3/kernels/$source"
done

for arch in $ARCHS; do
    case "$arch" in
        sm_80|sm_89|sm_90|sm_100|sm_121) ;;
        *) echo "ERROR: unsupported architecture: $arch" >&2; exit 1 ;;
    esac
    echo "=== Compiling for $arch ==="
    OUTDIR="$DIR/$arch"
    mkdir -p "$OUTDIR"
    rm -f "$OUTDIR"/*.ptx "$OUTDIR/manifest.json.tmp"
    ARTIFACTS=()

    for source in "${TOP_LEVEL_SOURCES[@]}"; do
        cu="$DIR/$source"
        base=$(basename "$cu" .cu)
        ptx="$OUTDIR/${base}.ptx"
        echo "  $base.cu -> $arch/$base.ptx"
        compile_kernel "$cu" "$arch" "$ptx"
        ARTIFACTS+=("${base}.ptx")
    done

    for source in "${V3_SOURCES[@]}"; do
        cu="$V3_KERNELS_DIR/$source"
        base=$(basename "$cu" .cu)
        ptx="$OUTDIR/${base}.ptx"
        echo "  (v3) $base.cu -> $arch/$base.ptx"
        compile_kernel "$cu" "$arch" "$ptx"
        ARTIFACTS+=("${base}.ptx")
    done

    "$DIR/gen_manifest.sh" "$OUTDIR" "$REVISION" "$KERNEL_ABI" "${ARTIFACTS[@]}"
done

echo ""
echo "Done. PTX files in: $DIR/<arch>/*.ptx"
