#!/usr/bin/env bash
# One-shot: tarball the tpu/ suite, scp to a TPU VM, install jax[tpu], run.
# Usage: ./deploy_to_tpu.sh <tpu-name> <zone> [--only <kernel>]
set -euo pipefail

NAME="${1:?tpu name}"
ZONE="${2:?zone}"
shift 2
PROJECT="${PROJECT:?set PROJECT to the target GCP project}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REVISION="${REVISION:-$(git -C "$ROOT/.." rev-parse HEAD 2>/dev/null || true)}"
[ -n "$REVISION" ] || { echo "ERROR: set REVISION outside a committed checkout" >&2; exit 1; }
RUN_ID="${RUN_ID:-${REVISION:0:12}}"
JAX_VERSION="${JAX_VERSION:-0.4.38}"
NUMPY_VERSION="${NUMPY_VERSION:-2.1.3}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid RUN_ID" >&2; exit 1; }
[[ "$JAX_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "ERROR: invalid JAX_VERSION" >&2; exit 1; }
[[ "$NUMPY_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "ERROR: invalid NUMPY_VERSION" >&2; exit 1; }

TAR="${TMPDIR:-/tmp}/rvllm-tpu-${RUN_ID}.tar.gz"
CHECKSUM="${TAR}.sha256"
echo ">> packaging $ROOT -> $TAR"
tar -czf "$TAR" -C "$(dirname "$ROOT")" \
    --exclude='tpu/.venv' --exclude='tpu/out' --exclude='tpu/__pycache__' \
    tpu
shasum -a 256 "$TAR" | awk '{print $1}' >"$CHECKSUM"

echo ">> uploading to $NAME"
gcloud compute tpus tpu-vm scp --zone="$ZONE" --project="$PROJECT" \
    "$TAR" "$CHECKSUM" "$NAME:/tmp/"

EXTRA_ARGS=$(printf ' %q' "$@")
REMOTE_CMD=$(cat <<EOF
set -euo pipefail
run_dir=\${HOME}/rvllm-runs/${RUN_ID}
archive=/tmp/$(basename "$TAR")
checksum=/tmp/$(basename "$CHECKSUM")
expected=\$(tr -d '[:space:]' <"\${checksum}")
actual=\$(shasum -a 256 "\${archive}" | awk '{print \$1}')
[ "\${actual}" = "\${expected}" ] || { echo "ERROR: package checksum mismatch" >&2; exit 1; }
if [ -e "\${run_dir}" ]; then
    echo "ERROR: run directory already exists: \${run_dir}" >&2
    exit 1
fi
mkdir -p "\${run_dir}"
tar -xzf "\${archive}" --no-same-owner -C "\${run_dir}"
cd "\${run_dir}"
cd tpu
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --quiet "aiohttp==3.14.1" "jax[tpu]==${JAX_VERSION}" "numpy==${NUMPY_VERSION}"
python3 -c "import jax; print('backend:', jax.default_backend(), 'devices:', jax.devices())"
python3 -m harness.execute_all${EXTRA_ARGS}
EOF
)

echo ">> executing on TPU VM"
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --project="$PROJECT" \
    --command="$REMOTE_CMD"
