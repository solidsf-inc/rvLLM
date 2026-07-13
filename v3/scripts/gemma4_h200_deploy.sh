#!/usr/bin/env bash
set -euo pipefail

# Deploy rvLLM to an existing H200 instance using preinstalled, pinned
# toolchains and source checkouts. No curl-pipe installers or model downloads.

INSTANCE_ID="${1:?usage: $0 <vast_instance_id>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HEAD_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
REVISION="${REVISION:-$HEAD_REVISION}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: REVISION must be a full commit" >&2; exit 1; }
[ "$HEAD_REVISION" = "$REVISION" ] || { echo "ERROR: REVISION must match HEAD" >&2; exit 1; }
[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ] || {
    echo "ERROR: refusing to deploy a dirty working tree" >&2; exit 1; }

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the authorized remote model directory}"
MODEL_REVISION="${MODEL_REVISION:?set MODEL_REVISION to the exact model revision}"
FA3_DIR="${FA3_DIR:?set FA3_DIR to the pinned remote FlashAttention hopper directory}"
CUTLASS_DIR="${CUTLASS_DIR:?set CUTLASS_DIR to the pinned remote CUTLASS checkout}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/rvllm-runs}"
SSH_KNOWN_HOSTS="${SSH_KNOWN_HOSTS:-${HOME}/.ssh/known_hosts}"
[ -f "$SSH_KNOWN_HOSTS" ] || { echo "ERROR: known-hosts file missing: $SSH_KNOWN_HOSTS" >&2; exit 1; }

SSH_URL=""
for _ in $(seq 1 60); do
    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    [ -n "$SSH_URL" ] && break
    sleep 10
done
[ -n "$SSH_URL" ] || { echo "ERROR: instance SSH endpoint did not become ready" >&2; exit 1; }
AUTHORITY="${SSH_URL#ssh://}"
SSH_PORT="${AUTHORITY##*:}"
SSH_TARGET="${AUTHORITY%:*}"
SSH_HOST="${SSH_TARGET#*@}"
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: could not parse SSH port" >&2; exit 1; }
ssh-keygen -F "[${SSH_HOST}]:${SSH_PORT}" -f "$SSH_KNOWN_HOSTS" >/dev/null || {
    echo "ERROR: verified host key for [${SSH_HOST}]:${SSH_PORT} is absent" >&2; exit 1; }
SSH=(ssh -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS}" -p "$SSH_PORT" "$SSH_TARGET")
SCP=(scp -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS}" -P "$SSH_PORT")

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-h200-deploy.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
TARBALL="${WORK_DIR}/rvllm-${REVISION}.tar.gz"
(cd "$REPO_ROOT" && git archive --format=tar "$REVISION" kernels v3 | gzip -n >"$TARBALL")
(cd "$WORK_DIR" && shasum -a 256 "$(basename "$TARBALL")" >"$(basename "$TARBALL").sha256")
"${SCP[@]}" "$TARBALL" "${TARBALL}.sha256" "${SSH_TARGET}:/tmp/"

REMOTE_COMMAND=$(printf 'REVISION=%q MODEL_DIR=%q MODEL_REVISION=%q FA3_DIR=%q CUTLASS_DIR=%q REMOTE_ROOT=%q bash -s' \
    "$REVISION" "$MODEL_DIR" "$MODEL_REVISION" "$FA3_DIR" "$CUTLASS_DIR" "$REMOTE_ROOT")
"${SSH[@]}" "$REMOTE_COMMAND" <<'REMOTE_SCRIPT'
set -euo pipefail
command -v cargo >/dev/null
command -v nvcc >/dev/null
command -v nvidia-smi >/dev/null

archive="/tmp/rvllm-${REVISION}.tar.gz"
(cd /tmp && sha256sum -c "$(basename "$archive").sha256")
archive_sha=$(awk '{print $1}' "${archive}.sha256")
run_dir="${REMOTE_ROOT}/${REVISION}"
source_archive="${run_dir}/source.tar.gz"
mkdir -p "$REMOTE_ROOT"
if [ -e "$run_dir" ]; then
    [ "$(cat "$run_dir/REVISION" 2>/dev/null)" = "$REVISION" ] || {
        echo "ERROR: existing run directory has a different revision" >&2; exit 1; }
    [ "$(cat "$run_dir/SOURCE_ARCHIVE_SHA256" 2>/dev/null)" = "$archive_sha" ] || {
        echo "ERROR: existing run directory has a different source digest" >&2; exit 1; }
    [ "$(sha256sum "$source_archive" | awk '{print $1}')" = "$archive_sha" ] || {
        echo "ERROR: stored source archive failed verification" >&2; exit 1; }
else
    stage_dir=$(mktemp -d "${REMOTE_ROOT}/.${REVISION}.XXXXXX")
    install -m 0444 "$archive" "$stage_dir/source.tar.gz"
    printf '%s\n' "$REVISION" >"$stage_dir/REVISION"
    printf '%s\n' "$archive_sha" >"$stage_dir/SOURCE_ARCHIVE_SHA256"
    mv "$stage_dir" "$run_dir"
fi
work_dir="${run_dir}/work"
rm -rf "$work_dir"
mkdir "$work_dir"
tar -xzf "$source_archive" --no-same-owner -C "$work_dir"
[ -d "$MODEL_DIR" ] || { echo "ERROR: model directory missing: $MODEL_DIR" >&2; exit 1; }
[ "$(cat "$MODEL_DIR/REVISION" 2>/dev/null)" = "$MODEL_REVISION" ] || {
    echo "ERROR: model REVISION does not match MODEL_REVISION" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

export REVISION FA3_DIR CUTLASS_DIR
"$work_dir/kernels/build.sh" sm_90
"$work_dir/kernels/build_fa_sm89_so.sh" sm_90a
"$work_dir/kernels/build_cutlass_so.sh" sm_90 "$CUTLASS_DIR"
"$work_dir/kernels/build_fa3.sh"
(cd "$work_dir/v3" && cargo build --release --locked --features cuda)
printf 'deployed revision=%s model_revision=%s\n' "$REVISION" "$MODEL_REVISION"
REMOTE_SCRIPT
