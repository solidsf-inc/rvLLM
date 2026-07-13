#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

command -v docker >/dev/null || {
  echo "docker is required" >&2
  exit 1
}

SOURCE_SHA=${SOURCE_SHA:-$(git rev-parse --verify HEAD 2>/dev/null || true)}
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Set SOURCE_SHA to the 40-character released source commit." >&2
  exit 1
}
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "Refusing to build a release image from a dirty tree." >&2
  exit 1
fi

IMAGE=${IMAGE_REPOSITORY:-rvllm}:${SOURCE_SHA}
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
git archive "$SOURCE_SHA" | tar -x -C "$context"
docker build \
  --label "org.opencontainers.image.revision=$SOURCE_SHA" \
  --tag "$IMAGE" \
  --file "$context/Dockerfile" \
  "$context"
echo "$IMAGE"
