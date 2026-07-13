#!/usr/bin/env bash
# Run a bounded rvLLM batch sweep. Required model/artifact paths are inherited.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$root/scripts/profile_compare.sh" --batches "${RVLLM_PROFILE_BATCHES:-1,8,32}" "$@"
