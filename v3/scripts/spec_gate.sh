#!/usr/bin/env bash
# Reproducible greedy speculative-decoding parity gate.
set -euo pipefail

BENCH=${BENCH:?set BENCH to the rvLLM benchmark executable}
GEN_TOKENS=${GEN_TOKENS:-128}
GEN_PROMPT=${GEN_PROMPT:-32}
SWEEP_KS=${SWEEP_KS:-"2 4 6 8"}
PROMPT_FILE=${PROMPT_FILE:-}
RING_PROMPT_FILE=${RING_PROMPT_FILE:?set RING_PROMPT_FILE to token IDs spanning a ring wrap}
RING_WINDOW_TOKENS=${RING_WINDOW_TOKENS:?set RING_WINDOW_TOKENS to the tested sliding-window capacity}
RING_GEN_TOKENS=${RING_GEN_TOKENS:-64}
SOURCE_REVISION=${SOURCE_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}

if [ -n "${OUT:-}" ]; then
  [ ! -e "$OUT" ] || { echo "ERROR: OUT already exists: $OUT" >&2; exit 1; }
  (umask 077 && mkdir -p "$OUT")
else
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-spec-gate.XXXXXX")
fi

fail() {
  echo "ERROR: $*" >&2
  echo "artifacts: $OUT" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[ -x "$BENCH" ] || fail "BENCH is not executable: $BENCH"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "SOURCE_REVISION must be a full commit"

for name in RVLLM_MODEL_DIR RVLLM_KERNELS_DIR RVLLM_FA3_SO RVLLM_CUTLASS_SO RVLLM_POLICY; do
  [ -n "${!name:-}" ] || fail "set $name"
done
[ -d "$RVLLM_MODEL_DIR" ] || fail "RVLLM_MODEL_DIR is not a directory"
[ -d "$RVLLM_KERNELS_DIR" ] || fail "RVLLM_KERNELS_DIR is not a directory"
for path in "$RVLLM_FA3_SO" "$RVLLM_CUTLASS_SO" "$RVLLM_POLICY" "$RING_PROMPT_FILE"; do
  [ -f "$path" ] || fail "required file is missing: $path"
done
if [ -n "$PROMPT_FILE" ]; then
  [ -f "$PROMPT_FILE" ] || fail "PROMPT_FILE is not a file"
fi

python3 - "$GEN_TOKENS" "$GEN_PROMPT" "$RING_GEN_TOKENS" "$RING_WINDOW_TOKENS" \
  "$SWEEP_KS" "$PROMPT_FILE" "$RING_PROMPT_FILE" <<'PY' || fail "invalid gate inputs"
import pathlib
import sys

gen_tokens, gen_prompt, ring_tokens, window = map(int, sys.argv[1:5])
ks = [int(value) for value in sys.argv[5].split()]
prompt_path, ring_path = sys.argv[6:8]
if not (1 <= gen_tokens <= 4096 and 1 <= gen_prompt <= 1_000_000):
    raise SystemExit(1)
if not (1 <= ring_tokens <= 4096 and 1 <= window <= 1_000_000):
    raise SystemExit(1)
if not ks or len(ks) > 32 or any(k < 1 or k > 64 for k in ks) or len(ks) != len(set(ks)):
    raise SystemExit(1)

def token_ids(path):
    if not path:
        return []
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    if len(raw.encode()) > 16 * 1024 * 1024:
        raise SystemExit("prompt file exceeds 16 MiB")
    values = [int(value) for value in raw.split()]
    if any(value < 0 or value > 2_147_483_647 for value in values):
        raise SystemExit("prompt token ID out of range")
    return values

token_ids(prompt_path)
ring = token_ids(ring_path)
if len(ring) <= window:
    raise SystemExit(f"ring prompt has {len(ring)} tokens; it must exceed window {window}")
PY

python3 - "$OUT/acceptance_oracle.json" <<'PY'
import json
import sys

def greedy_accept(target, draft):
    if len(target) < len(draft) + 1:
        raise ValueError("target must contain a correction/bonus token")
    for index, token in enumerate(draft):
        if token != target[index]:
            return draft[:index] + [target[index]]
    return draft + [target[len(draft)]]

vectors = [
    ([10, 11, 12], [99, 98], [10]),
    ([10, 11, 12], [10, 98], [10, 11]),
    ([10, 11, 12], [10, 11], [10, 11, 12]),
    ([7], [], [7]),
]
results = []
for target, draft, expected in vectors:
    actual = greedy_accept(target, draft)
    if actual != expected:
        raise SystemExit(f"acceptance oracle mismatch: {actual} != {expected}")
    results.append({"target": target, "draft": draft, "accepted": actual})
json.dump({"algorithm": "clean-room-greedy-prefix-v1", "vectors": results}, open(sys.argv[1], "w"), indent=2)
PY

run_one() {
  local name=$1
  local tokens=$2
  local prompt_len=$3
  local prompt_file=$4
  shift 4
  (
    unset RVLLM_SPEC_DECODE RVLLM_SPEC_K RVLLM_DECODE_GRAPH RVLLM_GEN_PROMPT_FILE
    export RVLLM_GEN_TOKENS="$tokens" RVLLM_GEN_PROMPT="$prompt_len" RVLLM_F16_KV=0
    [ -z "$prompt_file" ] || export RVLLM_GEN_PROMPT_FILE="$prompt_file"
    local assignment
    for assignment in "$@"; do export "${assignment?}"; done
    export RVLLM_GEN_DUMP="$OUT/$name.tokens"
    "$BENCH" --generate >"$OUT/$name.raw.json" 2>"$OUT/$name.log"
  ) || { tail -20 "$OUT/$name.log" >&2 || true; fail "$name failed"; }

  python3 - "$OUT/$name.raw.json" "$OUT/$name.tokens" "$OUT/$name.json" "$name" <<'PY'
import json
import math
import pathlib
import sys

raw_path, token_path, output_path, name = sys.argv[1:]
lines = [line for line in pathlib.Path(raw_path).read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"{name}: empty benchmark output")
record = json.loads(lines[-1])
for key in ("token_hash", "e2e_tok_per_sec"):
    if key not in record:
        raise SystemExit(f"{name}: missing {key}")
if not math.isfinite(float(record["e2e_tok_per_sec"])) or float(record["e2e_tok_per_sec"]) <= 0:
    raise SystemExit(f"{name}: invalid throughput")
tokens = [int(value) for value in pathlib.Path(token_path).read_text().split()]
if not tokens:
    raise SystemExit(f"{name}: empty token dump")
record = {
    "name": name,
    "token_hash": str(record["token_hash"]),
    "e2e_tok_per_sec": float(record["e2e_tok_per_sec"]),
    "token_count": len(tokens),
}
pathlib.Path(output_path).write_text(json.dumps(record, sort_keys=True) + "\n")
PY
}

compare_tokens() {
  local reference=$1
  local candidate=$2
  python3 - "$OUT/$reference.tokens" "$OUT/$candidate.tokens" "$reference" "$candidate" <<'PY'
import pathlib
import sys

ref_path, candidate_path, ref_name, candidate_name = sys.argv[1:]
reference = [int(value) for value in pathlib.Path(ref_path).read_text().split()]
candidate = [int(value) for value in pathlib.Path(candidate_path).read_text().split()]
if reference != candidate:
    common = min(len(reference), len(candidate))
    index = next((i for i in range(common) if reference[i] != candidate[i]), common)
    raise SystemExit(
        f"{candidate_name} diverges from {ref_name} at {index}; "
        f"lengths {len(reference)} and {len(candidate)}"
    )
PY
}

run_one baseline "$GEN_TOKENS" "$GEN_PROMPT" "$PROMPT_FILE" RVLLM_DECODE_GRAPH=1
run_one spec_k0 "$GEN_TOKENS" "$GEN_PROMPT" "$PROMPT_FILE" RVLLM_SPEC_DECODE=1 RVLLM_SPEC_K=0
run_one spec_k4 "$GEN_TOKENS" "$GEN_PROMPT" "$PROMPT_FILE" RVLLM_SPEC_DECODE=1 RVLLM_SPEC_K=4
compare_tokens baseline spec_k0
compare_tokens baseline spec_k4

ring_prompt_len=$(python3 - "$RING_PROMPT_FILE" <<'PY'
import pathlib, sys
print(len(pathlib.Path(sys.argv[1]).read_text().split()))
PY
)
run_one ring_baseline "$RING_GEN_TOKENS" "$ring_prompt_len" "$RING_PROMPT_FILE" RVLLM_DECODE_GRAPH=1
run_one ring_spec_k4 "$RING_GEN_TOKENS" "$ring_prompt_len" "$RING_PROMPT_FILE" RVLLM_SPEC_DECODE=1 RVLLM_SPEC_K=4
compare_tokens ring_baseline ring_spec_k4

for k in $SWEEP_KS; do
  run_one "sweep_k$k" "$GEN_TOKENS" "$GEN_PROMPT" "$PROMPT_FILE" RVLLM_SPEC_DECODE=1 "RVLLM_SPEC_K=$k"
  compare_tokens baseline "sweep_k$k"
done

uname -srm >"$OUT/system.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader >"$OUT/gpu.csv"
fi

python3 - "$OUT" "$SOURCE_REVISION" "$BENCH" "$RVLLM_FA3_SO" "$RVLLM_CUTLASS_SO" \
  "$RVLLM_POLICY" "$SWEEP_KS" <<'PY'
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
revision, bench, fa, cutlass, policy, sweep = sys.argv[2:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
names = ["baseline", "spec_k0", "spec_k4", "ring_baseline", "ring_spec_k4"]
names += [f"sweep_k{k}" for k in sweep.split()]
runs = [json.loads((out / f"{name}.json").read_text()) for name in names]
evidence = {
    "schema": "rvllm-spec-parity-v1",
    "source_revision": revision,
    "artifacts": {
        "bench_sha256": digest(bench),
        "attention_sha256": digest(fa),
        "cutlass_sha256": digest(cutlass),
        "policy_sha256": digest(policy),
    },
    "acceptance_oracle": json.loads((out / "acceptance_oracle.json").read_text()),
    "runs": runs,
    "result": "pass",
}
(out / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
print(f"PASS: {out / 'evidence.json'}")
PY
