#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
V3_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd -- "$V3_DIR/.." && pwd)

CARGO=${CARGO:-cargo}
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$V3_DIR/target"}
PPL_TEXT=${PPL_TEXT:-"give me the equations for 6dof angular momentum"}
LONG_PPL_TEXT=${LONG_PPL_TEXT:-"give me the equations for 6dof angular momentum. Use a rigid body with body-frame angular velocity omega, inertia tensor I, angular momentum H equals I omega, torque tau, and Euler equations including omega cross H. Also mention inertial position, attitude, force, and the coupled translational rotational six degree of freedom state in a concise derivation."}
CHAT_PROMPT=${CHAT_PROMPT:-"$PPL_TEXT"}
CHAT_MAX_TOKENS=${CHAT_MAX_TOKENS:-160}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-240}
CHAT_TIMEOUT_S=${CHAT_TIMEOUT_S:-1200}
HOST=${HOST:-127.0.0.1}
MODEL_NAME=${RVLLM_SERVED_MODEL_NAME:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-256}
MIN_PPL_TOK_S=${MIN_PPL_TOK_S:-5.0}
MAX_PPL=${MAX_PPL:-7000}
PPL_BASELINE_DELTA=${PPL_BASELINE_DELTA:-0.001}
MIN_PPL_INPUT_TOKENS=${MIN_PPL_INPUT_TOKENS:-2}
MIN_LONG_PPL_INPUT_TOKENS=${MIN_LONG_PPL_INPUT_TOKENS:-64}
MIN_LONG_PPL_TOK_S=${MIN_LONG_PPL_TOK_S:-5.0}
MAX_LONG_PPL=${MAX_LONG_PPL:-7000}
LONG_PPL_BASELINE_DELTA=${LONG_PPL_BASELINE_DELTA:-0.001}
MIN_CHAT_TOK_S=${MIN_CHAT_TOK_S:-1.0}
MAX_INPUT_BYTES=${MAX_INPUT_BYTES:-16384}
MODEL_REVISION=${RVLLM_MODEL_REVISION:-}
PPL_ORACLE_JSON=${PPL_ORACLE_JSON:-}
LONG_PPL_ORACLE_JSON=${LONG_PPL_ORACLE_JSON:-}

SERVER_BIN="$CARGO_TARGET_DIR/release/rvllm-server"
PPL_BIN="$CARGO_TARGET_DIR/release/rvllm-metal-ppl"
SERVER_PID=""
PORT_LOCK=""
RUN_TOKEN="$$.$RANDOM.$(date +%s)"

if [ -n "${OUT:-}" ]; then
  [ ! -e "$OUT" ] || { echo "FAIL: OUT already exists: $OUT" >&2; exit 1; }
  (umask 077 && mkdir -p "$OUT")
else
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/rvllm-metal-guardrail.XXXXXX")
fi

fail() {
  echo "FAIL: $*" >&2
  echo "artifacts in $OUT" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

cleanup() {
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ]; then
    case " $(jobs -pr 2>/dev/null || true) " in
      *" $SERVER_PID "*) kill -TERM "$SERVER_PID" 2>/dev/null || true ;;
    esac
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "${PORT_LOCK:-}" ] && [ -f "$PORT_LOCK/owner" ] &&
     [ "$(cat "$PORT_LOCK/owner" 2>/dev/null || true)" = "$RUN_TOKEN" ]; then
    rm -f "$PORT_LOCK/owner"
    rmdir "$PORT_LOCK" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

need_cmd "$CARGO"
need_cmd curl
need_cmd python3

[ -n "${RVLLM_MODEL_DIR:-}" ] || fail "set RVLLM_MODEL_DIR to the HF safetensors Gemma model dir"
[ -n "$MODEL_NAME" ] || fail "set RVLLM_SERVED_MODEL_NAME to the public model identifier"
[ -n "$MODEL_REVISION" ] || fail "set RVLLM_MODEL_REVISION to the immutable public model revision"
[ -n "$PPL_ORACLE_JSON" ] || fail "set PPL_ORACLE_JSON to independent-reference output"
[ -n "$LONG_PPL_ORACLE_JSON" ] || fail "set LONG_PPL_ORACLE_JSON to independent-reference output"
[ -d "$RVLLM_MODEL_DIR" ] || fail "RVLLM_MODEL_DIR is not a directory: $RVLLM_MODEL_DIR"
[ -f "$RVLLM_MODEL_DIR/config.json" ] || fail "missing config.json under RVLLM_MODEL_DIR"
[ -f "$RVLLM_MODEL_DIR/tokenizer.json" ] || fail "missing tokenizer.json under RVLLM_MODEL_DIR"
[ -f "$PPL_ORACLE_JSON" ] || fail "PPL oracle is not a file: $PPL_ORACLE_JSON"
[ -f "$LONG_PPL_ORACLE_JSON" ] || fail "long PPL oracle is not a file: $LONG_PPL_ORACLE_JSON"
[ "$HOST" = "127.0.0.1" ] || fail "HOST must be 127.0.0.1 for an unauthenticated local guardrail"
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$ ]] || fail "invalid served model identifier"
[[ "$MODEL_REVISION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "invalid model revision"

python3 - "$CHAT_MAX_TOKENS" "$MAX_MODEL_LEN" "$READY_TIMEOUT_S" "$CHAT_TIMEOUT_S" \
  "$MIN_PPL_INPUT_TOKENS" "$MIN_LONG_PPL_INPUT_TOKENS" "$MAX_INPUT_BYTES" \
  "$MIN_PPL_TOK_S" "$MAX_PPL" "$PPL_BASELINE_DELTA" "$MIN_LONG_PPL_TOK_S" \
  "$MAX_LONG_PPL" "$LONG_PPL_BASELINE_DELTA" "$MIN_CHAT_TOK_S" \
  "$PPL_TEXT" "$LONG_PPL_TEXT" "$CHAT_PROMPT" <<'PY' || fail "invalid numeric or input bounds"
import math
import sys

chat_tokens, model_len, ready_timeout, chat_timeout, min_ppl_tokens, min_long_tokens, max_bytes = map(int, sys.argv[1:8])
floats = list(map(float, sys.argv[8:15]))
texts = sys.argv[15:18]
if not (1 <= chat_tokens <= model_len <= 1_000_000):
    raise SystemExit(1)
if not (1 <= ready_timeout <= 600 and 1 <= chat_timeout <= 3600):
    raise SystemExit(1)
if not (2 <= min_ppl_tokens <= model_len and 2 <= min_long_tokens <= model_len):
    raise SystemExit(1)
if not (1 <= max_bytes <= 1_048_576):
    raise SystemExit(1)
if any(not math.isfinite(value) or value < 0 for value in floats):
    raise SystemExit(1)
if any(len(text.encode("utf-8")) == 0 or len(text.encode("utf-8")) > max_bytes for text in texts):
    raise SystemExit(1)
PY

case "$(printf '%s' "$RVLLM_MODEL_DIR" | tr '[:upper:]' '[:lower:]')" in
  *gguf*|*llama*) fail "refusing non-rvllm/Gemma-looking model path: $RVLLM_MODEL_DIR" ;;
esac
if find "$RVLLM_MODEL_DIR" -maxdepth 2 -iname '*.gguf' -print -quit | grep -q .; then
  fail "refusing GGUF artifact under RVLLM_MODEL_DIR"
fi
case "${RVLLM_DRY_RUN:-}" in
  1|true|TRUE|yes|YES) fail "RVLLM_DRY_RUN is set; this guardrail must run the real Metal backend" ;;
esac

if [ -z "${PORT:-}" ]; then
  PORT=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1024 ] && [ "$PORT" -le 65535 ] ||
  fail "PORT must be an unprivileged TCP port"
LOCK_ROOT="${TMPDIR:-/tmp}/rvllm-metal-port-locks"
mkdir -p "$LOCK_ROOT"
PORT_LOCK="$LOCK_ROOT/$PORT.lock"
mkdir "$PORT_LOCK" 2>/dev/null || fail "port lock is already held: $PORT_LOCK"
printf '%s\n' "$RUN_TOKEN" >"$PORT_LOCK/owner"
python3 - "$HOST" "$PORT" <<'PY' || fail "port is already in use: $HOST:$PORT"
import socket
import sys

with socket.socket() as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY
BASE_URL="http://$HOST:$PORT"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO_DIR"
  echo "v3=$V3_DIR"
  echo "out=$OUT"
  echo "model_dir=$RVLLM_MODEL_DIR"
  echo "backend=metal"
  echo "server_bin=$SERVER_BIN"
  echo "ppl_bin=$PPL_BIN"
  echo "base_url=$BASE_URL"
  git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null | sed 's/^/git_head=/'
} >"$OUT/run.env"
git -C "$REPO_DIR" status --short >"$OUT/git_status.txt" 2>/dev/null || true
printf '%s\n' "$PPL_TEXT" >"$OUT/ppl.prompt.txt"
printf '%s\n' "$LONG_PPL_TEXT" >"$OUT/long_ppl.prompt.txt"
printf '%s\n' "$CHAT_PROMPT" >"$OUT/chat.prompt.txt"

run_build() {
  local name="$1"
  shift
  echo "+ (cd $V3_DIR && CARGO_TARGET_DIR=$CARGO_TARGET_DIR $CARGO $*)" | tee -a "$OUT/commands.log"
  if ! (cd "$V3_DIR" && CARGO_TARGET_DIR="$CARGO_TARGET_DIR" "$CARGO" "$@") >"$OUT/build_$name.log" 2>&1; then
    tail -80 "$OUT/build_$name.log" >&2 || true
    fail "build failed: $name"
  fi
}

run_build server build --release -p rvllm-serve --features metal --bin rvllm-server
run_build ppl build --release -p rvllm-bench --features metal --bin rvllm-metal-ppl

[ -x "$SERVER_BIN" ] || fail "server binary missing after build: $SERVER_BIN"
[ -x "$PPL_BIN" ] || fail "PPL binary missing after build: $PPL_BIN"

echo "+ RVLLM_PROMPT=<ppl.prompt.txt> $PPL_BIN" | tee -a "$OUT/commands.log"
if ! RVLLM_PROMPT="$PPL_TEXT" "$PPL_BIN" >"$OUT/ppl.json" 2>"$OUT/ppl.log"; then
  tail -80 "$OUT/ppl.log" >&2 || true
  fail "rvllm-metal-ppl failed"
fi
python3 - "$OUT/ppl.json" "$PPL_ORACLE_JSON" "$MAX_PPL" "$MIN_PPL_TOK_S" \
  "$PPL_BASELINE_DELTA" "$MIN_PPL_INPUT_TOKENS" "$RVLLM_MODEL_DIR/config.json" "$RVLLM_MODEL_DIR/tokenizer.json" \
  "$MODEL_REVISION" "$PPL_TEXT" >"$OUT/ppl.check.txt" <<'PY'
import hashlib, json, math, sys
ppl_path, oracle_path = sys.argv[1], sys.argv[2]
max_ppl, min_tok_s, max_delta = map(float, sys.argv[3:6])
min_input_tokens = int(sys.argv[6])
config_path, tokenizer_path, model_revision, prompt = sys.argv[7:11]
def load(path):
    lines = [line.strip() for line in open(path) if line.strip()]
    if not lines:
        raise SystemExit(f"empty PPL output: {path}")
    return json.loads(lines[-1])
d, oracle = load(ppl_path), load(oracle_path)
implementation = str(oracle.get("implementation", "")).strip()
if not implementation or "rvllm" in implementation.lower():
    raise SystemExit("oracle implementation must identify an independent implementation")
def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
expected_meta = {
    "model_revision": model_revision,
    "config_sha256": digest_file(config_path),
    "tokenizer_sha256": digest_file(tokenizer_path),
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
}
for key, expected in expected_meta.items():
    if oracle.get(key) != expected:
        raise SystemExit(f"oracle {key} mismatch: {oracle.get(key)!r} != {expected!r}")
for key in ("perplexity", "total_nll", "tokens", "input_tokens", "tok_per_s"):
    if key not in d:
        raise SystemExit(f"missing {key} in PPL JSON")
if d["tokens"] <= 0 or d["input_tokens"] < min_input_tokens:
    raise SystemExit(f"invalid token counts: {d}")
if d["tokens"] != oracle.get("tokens") or d["input_tokens"] != oracle.get("input_tokens"):
    raise SystemExit(f"PPL oracle token mismatch: rvLLM={d} oracle={oracle}")
if not math.isfinite(d["total_nll"]) or d["total_nll"] <= 0:
    raise SystemExit(f"invalid total_nll: {d['total_nll']}")
if not math.isfinite(d["perplexity"]) or d["perplexity"] <= 0:
    raise SystemExit(f"invalid perplexity: {d['perplexity']}")
if d["perplexity"] > max_ppl:
    raise SystemExit(f"perplexity {d['perplexity']} exceeds MAX_PPL {max_ppl}")
if d["tok_per_s"] < min_tok_s:
    raise SystemExit(f"PPL tok_per_s {d['tok_per_s']} below MIN_PPL_TOK_S {min_tok_s}")
if not math.isfinite(float(oracle.get("total_nll", math.nan))):
    raise SystemExit("oracle total_nll is not finite")
delta = abs(float(d["total_nll"]) - float(oracle["total_nll"]))
if delta > max_delta:
    raise SystemExit(f"PPL total_nll delta {delta:.9f} exceeds {max_delta}: rvLLM={d} oracle={oracle}")
print(
    "REAL_NLL_PPL=pass "
    f"perplexity={d['perplexity']:.6f} total_nll={d['total_nll']:.6f} "
    f"oracle={implementation} oracle_total_nll={float(oracle['total_nll']):.6f} delta={delta:.9f} "
    f"tokens={d['tokens']} input_tokens={d['input_tokens']} tok_per_s={d['tok_per_s']:.3f}"
)
PY

echo "+ RVLLM_PROMPT=<long_ppl.prompt.txt> $PPL_BIN" | tee -a "$OUT/commands.log"
if ! RVLLM_PROMPT="$LONG_PPL_TEXT" "$PPL_BIN" >"$OUT/long_ppl.json" 2>"$OUT/long_ppl.log"; then
  tail -80 "$OUT/long_ppl.log" >&2 || true
  fail "rvllm-metal-ppl long prompt failed"
fi
python3 - "$OUT/long_ppl.json" "$LONG_PPL_ORACLE_JSON" "$MAX_LONG_PPL" \
  "$MIN_LONG_PPL_TOK_S" "$LONG_PPL_BASELINE_DELTA" "$MIN_LONG_PPL_INPUT_TOKENS" \
  "$RVLLM_MODEL_DIR/config.json" "$RVLLM_MODEL_DIR/tokenizer.json" "$MODEL_REVISION" \
  "$LONG_PPL_TEXT" >"$OUT/long_ppl.check.txt" <<'PY'
import hashlib, json, math, sys
ppl_path, oracle_path = sys.argv[1], sys.argv[2]
max_ppl, min_tok_s, max_delta = map(float, sys.argv[3:6])
min_input_tokens = int(sys.argv[6])
config_path, tokenizer_path, model_revision, prompt = sys.argv[7:11]
def load(path):
    lines = [line.strip() for line in open(path) if line.strip()]
    if not lines:
        raise SystemExit(f"empty PPL output: {path}")
    return json.loads(lines[-1])
d, oracle = load(ppl_path), load(oracle_path)
implementation = str(oracle.get("implementation", "")).strip()
if not implementation or "rvllm" in implementation.lower():
    raise SystemExit("oracle implementation must identify an independent implementation")
def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
expected_meta = {
    "model_revision": model_revision,
    "config_sha256": digest_file(config_path),
    "tokenizer_sha256": digest_file(tokenizer_path),
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
}
for key, expected in expected_meta.items():
    if oracle.get(key) != expected:
        raise SystemExit(f"oracle {key} mismatch: {oracle.get(key)!r} != {expected!r}")
for key in ("perplexity", "total_nll", "tokens", "input_tokens", "tok_per_s"):
    if key not in d:
        raise SystemExit(f"missing {key} in long PPL JSON")
if d["input_tokens"] < min_input_tokens:
    raise SystemExit(f"long PPL input_tokens {d['input_tokens']} below {min_input_tokens}")
if d["tokens"] <= 0:
    raise SystemExit(f"invalid long PPL token count: {d}")
if d["tokens"] != oracle.get("tokens") or d["input_tokens"] != oracle.get("input_tokens"):
    raise SystemExit(f"long PPL oracle token mismatch: rvLLM={d} oracle={oracle}")
if not math.isfinite(d["total_nll"]) or d["total_nll"] <= 0:
    raise SystemExit(f"invalid long total_nll: {d['total_nll']}")
if not math.isfinite(d["perplexity"]) or d["perplexity"] <= 0:
    raise SystemExit(f"invalid long perplexity: {d['perplexity']}")
if d["perplexity"] > max_ppl:
    raise SystemExit(f"long perplexity {d['perplexity']} exceeds MAX_LONG_PPL {max_ppl}")
if d["tok_per_s"] < min_tok_s:
    raise SystemExit(f"long PPL tok_per_s {d['tok_per_s']} below MIN_LONG_PPL_TOK_S {min_tok_s}")
if not math.isfinite(float(oracle.get("total_nll", math.nan))):
    raise SystemExit("oracle total_nll is not finite")
delta = abs(float(d["total_nll"]) - float(oracle["total_nll"]))
if delta > max_delta:
    raise SystemExit(f"long PPL total_nll delta {delta:.9f} exceeds {max_delta}: rvLLM={d} oracle={oracle}")
print(
    "LONG_REAL_NLL_PPL=pass "
    f"perplexity={d['perplexity']:.6f} total_nll={d['total_nll']:.6f} "
    f"oracle={implementation} oracle_total_nll={float(oracle['total_nll']):.6f} delta={delta:.9f} "
    f"tokens={d['tokens']} input_tokens={d['input_tokens']} tok_per_s={d['tok_per_s']:.3f}"
)
PY

echo "+ RVLLM_BACKEND=metal $SERVER_BIN --backend metal --host $HOST --port $PORT --max-model-len $MAX_MODEL_LEN" | tee -a "$OUT/commands.log"
(
  export RVLLM_BACKEND=metal
  export RVLLM_SERVED_MODEL_NAME="$MODEL_NAME"
  exec env \
    -u RVLLM_MODEL_REVISION \
    -u RVLLM_SOURCE_SHA \
    -u RVLLM_MODEL_ID \
    -u RVLLM_MODEL_SHA256 \
    -u RVLLM_HARDWARE \
    -u RVLLM_DRIVER \
    -u RVLLM_TOOLCHAIN \
    "$SERVER_BIN" --backend metal --host "$HOST" --port "$PORT" --max-model-len "$MAX_MODEL_LEN"
) >"$OUT/server.stdout" 2>"$OUT/server.log" &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"

deadline=$((SECONDS + READY_TIMEOUT_S))
until curl -fsS "$BASE_URL/status" >"$OUT/status.ready.json" 2>"$OUT/status.poll.log"; do
  kill -0 "$SERVER_PID" 2>/dev/null || {
    tail -120 "$OUT/server.log" >&2 || true
    fail "server exited before readiness"
  }
  [ "$SECONDS" -lt "$deadline" ] || {
    tail -120 "$OUT/server.log" >&2 || true
    fail "server did not become ready within ${READY_TIMEOUT_S}s"
  }
  sleep 1
done
python3 - "$OUT/status.ready.json" "$MODEL_NAME" "$MAX_MODEL_LEN" >"$OUT/status.check.txt" <<'PY'
import json, sys
path, model, max_len = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(path))
if d.get("backend") != "metal":
    raise SystemExit(f"backend is not metal: {d}")
if d.get("model") != model:
    raise SystemExit(f"model mismatch: {d.get('model')} != {model}")
if int(d.get("max_model_len", -1)) != max_len:
    raise SystemExit(f"max_model_len mismatch: {d.get('max_model_len')} != {max_len}")
print(f"STATUS=pass backend={d['backend']} model={d['model']} max_model_len={d['max_model_len']}")
PY

write_chat_request() {
  local path="$1"
  python3 - "$MODEL_NAME" "$CHAT_PROMPT" "$CHAT_MAX_TOKENS" >"$path" <<'PY'
import json, sys
model, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": max_tokens,
    "stream": False,
}, sys.stdout)
PY
}

check_chat_response() {
  local name="$1"
  local elapsed="$2"
  python3 - "$OUT/$name.response.json" "$OUT/$name.http_status" "$OUT/$name.content.txt" "$name" "$elapsed" "$MIN_CHAT_TOK_S" >"$OUT/$name.check.txt" <<'PY'
import json, re, sys
resp_path, status_path, content_path, name, elapsed, min_tok_s = sys.argv[1:7]
elapsed = float(elapsed)
min_tok_s = float(min_tok_s)
status = open(status_path).read().strip()
raw = open(resp_path).read()
if status != "200":
    raise SystemExit(f"{name}: HTTP {status}: {raw[:500]}")
d = json.loads(raw)
content = d["choices"][0]["message"]["content"]
open(content_path, "w").write(content)
text = content.strip()
words = re.findall(r"\S+", text)
if len(text) < 40 or len(words) < 8:
    raise SystemExit(f"{name}: too little text for coherence: {text!r}")
if "RVLLM_DRY_RUN" in text:
    raise SystemExit(f"{name}: dry-run marker in response")
if "\ufffd" in text:
    raise SystemExit(f"{name}: replacement character in response")
if words:
    top = max(words.count(w) for w in set(words)) / len(words)
    if len(words) >= 20 and top > 0.55:
        raise SystemExit(f"{name}: degenerate repetition ratio {top:.2f}")
lower = text.lower()
has_equation = (
    "=" in text
    or "×" in text
    or "\\dot" in text
    or re.search(r"\br\s*x\s*p\b", lower)
    or re.search(r"\b(dot|cross)\b", lower)
)
has_topic = any(t in lower for t in ("angular", "momentum", "torque", "inertia", "omega", "roll", "pitch", "yaw", "6dof")) or any(t in text for t in ("ω", "τ"))
has_symbol = any(re.search(p, text) for p in (r"\bL\b", r"\bH\b", r"\bI\b", r"\btau\b", r"\bomega\b", r"ω", r"τ"))
if not (has_equation and has_topic and has_symbol):
    raise SystemExit(f"{name}: response lacks equation/topic markers: {text[:500]!r}")
usage = d.get("usage", {})
completion_tokens = usage.get("completion_tokens")
if isinstance(completion_tokens, int) and elapsed > 0:
    tok_s = completion_tokens / elapsed
    if tok_s < min_tok_s:
        raise SystemExit(f"{name}: chat tok/s {tok_s:.3f} below MIN_CHAT_TOK_S {min_tok_s}")
else:
    tok_s = 0.0
print(
    f"COHERENCE_{name.upper()}=pass chars={len(text)} words={len(words)} "
    f"completion_tokens={usage.get('completion_tokens', 'na')} elapsed_s={elapsed:g} tok_s={tok_s:.3f}"
)
PY
}

chat_once() {
  local name="$1"
  local req="$OUT/$name.request.json"
  local resp="$OUT/$name.response.json"
  write_chat_request "$req"
  echo "+ curl $BASE_URL/v1/chat/completions ($name)" | tee -a "$OUT/commands.log"
  local t0=$SECONDS
  if ! curl -sS --max-time "$CHAT_TIMEOUT_S" \
      -o "$resp" \
      -w "%{http_code}" \
      -H "Content-Type: application/json" \
      -d @"$req" \
      "$BASE_URL/v1/chat/completions" >"$OUT/$name.http_status"; then
    tail -120 "$OUT/server.log" >&2 || true
    fail "$name chat request failed"
  fi
  check_chat_response "$name" "$((SECONDS - t0))"
}

chat_once cold
chat_once warm
curl -fsS "$BASE_URL/metrics" >"$OUT/metrics.after.txt" 2>"$OUT/metrics.after.log" || true

{
  cat "$OUT/ppl.check.txt"
  cat "$OUT/long_ppl.check.txt"
  cat "$OUT/status.check.txt"
  cat "$OUT/cold.check.txt"
  cat "$OUT/warm.check.txt"
  echo "ARTIFACTS=$OUT"
} | tee "$OUT/summary.txt"
