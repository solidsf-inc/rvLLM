#!/usr/bin/env bash
set -euo pipefail

: "${RVLLM_URL:?set RVLLM_URL to a running rvLLM server}"
: "${RVLLM_MODEL:?set RVLLM_MODEL to the served model name}"

RVLLM_URL=$(python3 - "$RVLLM_URL" <<'PY'
import sys, urllib.parse
raw = sys.argv[1]
if raw.strip() != raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
    raise SystemExit("RVLLM_URL must not contain whitespace or control characters")
try:
    endpoint = urllib.parse.urlsplit(raw)
    port = endpoint.port
except ValueError as error:
    raise SystemExit(f"invalid RVLLM_URL: {error}") from error
if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
    raise SystemExit("RVLLM_URL must be an http(s) origin")
if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
    raise SystemExit("RVLLM_URL must not contain credentials, a query, or a fragment")
if endpoint.path not in ("", "/"):
    raise SystemExit("RVLLM_URL must be an origin without an API path")
if endpoint.scheme == "http" and endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
    raise SystemExit("plaintext HTTP is allowed only for loopback endpoints")
netloc = endpoint.hostname
if ":" in netloc:
    netloc = f"[{netloc}]"
if port is not None:
    netloc = f"{netloc}:{port}"
print(urllib.parse.urlunsplit((endpoint.scheme, netloc, "", "", "")))
PY
)

headers=(-H 'Content-Type: application/json')
if [[ -n "${RVLLM_API_KEY:-}" ]]; then
    if [[ "$RVLLM_API_KEY" =~ [[:space:][:cntrl:]] ]]; then
        echo "RVLLM_API_KEY contains whitespace or control characters" >&2
        exit 1
    fi
    headers+=(-H "Authorization: Bearer ${RVLLM_API_KEY}")
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl --fail --silent --show-error --proto '=http,https' --max-time 10 "${headers[@]}" \
    "${RVLLM_URL%/}/health" > "$tmp/health"
[[ "$(<"$tmp/health")" == "ok" ]]

curl --fail --silent --show-error --proto '=http,https' --max-time 30 "${headers[@]}" \
    "${RVLLM_URL%/}/v1/models" > "$tmp/models.json"
python3 - "$tmp/models.json" "$RVLLM_MODEL" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert sys.argv[2] in [item["id"] for item in data["data"]]
PY

python3 - "$tmp/request.json" "$RVLLM_MODEL" <<'PY'
import json, sys
json.dump({
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Reply with ready."}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": False,
}, open(sys.argv[1], "w", encoding="utf-8"))
PY
curl --fail --silent --show-error --proto '=http,https' --max-time "${RVLLM_SMOKE_TIMEOUT:-300}" \
    "${headers[@]}" --data-binary "@$tmp/request.json" \
    "${RVLLM_URL%/}/v1/chat/completions" > "$tmp/response.json"
python3 - "$tmp/response.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["object"] == "chat.completion"
assert data["choices"][0]["message"]["role"] == "assistant"
PY
echo "rvLLM smoke test passed"
