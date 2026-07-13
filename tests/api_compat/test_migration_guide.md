# API migration notes

rvLLM implements a limited OpenAI-compatible surface; it is not a universal
drop-in replacement.

| Route | Status | Notes |
|---|---|---|
| `GET /health` | implemented | local supervision |
| `GET /status`, `/metrics` | implemented | rvLLM-specific metadata |
| `GET /v1/models` | implemented | one served model |
| `POST /v1/completions` | implemented | one prompt, non-streaming |
| `POST /v1/chat/completions` | implemented | `n=1`, non-streaming |

Supported generation fields include `model`, `max_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, and `ignore_eos` where represented by the request
type. Requests with `stream=true` or `stop` are rejected. Multiple choices,
stored responses, tools, assistants, and the Responses API are not implemented.

Set an explicit model name and timeout. The server accepts loopback binds only;
use `RVLLM_API_KEY` for local bearer authentication and a trusted proxy for
remote TLS. Run the live matrix with:

```bash
RVLLM_URL=http://127.0.0.1:8080 RVLLM_MODEL=<served-name> \
  tests/api_compat/run_compat_tests.sh
```
