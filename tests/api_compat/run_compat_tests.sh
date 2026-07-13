#!/usr/bin/env bash
set -euo pipefail
: "${RVLLM_URL:?set RVLLM_URL to a running rvLLM server}"
: "${RVLLM_MODEL:?set RVLLM_MODEL to the served model name}"
python3 tests/api_compat/test_openai_client.py -v
