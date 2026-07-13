#!/usr/bin/env python3
"""Run bounded non-streaming chat load and emit a self-hashed receipt."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import time
import urllib.error
import urllib.parse
import urllib.request

PROMPTS = (
    "Write one short sentence about inference.",
    "Name three primary colors.",
    "Continue: alpha beta gamma",
    "Return the word ready.",
)


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = urllib.request.build_opener(RejectRedirects())


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percent / 100)]


def request_once(url, model, max_tokens, timeout, index):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("RVLLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/v1/chat/completions", body, headers, method="POST"
        )
        with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            payload = json.load(response)
        latency_ms = (time.perf_counter() - started) * 1000
        if not payload.get("choices"):
            raise ValueError("response has no choices")
        usage = payload.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        return {"latency_ms": latency_ms, "completion_tokens": completion_tokens, "error": None}
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as error:
        return {"latency_ms": None, "completion_tokens": None, "error": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise SystemExit("--source-sha must be a 40-character lowercase commit SHA")
    if not 1 <= args.requests <= 10_000:
        raise SystemExit("--requests must be 1..10000")
    if not 1 <= args.concurrency <= 256:
        raise SystemExit("--concurrency must be 1..256")
    if not 1 <= args.max_tokens <= 1_048_576:
        raise SystemExit("--max-tokens must be 1..1048576")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3_600:
        raise SystemExit("--timeout must be greater than 0 and at most 3600 seconds")
    for flag, value in (("--model", args.model), ("--hardware", args.hardware)):
        if not value or len(value) > 256 or value.strip() != value:
            raise SystemExit(f"{flag} must be 1..256 characters without outer whitespace")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise SystemExit(f"{flag} must not contain control characters")
    endpoint = urllib.parse.urlsplit(args.url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise SystemExit("--url must be an http(s) origin")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise SystemExit("--url must not contain credentials, a query, or a fragment")
    if endpoint.path not in ("", "/"):
        raise SystemExit("--url must be an origin without an API path")
    try:
        endpoint.port
    except ValueError as error:
        raise SystemExit(f"invalid --url port: {error}") from error
    if endpoint.scheme == "http" and endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("plaintext HTTP is allowed only for loopback endpoints")
    url = urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, "", "", ""))

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda index: request_once(
                    url, args.model, args.max_tokens, args.timeout, index
                ),
                range(args.requests),
            )
        )
    elapsed = time.perf_counter() - started
    successes = [result for result in results if result["error"] is None]
    if not successes:
        raise SystemExit("all requests failed")
    latencies = [result["latency_ms"] for result in successes]
    token_counts = [result["completion_tokens"] for result in successes]
    token_usage_complete = all(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
        for value in token_counts
    )
    completion_tokens = sum(token_counts) if token_usage_complete else None
    errors = {}
    for result in results:
        if result["error"]:
            errors[result["error"]] = errors.get(result["error"], 0) + 1

    receipt = {
        "schema": "rvllm.benchmark.v1",
        "source_sha": args.source_sha,
        "hardware": args.hardware,
        "model": args.model,
        "endpoint": url,
        "request_count": args.requests,
        "success_count": len(successes),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed if completion_tokens is not None else None,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "errors": errors,
        "client": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
        },
        "command": [
            "python3",
            "deploy/benchmark_client.py",
            "--url",
            url,
            "--model",
            args.model,
            "--source-sha",
            args.source_sha,
            "--hardware",
            args.hardware,
            "--requests",
            str(args.requests),
            "--concurrency",
            str(args.concurrency),
            "--max-tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.timeout),
            "--output",
            "<receipt.json>",
        ],
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
