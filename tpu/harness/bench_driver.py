"""Fail-closed rvLLM HTTP throughput and total-latency benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import urllib.parse

import aiohttp


def validate_base_url(raw):
    if not raw or raw.strip() != raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ValueError("--base-url must not contain whitespace or control characters")
    try:
        endpoint = urllib.parse.urlsplit(raw)
        port = endpoint.port
    except ValueError as error:
        raise ValueError(f"invalid --base-url: {error}") from error
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("--base-url must be an http(s) origin")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("--base-url must not contain credentials, a query, or a fragment")
    if endpoint.path not in ("", "/"):
        raise ValueError("--base-url must be an origin without an API path")
    if endpoint.scheme == "http" and endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("plaintext HTTP is allowed only for loopback endpoints")
    netloc = endpoint.hostname
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit((endpoint.scheme, netloc, "", "", ""))


async def completion_request(session, url, model, prompt, max_tokens):
    started = time.perf_counter()
    async with session.post(
        f"{url}/v1/completions",
        allow_redirects=False,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
    ) as response:
        if 300 <= response.status < 400:
            raise RuntimeError("completion endpoint returned a redirect")
        response.raise_for_status()
        body = await response.json()
    elapsed = time.perf_counter() - started
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("completion response omitted usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise RuntimeError("invalid prompt token count")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        raise RuntimeError("invalid completion token count")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
    }


async def chat_latency_ms(session, url, model, prompt, max_tokens):
    started = time.perf_counter()
    async with session.post(
        f"{url}/v1/chat/completions",
        allow_redirects=False,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
    ) as response:
        if 300 <= response.status < 400:
            raise RuntimeError("chat endpoint returned a redirect")
        response.raise_for_status()
        body = await response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("chat response omitted choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("chat response omitted assistant content")
    return (time.perf_counter() - started) * 1_000


async def run_batch(session, url, model, prompt, concurrency, max_tokens):
    started = time.perf_counter()
    results = await asyncio.gather(
        *[
            completion_request(session, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
    )
    wall = time.perf_counter() - started
    completion_tokens = sum(result["completion_tokens"] for result in results)
    prompt_counts = {result["prompt_tokens"] for result in results}
    if len(prompt_counts) != 1 or wall <= 0:
        raise RuntimeError("inconsistent prompt accounting or invalid duration")
    return {
        "concurrency": concurrency,
        "prompt_tokens_per_request": prompt_counts.pop(),
        "completion_tokens": completion_tokens,
        "wall_s": wall,
        "output_tokens_per_second": completion_tokens / wall,
    }


async def main_async(args):
    headers = {}
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"{args.api_key_env} is not set")
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in api_key):
            raise RuntimeError(f"{args.api_key_env} contains whitespace or control characters")
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.batches))
    result = {
        "kind": "rvllm-http-diagnostic",
        "model": args.model,
        "base_url": args.base_url,
        "runs": [],
    }
    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        for concurrency in args.batches:
            for phase in ("cold", "hot"):
                run = await run_batch(
                    session,
                    args.base_url,
                    args.model,
                    args.prompt,
                    concurrency,
                    args.max_tokens,
                )
                run["phase"] = phase
                run["chat_latency_ms"] = await chat_latency_ms(
                    session,
                    args.base_url,
                    args.model,
                    args.prompt,
                    args.max_tokens,
                )
                if not all(
                    math.isfinite(run[key])
                    for key in ("wall_s", "output_tokens_per_second", "chat_latency_ms")
                ):
                    raise RuntimeError("benchmark produced a non-finite metric")
                result["runs"].append(run)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain bounded memory allocation in Rust.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batches", default="1,8,16,64")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--api-key-env", default="RVLLM_API_KEY")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        args.base_url = validate_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    try:
        args.batches = [int(value) for value in args.batches.split(",")]
    except ValueError:
        parser.error("--batches must be comma-separated integers")
    if not args.batches or any(value <= 0 or value > 1024 for value in args.batches):
        parser.error("every batch size must be in 1..1024")
    if args.max_tokens <= 0 or args.timeout <= 0 or not args.prompt:
        parser.error("prompt, token count, and timeout must be positive")
    return args


def main():
    args = parse_args()
    result = asyncio.run(main_async(args))
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
