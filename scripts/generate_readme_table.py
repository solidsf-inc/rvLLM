#!/usr/bin/env python3
"""Validate an rvLLM benchmark receipt and render a compact Markdown table."""

import argparse
import hashlib
import html
import json
import math
import re


def load_receipt(path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    with open(path, encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )


def verify(receipt):
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a JSON object")
    expected = receipt.pop("receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt["receipt_sha256"] = expected
    if expected != actual:
        raise ValueError("receipt_sha256 does not match receipt contents")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("receipt_sha256 must be 64 lowercase hexadecimal characters")
    required = {
        "schema",
        "source_sha",
        "hardware",
        "model",
        "request_count",
        "success_count",
        "concurrency",
        "latency_ms",
    }
    missing = sorted(required - receipt.keys())
    if missing:
        raise ValueError(f"missing receipt fields: {', '.join(missing)}")
    if receipt["schema"] != "rvllm.benchmark.v1":
        raise ValueError("unsupported receipt schema")
    if not isinstance(receipt["source_sha"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", receipt["source_sha"]
    ):
        raise ValueError("source_sha must be 40 lowercase hexadecimal characters")
    for key in ("hardware", "model"):
        value = receipt[key]
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(f"{key} must be a non-empty string of at most 256 characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{key} must not contain control characters")
    for key in ("request_count", "success_count", "concurrency"):
        value = receipt[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if receipt["success_count"] > receipt["request_count"]:
        raise ValueError("success_count exceeds request_count")
    latency = receipt["latency_ms"]
    if not isinstance(latency, dict):
        raise ValueError("latency_ms must be an object")
    for key in ("mean", "p95"):
        value = latency.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"latency_ms.{key} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"latency_ms.{key} must be finite and non-negative")
    throughput = receipt.get("completion_tokens_per_second")
    if throughput is not None:
        if isinstance(throughput, bool) or not isinstance(throughput, (int, float)):
            raise ValueError("completion_tokens_per_second must be a number or null")
        if not math.isfinite(throughput) or throughput < 0:
            raise ValueError("completion_tokens_per_second must be finite and non-negative")


def markdown(value):
    return html.escape(value, quote=False).replace("`", "&#96;").replace("|", "&#124;")


def render(receipt):
    latency = receipt["latency_ms"]
    throughput = receipt.get("completion_tokens_per_second")
    throughput_text = "not reported" if throughput is None else f"{throughput:.2f} tok/s"
    return "\n".join(
        [
            f"Source `{receipt['source_sha']}` · {markdown(receipt['hardware'])} · "
            f"`{markdown(receipt['model'])}`",
            "",
            "| Successful requests | Concurrency | Completion throughput | Mean latency | P95 latency |",
            "|---:|---:|---:|---:|---:|",
            f"| {receipt['success_count']} | {receipt['concurrency']} | {throughput_text} | "
            f"{latency['mean']:.2f} ms | {latency['p95']:.2f} ms |",
            "",
            f"Receipt SHA-256: `{receipt['receipt_sha256']}`",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = load_receipt(args.input)
    verify(receipt)
    markdown = render(receipt) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
