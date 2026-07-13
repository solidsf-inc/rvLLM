# Release validation checklist

Portable gates:

```bash
cargo fmt --manifest-path v3/Cargo.toml --all -- --check
cargo check --manifest-path v3/Cargo.toml --workspace --all-targets --locked
cargo test --manifest-path v3/Cargo.toml --workspace --locked
cargo check --manifest-path chat-client/Cargo.toml --locked
python3 -m compileall -q deploy scripts tests fp8_precision_check.py
scripts/validate-local.sh
```

Hardware gates must attach immutable receipts for CUDA/Metal kernel parity,
eager/replay identity, real-weight logits and perplexity, KV page/ring
boundaries, long generation, memory limits, and sanitizer/profiler runs.

Server gate:

```bash
RVLLM_URL=http://127.0.0.1:8080 RVLLM_MODEL=<served-name> \
  tests/api_compat/run_compat_tests.sh
```

Also verify the kernel manifest, third-party provenance, dependency licenses,
container digests, leak scan, and absence of models/secrets/generated captures.
Do not mark the release correct or performant until the corresponding real
hardware receipts are published.
