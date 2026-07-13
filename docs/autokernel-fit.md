# Kernel fitting workflow

Use `v3/crates/rvllm-bench` for a fixed batch and iteration count. Change one
kernel choice at a time, run correctness tests first, then profile the same
source revision and inputs.

```bash
cargo test --manifest-path v3/Cargo.toml --workspace --locked
RVLLM_PROFILE_BATCHES=1,8,32 scripts/autokernel_loop.sh --iters 100 --warmup 10
```

Accept a dispatch change only when all supported shapes pass reference/parity
tests and the raw receipt shows a repeatable improvement on named hardware.
Reject results with warmup failures, hidden fallbacks, missing artifacts,
different models, or unmatched source SHAs.
