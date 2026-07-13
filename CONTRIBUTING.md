# Contributing to rvLLM

Work from a branch and keep changes scoped. The maintained Rust workspace is
`v3/`; root-level legacy paths are not release targets.

Before opening a pull request:

```bash
cargo fmt --manifest-path v3/Cargo.toml --all -- --check
cargo check --manifest-path v3/Cargo.toml --workspace --all-targets --locked
cargo test --manifest-path v3/Cargo.toml --workspace --locked
```

CUDA and Metal changes also need real-device tests. Include the hardware,
driver/toolchain versions, exact source SHA, command, raw output, and failures.
Do not describe a scaffold or compile-only path as supported.

New dependencies must have a pinned lockfile entry, compatible license, and a
clear need. Ports must name the exact upstream revision and source files and
preserve required notices. Never commit models, credentials, private hostnames,
customer data, local paths, or generated benchmark claims without a complete
receipt.

Security issues should be reported privately to the repository owner, not in a
public issue.
