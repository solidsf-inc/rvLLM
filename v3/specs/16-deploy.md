# Deployment

The public release supports source builds and the pinned repository Dockerfile.
It does not publish a trusted binary/kernel artifact channel yet.

A deployable bundle must bind these items together by digest: source commit,
Rust/CUDA toolchains, target architecture, executable, shared libraries,
compiled kernels and manifest, policy file, configuration, and model revision.
The kernel loader verifies listed artifact hashes. A signature and public key
are still required before third-party prebuilt bundles can be recommended.

`rvllm-server` binds loopback only. Use read-only model/artifact mounts, a
non-root runtime user, explicit resource limits, and an audited local proxy for
remote TLS/authentication. Never pull mutable tags, fall back to “latest,” or
download executable artifacts from a personal namespace.
