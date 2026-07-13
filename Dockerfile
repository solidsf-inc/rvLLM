ARG RUST_IMAGE=rust:1.95.0-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7

FROM ${RUST_IMAGE} AS rust-toolchain

FROM ${CUDA_DEVEL_IMAGE} AS builder
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY v3 v3
RUN cargo build --manifest-path v3/Cargo.toml --locked --release \
    -p rvllm-serve --bin rvllm-server --features cuda

FROM ${CUDA_RUNTIME_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/v3/target/release/rvllm-server /usr/local/bin/rvllm-server
ENV RVLLM_BACKEND=cuda
ENV RVLLM_KERNELS_DIR=/artifacts
EXPOSE 8080
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/rvllm-server"]
CMD ["--help"]
