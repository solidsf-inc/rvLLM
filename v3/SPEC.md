# rvLLM v3 contract

rvLLM is an accelerator-oriented Rust inference workspace. Its core contract is
explicit state: validated model files, bounded memory regions, immutable kernel
artifacts, checked metadata, typed backend dispatch, and a scheduler-owned
request lifecycle.

The integrated CUDA text path is the primary release target. The server exposes
a limited local OpenAI-compatible API. Prompt-lookup speculative decoding is
opt-in. Metal and vision modules are present but do not constitute end-to-end
support until public real-device and real-weight gates pass.

No operation may report success after a no-op, zero-output placeholder, missing
kernel, unsupported dtype/shape, or stale graph fingerprint. Files, requests,
policy JSON, and compiled artifacts are untrusted inputs. All extents and
offsets use checked arithmetic and all asynchronous lifetimes extend through
device completion.

The numbered documents describe subsystem contracts. Source and executable
tests take precedence over stale prose; performance requires an immutable
receipt.
