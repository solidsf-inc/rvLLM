# Configuration

`rvllm_serve::ServeConfig::from_env_and_args` is the authoritative server
registry. `rvllm-server --help` and `.env.example` are the public views; new
server variables must be added to the parser and tests before documentation.

Safe defaults are loopback host, port 8080, one active sequence, bounded
in-flight requests, and no vision or speculative decoding. Paths must be
explicit and contained within their configured roots. Empty, zero, inconsistent,
or unknown values are errors.

`RVLLM_API_KEY` is a secret and must never be logged or committed. The built-in
HTTP server is local-only even when a bearer key is set. Remote access requires
an audited TLS/authentication proxy.

Kernel and benchmark binaries have additional variables documented next to
their source. They are not silently accepted by the server parser.
