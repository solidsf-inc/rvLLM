# Error contract

Public API errors have a stable HTTP status, short type, and sanitized message.
Internal errors retain subsystem, operation, dimensions, backend, and source
chain for local logs. Neither surface may contain API keys, prompts, model
contents, full local paths, hostnames, or pointer values.

Validation and unsupported-feature errors occur before launch. Device failures
preserve the original error class; cleanup errors do not hide the primary
failure. Panics are reserved for internal invariant violations and are not an
input-validation strategy.

The current Rust error enums are implementation detail unless explicitly
documented. Tests must assert redaction and the API error envelope rather than
unstable debug text.
