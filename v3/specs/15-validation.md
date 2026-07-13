# Validation

Portable CI runs formatting, locked workspace checks/tests, dependency license
and advisory review, script syntax, leak scans, and attribution checks.

Hardware release gates are separate and receipt-bound:

- CUDA and Metal kernel/reference parity for every enabled shape;
- eager versus graph replay;
- malformed model/metadata/artifact rejection;
- real-weight logits and perplexity against a pinned public reference;
- long generation with KV page/ring boundaries;
- server smoke, non-streaming rejection, authentication, and limits;
- sanitizer/profiler runs where the toolchain supports them.

A gate is “passed” only when its raw output names the source SHA, model
revision, hardware, driver/toolchain, artifacts, command, and digest. This
document records requirements, not results.
