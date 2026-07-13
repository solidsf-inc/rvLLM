# Scheduler

The general scheduler produces bounded batch plans, while the serving path
currently executes through one model worker/session. Admission limits are
configured separately from the model's supported active sequence count.

A plan owns request IDs, logical sequence lengths, page/slot mappings, prompt
or decode phase, token budget, and cancellation state. It may reference only
committed KV positions. Fairness is best-effort FIFO at the worker boundary;
there is no published starvation guarantee.

Invalid lengths, duplicate ownership, exhausted pages, unsupported batch sizes,
or arithmetic overflow reject the plan. Tests cover admission saturation,
cancellation, EOS/stop completion, page exhaustion, and deterministic plan
construction.
