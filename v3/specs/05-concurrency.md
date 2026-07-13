# Concurrency

The current HTTP layer accepts a TCP connection and uses one blocking OS thread
per connection. Generation is admitted through a bounded worker queue and a
single persistent model session; excess work returns a busy response. This is
not an async HTTP runtime or a claim of unbounded continuous batching.

Limits include header/body bytes, connection/request timeouts, in-flight
requests, sequence count, model length, and prefill chunk size. Cancellation
must not free buffers still used by the device. Shutdown must stop admission,
drain or cancel queued work, fence device activity, and then destroy graphs and
memory.

A future async frontend may replace connection threads without changing the
worker ownership contract.
