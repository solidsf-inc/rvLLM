//! rvllm-sampling: GPU-side greedy argmax + temperature/top-k/top-p.
//!
//! DtoH coordination uses a consume-once `DtoHTicket<'p>` that borrows
//! `&mut PinnedTokens`. The type state makes "launch twice before
//! collect" and "read without wait" compile errors.
//!
//! Non-greedy sampling: the `sample_topk_f32` kernel (launched via
//! `SampleTopKLaunch`) does 1/T scale + deterministic top-K' partial
//! selection on device; `sample_from_candidates` finishes on host over
//! <= 1024 floats with a seeded SplitMix64 (`SampleRng`).

//! Speculative-decode greedy accept/reject lives in `spec_accept`.

pub mod dtoh;
pub mod params;
pub mod sampler;
pub mod spec_accept;

pub use dtoh::{DtoHTicket, PinnedTokens};
pub use params::{SamplingParams, KERNEL_K_CAP, MIN_TEMPERATURE};
pub use sampler::{sample_from_candidates, SampleRng, SampleTopKLaunch};
pub use spec_accept::{greedy_accept, Accepted};
