//! Scheduler per spec 07.
//!
//! Emits one `BatchPlan` per step of exactly one variant (`Prefill`,
//! `Decode`, or `Idle`). No mixed prefill+decode in the same step —
//! that was one of the metadata-coupling sources in v2.

use std::collections::HashSet;

use rvllm_core::{ReqId, Result, RvllmError, SchedulerError, TokenId};

use crate::sched_state::{ReqState, Request};

/// Bucket list for decode. Must match graph-capture buckets.
pub const DECODE_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256];
pub const MAX_REQUESTS: usize = 256;

/// Smallest decode bucket that holds `actual` sequences.
pub fn bucket_for(actual: u32) -> Option<u32> {
    (actual > 0)
        .then(|| DECODE_BUCKETS.iter().copied().find(|&b| b >= actual))
        .flatten()
}

/// Scheduler output for one step.
#[derive(Debug)]
pub enum BatchPlan {
    Idle,
    Prefill {
        req_ids: Vec<ReqId>,
        prompt_tokens_flat: Vec<TokenId>,
        cu_seqlens_q: Vec<u32>,
    },
    Decode {
        req_ids: Vec<ReqId>,
        bucket: u32,
        last_tokens: Vec<TokenId>,
        positions: Vec<u32>,
        context_lens: Vec<u32>,
    },
}

pub struct Scheduler {
    requests: Vec<Request>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            requests: Vec::with_capacity(256),
        }
    }

    pub fn enqueue(&mut self, req: Request) -> Result<()> {
        if self.requests.len() >= MAX_REQUESTS {
            return Err(scheduler_error(SchedulerError::QueueFull, Some(req.id)));
        }
        if self.requests.iter().any(|existing| existing.id == req.id) {
            return Err(scheduler_error(
                SchedulerError::DuplicateRequest,
                Some(req.id),
            ));
        }
        self.requests.push(req);
        Ok(())
    }

    pub fn num_alive(&self) -> usize {
        self.requests.iter().filter(|r| r.is_alive()).count()
    }

    /// Pick the next step's plan. Prefill wins over decode when any
    /// request is in `Queued` or `Prefilling` state.
    pub fn schedule(&self) -> Result<BatchPlan> {
        // Prefill is planned without mutating request state. The transition to
        // decoding happens only after the executor successfully collects it.
        let mut to_prefill: Vec<usize> = Vec::new();
        for (i, r) in self.requests.iter().enumerate() {
            if r.state == ReqState::Queued {
                to_prefill.push(i);
            }
        }
        if !to_prefill.is_empty() {
            let mut req_ids = Vec::with_capacity(to_prefill.len());
            let mut prompt_tokens_flat: Vec<TokenId> = Vec::new();
            let mut cu_seqlens_q: Vec<u32> = Vec::with_capacity(to_prefill.len() + 1);
            cu_seqlens_q.push(0);
            for &i in &to_prefill {
                let r = &self.requests[i];
                req_ids.push(r.id);
                prompt_tokens_flat.extend(r.prompt_tokens.iter().copied());
                cu_seqlens_q.push(u32::try_from(prompt_tokens_flat.len()).map_err(|_| {
                    scheduler_error(
                        SchedulerError::InvalidRequest {
                            reason: "prefill token count exceeds u32".into(),
                        },
                        Some(r.id),
                    )
                })?);
            }
            return Ok(BatchPlan::Prefill {
                req_ids,
                prompt_tokens_flat,
                cu_seqlens_q,
            });
        }

        // Decode: collect Decoding requests into smallest bucket.
        let active: Vec<&Request> = self.requests.iter().filter(|r| r.is_decoding()).collect();
        if active.is_empty() {
            return Ok(BatchPlan::Idle);
        }
        let actual = u32::try_from(active.len()).map_err(|_| {
            scheduler_error(
                SchedulerError::TooManyActive {
                    active: u32::MAX,
                    max: *DECODE_BUCKETS.last().unwrap_or(&0),
                },
                None,
            )
        })?;
        let Some(bucket) = bucket_for(actual) else {
            return Err(scheduler_error(
                SchedulerError::TooManyActive {
                    active: actual,
                    max: *DECODE_BUCKETS.last().unwrap_or(&0),
                },
                None,
            ));
        };
        let mut req_ids = Vec::with_capacity(active.len());
        let mut last_tokens = Vec::with_capacity(active.len());
        let mut positions = Vec::with_capacity(active.len());
        let mut context_lens = Vec::with_capacity(active.len());
        for r in &active {
            req_ids.push(r.id);
            last_tokens.push(
                *r.output_tokens
                    .last()
                    .unwrap_or(&r.prompt_tokens[r.prompt_tokens.len() - 1]),
            );
            let context_len = r.context_len()?;
            positions.push(context_len - 1);
            context_lens.push(context_len);
        }
        Ok(BatchPlan::Decode {
            req_ids,
            bucket,
            last_tokens,
            positions,
            context_lens,
        })
    }

    pub fn commit_prefill(&mut self, req_ids: &[ReqId]) -> Result<()> {
        validate_unique(req_ids)?;
        for &id in req_ids {
            let request = self
                .requests
                .iter()
                .find(|request| request.id == id)
                .ok_or_else(|| scheduler_error(SchedulerError::RequestNotFound, Some(id)))?;
            if request.state != ReqState::Queued {
                return Err(invalid_commit(id, "prefill request is not queued"));
            }
        }
        for &id in req_ids {
            self.requests
                .iter_mut()
                .find(|request| request.id == id)
                .ok_or_else(|| scheduler_error(SchedulerError::RequestNotFound, Some(id)))?
                .state = ReqState::Decoding;
        }
        Ok(())
    }

    /// Commit per-seq outputs only after a completed decode step.
    pub fn commit_decode(&mut self, req_tokens: &[(ReqId, TokenId)]) -> Result<()> {
        let ids: Vec<ReqId> = req_tokens.iter().map(|(id, _)| *id).collect();
        validate_unique(&ids)?;
        for &id in &ids {
            let request = self
                .requests
                .iter()
                .find(|request| request.id == id)
                .ok_or_else(|| scheduler_error(SchedulerError::RequestNotFound, Some(id)))?;
            if !request.is_decoding() {
                return Err(invalid_commit(id, "decode request is not decoding"));
            }
        }
        for &(id, tok) in req_tokens {
            self.requests
                .iter_mut()
                .find(|request| request.id == id)
                .ok_or_else(|| scheduler_error(SchedulerError::RequestNotFound, Some(id)))?
                .push_output(tok)?;
        }
        Ok(())
    }

    pub fn abort(&mut self, id: ReqId) -> bool {
        let Some(request) = self.requests.iter_mut().find(|request| request.id == id) else {
            return false;
        };
        if request.is_alive() {
            request.state = ReqState::Aborted;
            true
        } else {
            false
        }
    }

    pub fn request(&self, id: ReqId) -> Option<&Request> {
        self.requests.iter().find(|request| request.id == id)
    }
}

fn validate_unique(ids: &[ReqId]) -> Result<()> {
    let mut seen = HashSet::with_capacity(ids.len());
    for &id in ids {
        if !seen.insert(id) {
            return Err(invalid_commit(id, "duplicate request id"));
        }
    }
    Ok(())
}

fn invalid_commit(id: ReqId, reason: impl Into<String>) -> RvllmError {
    scheduler_error(
        SchedulerError::InvalidCommit {
            reason: reason.into(),
        },
        Some(id),
    )
}

fn scheduler_error(err: SchedulerError, req_id: Option<ReqId>) -> RvllmError {
    RvllmError::Scheduler { err, req_id }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_rounds_up() {
        assert_eq!(bucket_for(1), Some(1));
        assert_eq!(bucket_for(3), Some(4));
        assert_eq!(bucket_for(100), Some(128));
        assert_eq!(bucket_for(256), Some(256));
        assert_eq!(bucket_for(257), None);
    }

    #[test]
    fn schedule_emits_prefill_then_decode() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4).unwrap())
            .unwrap();
        s.enqueue(Request::new(ReqId(2), vec![TokenId(20), TokenId(21)], 4).unwrap())
            .unwrap();
        let prefill_ids = match s.schedule().unwrap() {
            BatchPlan::Prefill { req_ids, .. } => assert_eq!(req_ids.len(), 2),
            other => panic!("expected Prefill, got {other:?}"),
        };
        let _ = prefill_ids;
        s.commit_prefill(&[ReqId(1), ReqId(2)]).unwrap();
        // After commit of first prefill round, next schedule is Decode.
        match s.schedule().unwrap() {
            BatchPlan::Decode {
                req_ids, bucket, ..
            } => {
                assert_eq!(req_ids.len(), 2);
                assert_eq!(bucket, 2);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_ids_and_decode_commit_before_prefill() {
        let request = Request::new(ReqId(7), vec![TokenId(1)], 1).unwrap();
        let mut scheduler = Scheduler::new();
        scheduler.enqueue(request).unwrap();
        assert!(scheduler
            .enqueue(Request::new(ReqId(7), vec![TokenId(2)], 1).unwrap())
            .is_err());
        assert!(scheduler.commit_decode(&[(ReqId(7), TokenId(3))]).is_err());
    }
}
