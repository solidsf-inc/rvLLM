//! Request state machine per spec 07.
//!
//! Transitions are explicit: `Queued → Prefilling → Decoding → Finished`.
//! `Aborted` reachable from any state.

use rvllm_core::{ReqId, Result, RvllmError, SchedulerError, TokenId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReqState {
    Queued,
    Prefilling,
    Decoding,
    Finished,
    Aborted,
}

#[derive(Debug)]
pub struct Request {
    pub id: ReqId,
    pub state: ReqState,
    pub prompt_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub max_output_tokens: u32,
}

impl Request {
    pub fn new(id: ReqId, prompt_tokens: Vec<TokenId>, max_output_tokens: u32) -> Result<Self> {
        if prompt_tokens.is_empty() {
            return Err(invalid(id, "prompt must contain at least one token"));
        }
        if max_output_tokens == 0 {
            return Err(invalid(id, "max_output_tokens must be greater than zero"));
        }
        let max_context = prompt_tokens
            .len()
            .checked_add(max_output_tokens as usize)
            .ok_or_else(|| invalid(id, "maximum context length overflow"))?;
        u32::try_from(max_context)
            .map_err(|_| invalid(id, "maximum context length exceeds u32"))?;
        Ok(Self {
            id,
            state: ReqState::Queued,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_output_tokens,
        })
    }

    pub fn is_alive(&self) -> bool {
        !matches!(self.state, ReqState::Finished | ReqState::Aborted)
    }

    pub fn is_decoding(&self) -> bool {
        matches!(self.state, ReqState::Decoding)
    }

    pub fn context_len(&self) -> Result<u32> {
        let len = self
            .prompt_tokens
            .len()
            .checked_add(self.output_tokens.len())
            .ok_or_else(|| invalid(self.id, "context length overflow"))?;
        u32::try_from(len).map_err(|_| invalid(self.id, "context length exceeds u32"))
    }

    pub fn push_output(&mut self, tok: TokenId) -> Result<()> {
        if !self.is_decoding() {
            return Err(invalid(
                self.id,
                "output can only be committed while decoding",
            ));
        }
        if self.output_tokens.len() >= self.max_output_tokens as usize {
            return Err(invalid(self.id, "output budget already exhausted"));
        }
        self.output_tokens.push(tok);
        if self.output_tokens.len() >= self.max_output_tokens as usize {
            self.state = ReqState::Finished;
        }
        Ok(())
    }
}

fn invalid(id: ReqId, reason: impl Into<String>) -> RvllmError {
    RvllmError::Scheduler {
        err: SchedulerError::InvalidRequest {
            reason: reason.into(),
        },
        req_id: Some(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_finishes_at_max_tokens() {
        let mut r = Request::new(ReqId(1), vec![TokenId(0); 4], 3).unwrap();
        r.state = ReqState::Decoding;
        r.push_output(TokenId(1)).unwrap();
        r.push_output(TokenId(2)).unwrap();
        assert!(r.is_decoding());
        r.push_output(TokenId(3)).unwrap();
        assert_eq!(r.state, ReqState::Finished);
        assert!(!r.is_alive());
    }

    #[test]
    fn rejects_empty_prompt_and_zero_budget() {
        assert!(Request::new(ReqId(1), Vec::new(), 1).is_err());
        assert!(Request::new(ReqId(1), vec![TokenId(1)], 0).is_err());
    }
}
