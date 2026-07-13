//! Type-state launch/collect engine boundary.

use rvllm_core::{ConfigError, ReqId, Result, RvllmError, TokenId};

use crate::scheduler::{BatchPlan, Scheduler};

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub req_id: ReqId,
    pub new_token: TokenId,
    pub finished: bool,
}

/// Backend-owned work launched for one scheduler plan.
pub trait StepTicket: Send {
    /// Wait for completion and return one token per decode request. Prefill
    /// tickets return an empty vector.
    fn collect(self: Box<Self>) -> Result<Vec<(ReqId, TokenId)>>;

    /// Cancel or fence abandoned work. Implementations must leave resources
    /// safe to reuse before returning.
    fn cancel(self: Box<Self>) -> Result<()>;
}

/// Compute backend used by the type-state scheduler shell.
pub trait StepExecutor: Send {
    fn launch(&mut self, plan: &BatchPlan) -> Result<Box<dyn StepTicket>>;
}

pub struct Engine {
    pub scheduler: Scheduler,
    executor: Option<Box<dyn StepExecutor>>,
}

impl Engine {
    /// Construct an engine without a compute backend. Idle steps are usable;
    /// non-idle launches fail closed until an executor is installed.
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            executor: None,
        }
    }

    pub fn with_executor(executor: impl StepExecutor + 'static) -> Self {
        Self {
            scheduler: Scheduler::new(),
            executor: Some(Box::new(executor)),
        }
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.num_alive() > 0
    }

    pub fn step_launch(&mut self) -> Result<PendingStep<'_>> {
        let plan = self.scheduler.schedule()?;
        let ticket = match plan {
            BatchPlan::Idle => None,
            _ => Some(
                self.executor
                    .as_mut()
                    .ok_or_else(executor_missing)?
                    .launch(&plan)?,
            ),
        };
        Ok(PendingStep {
            engine: self,
            plan: Some(plan),
            ticket,
        })
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use = "PendingStep must be collected or cancelled"]
pub struct PendingStep<'e> {
    engine: &'e mut Engine,
    plan: Option<BatchPlan>,
    ticket: Option<Box<dyn StepTicket>>,
}

impl<'e> PendingStep<'e> {
    pub fn plan(&self) -> Option<&BatchPlan> {
        self.plan.as_ref()
    }

    pub fn collect(mut self) -> Result<Vec<StepOutput>> {
        let plan = self.plan.take().ok_or_else(invalid_pending_step)?;
        let executor_output = match self.ticket.take() {
            Some(ticket) => ticket.collect()?,
            None if matches!(plan, BatchPlan::Idle) => Vec::new(),
            None => return Err(invalid_pending_step()),
        };

        match plan {
            BatchPlan::Idle => {
                if !executor_output.is_empty() {
                    return Err(invalid_executor_output("idle step returned tokens"));
                }
                Ok(Vec::new())
            }
            BatchPlan::Prefill { req_ids, .. } => {
                if !executor_output.is_empty() {
                    return Err(invalid_executor_output("prefill step returned tokens"));
                }
                self.engine.scheduler.commit_prefill(&req_ids)?;
                Ok(Vec::new())
            }
            BatchPlan::Decode { req_ids, .. } => {
                if executor_output.len() != req_ids.len()
                    || executor_output
                        .iter()
                        .zip(&req_ids)
                        .any(|((actual, _), expected)| actual != expected)
                {
                    return Err(invalid_executor_output(
                        "decode output ids must exactly match scheduled order",
                    ));
                }
                self.engine.scheduler.commit_decode(&executor_output)?;
                executor_output
                    .into_iter()
                    .map(|(req_id, new_token)| {
                        let request =
                            self.engine.scheduler.request(req_id).ok_or_else(|| {
                                invalid_executor_output("committed request missing")
                            })?;
                        Ok(StepOutput {
                            req_id,
                            new_token,
                            finished: !request.is_alive(),
                        })
                    })
                    .collect()
            }
        }
    }

    pub fn cancel(mut self) -> Result<()> {
        self.plan.take();
        if let Some(ticket) = self.ticket.take() {
            ticket.cancel()?;
        }
        Ok(())
    }
}

impl Drop for PendingStep<'_> {
    fn drop(&mut self) {
        if self.plan.take().is_some() {
            if let Some(ticket) = self.ticket.take() {
                if let Err(error) = ticket.cancel() {
                    tracing::error!(%error, "failed to cancel dropped pending step");
                }
            }
        }
    }
}

fn executor_missing() -> RvllmError {
    RvllmError::config(
        ConfigError::MissingField {
            name: "step_executor",
        },
        "step_executor",
    )
}

fn invalid_pending_step() -> RvllmError {
    invalid_executor_output("pending step has already been drained")
}

fn invalid_executor_output(reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "step_executor",
            reason: reason.into(),
        },
        "step_executor",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched_state::Request;

    struct ImmediateExecutor;
    struct ImmediateTicket(Vec<(ReqId, TokenId)>);

    impl StepExecutor for ImmediateExecutor {
        fn launch(&mut self, plan: &BatchPlan) -> Result<Box<dyn StepTicket>> {
            let output = match plan {
                BatchPlan::Decode { req_ids, .. } => {
                    req_ids.iter().map(|&id| (id, TokenId(42))).collect()
                }
                _ => Vec::new(),
            };
            Ok(Box::new(ImmediateTicket(output)))
        }
    }

    impl StepTicket for ImmediateTicket {
        fn collect(self: Box<Self>) -> Result<Vec<(ReqId, TokenId)>> {
            Ok(self.0)
        }

        fn cancel(self: Box<Self>) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn empty_engine_has_no_pending_work() {
        let mut engine = Engine::new();
        assert!(!engine.has_pending_work());
        assert!(engine.step_launch().unwrap().collect().unwrap().is_empty());
    }

    #[test]
    fn non_idle_engine_without_executor_fails_closed() {
        let mut engine = Engine::new();
        engine
            .scheduler
            .enqueue(Request::new(ReqId(1), vec![TokenId(0)], 1).unwrap())
            .unwrap();
        assert!(engine.step_launch().is_err());
        assert_eq!(
            engine.scheduler.request(ReqId(1)).unwrap().state,
            crate::sched_state::ReqState::Queued
        );
    }

    #[test]
    fn successful_collect_commits_after_execution() {
        let mut engine = Engine::with_executor(ImmediateExecutor);
        engine
            .scheduler
            .enqueue(Request::new(ReqId(1), vec![TokenId(0)], 1).unwrap())
            .unwrap();
        assert!(engine.step_launch().unwrap().collect().unwrap().is_empty());
        let output = engine.step_launch().unwrap().collect().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].new_token, TokenId(42));
        assert!(output[0].finished);
    }
}
