//! Per-request sampling parameters.
//!
//! `temperature == 0.0` means greedy. Sampled mode requires an explicit
//! `top_k` when the vocabulary is larger than the device candidate capacity;
//! rvLLM never presents a truncated distribution as an exact full-vocabulary
//! draw.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError};

pub const MIN_TEMPERATURE: f32 = 1e-3;
pub const DEFAULT_TOP_K: u32 = 50;
pub const KERNEL_K_CAP: u32 = 1024;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// Zero, or a non-negative value below `MIN_TEMPERATURE`, selects greedy.
    pub temperature: f32,
    /// Zero requests an exact full-vocabulary draw. Large vocabularies require
    /// an explicit value no greater than `KERNEL_K_CAP`.
    pub top_k: u32,
    /// Nucleus cutoff in `0..=1`; `1` disables the cutoff.
    pub top_p: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }

    pub const fn sampled(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k: DEFAULT_TOP_K,
            top_p,
            seed,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature.is_finite()
            && self.temperature >= 0.0
            && self.temperature < MIN_TEMPERATURE
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid("temperature", "must be finite and >= 0"));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(invalid("top_p", "must be finite and in 0..=1"));
        }
        if self.top_k > KERNEL_K_CAP {
            return Err(invalid("top_k", "must be 0 or <= KERNEL_K_CAP"));
        }
        Ok(())
    }

    pub fn kernel_k(&self, vocab: u32) -> Result<u32> {
        self.validate()?;
        if vocab == 0 {
            return Err(invalid("vocab", "must be > 0"));
        }
        match self.top_k {
            0 if vocab <= KERNEL_K_CAP => Ok(vocab),
            0 => Err(invalid(
                "top_k",
                "set an explicit top_k <= KERNEL_K_CAP for this vocabulary",
            )),
            k if k <= vocab => Ok(k),
            _ => Err(invalid("top_k", "must be <= vocab")),
        }
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "sampling params",
            stream: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_semantics_are_explicit() {
        assert!(SamplingParams::greedy().is_greedy());
        let mut p = SamplingParams::greedy();
        p.temperature = 0.7;
        assert!(!p.is_greedy());
        p.temperature = 1e-6;
        assert!(p.is_greedy());
    }

    #[test]
    fn full_distribution_must_fit() {
        let mut p = SamplingParams::greedy();
        assert_eq!(p.kernel_k(512).unwrap(), 512);
        assert!(p.kernel_k(262_144).is_err());
        p.top_k = 50;
        assert_eq!(p.kernel_k(262_144).unwrap(), 50);
        p.top_k = 4096;
        assert!(p.kernel_k(262_144).is_err());
    }

    #[test]
    fn sampled_default_is_explicitly_bounded() {
        let p = SamplingParams::sampled(0.7, 1.0, 42);
        assert_eq!(p.top_k, DEFAULT_TOP_K);
        assert_eq!(p.kernel_k(262_144).unwrap(), DEFAULT_TOP_K);
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_params() {
        let mut p = SamplingParams::greedy();
        p.temperature = f32::NAN;
        assert!(p.validate().is_err());
        p.temperature = -1.0;
        assert!(p.validate().is_err());
        p.temperature = 0.7;
        p.top_p = f32::INFINITY;
        assert!(p.validate().is_err());
        p.top_p = 1.1;
        assert!(p.validate().is_err());
    }
}
