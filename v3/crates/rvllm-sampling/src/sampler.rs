//! Host sampling tail + the `sample_topk_f32` kernel launcher.
//!
//! Per sampled decode step the device runs ONE kernel
//! (`sample_topk_f32_kernel`: 1/T scale + deterministic top-K' partial
//! selection, K' <= 1024) and the host finishes over the compact
//! (value, index) candidates: canonical sort, top-k truncation, softmax,
//! top-p nucleus cut, categorical draw from a seeded SplitMix64. Std-only —
//! no rand crate and no cuRAND. Identical candidate bits and math-library
//! behavior produce the same draw from the same seed.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError};
use rvllm_kernels::KernelFn;

use crate::params::KERNEL_K_CAP;

/// SplitMix64 per-request draw stream with a 64-bit state.
pub struct SampleRng(u64);

impl SampleRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random mantissa bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Draw one token from the device-selected candidates.
///
/// `cands` are (logit/T, token_id) pairs in arbitrary device-arrival order;
/// sorted in place to the canonical (value desc, id asc) order so the result
/// is independent of GPU compaction order. `top_k == 0` and `top_p >= 1.0`
/// disable the respective cut; the nucleus cut keeps at least one token.
/// Softmax runs in f64 over the candidate set (values are already
/// temperature-scaled).
pub fn sample_from_candidates(
    cands: &mut [(f32, u32)],
    top_k: u32,
    top_p: f32,
    rng: &mut SampleRng,
) -> Result<u32> {
    sample_from_candidates_inner(cands, None, top_k, top_p, rng)
}

/// Draw one token while also enforcing the model vocabulary bound.
pub fn sample_from_candidates_in_vocab(
    cands: &mut [(f32, u32)],
    vocab: u32,
    top_k: u32,
    top_p: f32,
    rng: &mut SampleRng,
) -> Result<u32> {
    if vocab == 0 {
        return Err(invalid("vocab", "must be greater than zero"));
    }
    sample_from_candidates_inner(cands, Some(vocab), top_k, top_p, rng)
}

fn sample_from_candidates_inner(
    cands: &mut [(f32, u32)],
    vocab: Option<u32>,
    top_k: u32,
    top_p: f32,
    rng: &mut SampleRng,
) -> Result<u32> {
    if cands.is_empty() {
        return Err(invalid("candidates", "must be non-empty"));
    }
    if cands.iter().any(|(value, _)| !value.is_finite()) {
        return Err(invalid("candidates", "logits must be finite"));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(invalid("top_p", "must be finite and in 0..=1"));
    }
    if top_k > 0 && top_k as usize > cands.len() {
        return Err(invalid("top_k", "must not exceed candidate count"));
    }
    if cands.iter().any(|(_, token)| *token > i32::MAX as u32) {
        return Err(invalid(
            "candidates",
            "token IDs must be non-negative device i32 values",
        ));
    }
    if let Some(vocab) = vocab {
        if cands.iter().any(|(_, token)| *token >= vocab) {
            return Err(invalid("candidates", "token ID must be below vocab"));
        }
    }
    cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut n = cands.len();
    if top_k > 0 {
        n = n.min(top_k as usize);
    }
    let pool = &cands[..n];

    let max = pool[0].0 as f64;
    let weights: Vec<f64> = pool.iter().map(|&(v, _)| (v as f64 - max).exp()).collect();
    let total: f64 = weights.iter().sum();

    // Nucleus cut: smallest prefix of the descending-sorted pool whose
    // probability mass reaches top_p. Compare unnormalized cumsums against
    // top_p * total — same cut, no divisions.
    let mut cut = n;
    if top_p < 1.0 {
        let target = (top_p.max(0.0) as f64) * total;
        let mut cum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cum += w;
            if cum >= target {
                cut = i + 1;
                break;
            }
        }
    }

    let z: f64 = weights[..cut].iter().sum();
    let mut u = rng.next_f64() * z;
    for (i, w) in weights[..cut].iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return Ok(pool[i].1);
        }
    }
    Ok(pool[cut - 1].1)
}

/// Launcher for `sample_topk_f32_kernel` (one f32 logits row).
/// Grid (1,1,1), block (1024,1,1) — see the kernel header for why a single
/// block is the right shape here.
#[derive(Copy, Clone, Debug)]
pub struct SampleTopKLaunch {
    pub vocab: u32,
    pub k_select: u32,
    pub out_capacity: u32,
    pub inv_temp: f32,
}

impl SampleTopKLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 {
            return Err(invalid("vocab", "must be > 0"));
        }
        if self.k_select == 0 || self.k_select > KERNEL_K_CAP {
            return Err(invalid("k_select", "must be in 1..=1024"));
        }
        if self.k_select > self.vocab {
            return Err(invalid("k_select", "must be <= vocab"));
        }
        if self.out_capacity < self.k_select {
            return Err(invalid("out_capacity", "must be >= k_select"));
        }
        if !(self.inv_temp.is_finite() && self.inv_temp > 0.0) {
            return Err(invalid("inv_temp", "must be finite and > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(logits_f32, inv_temp, vocab, k_select, out_vals_f32,
    /// out_idx_i32, out_count_i32)`.
    ///
    /// # Safety
    /// Caller owns the device pointers for the call's duration; `out_vals` /
    /// `out_idx` hold at least `k_select` elements, `out_count` one i32.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        logits_ptr: u64,
        out_vals_ptr: u64,
        out_idx_ptr: u64,
        out_count_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        if logits_ptr == 0 || out_vals_ptr == 0 || out_idx_ptr == 0 || out_count_ptr == 0 {
            return Err(invalid("device pointer", "must be non-null"));
        }
        let mut logits_ptr = logits_ptr;
        let mut inv_temp = self.inv_temp;
        let mut vocab = self.vocab as i32;
        let mut k_select = self.k_select as i32;
        let mut out_vals_ptr = out_vals_ptr;
        let mut out_idx_ptr = out_idx_ptr;
        let mut out_count_ptr = out_count_ptr;
        let args = [
            (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut inv_temp) as *mut f32 as *mut core::ffi::c_void,
            (&mut vocab) as *mut i32 as *mut core::ffi::c_void,
            (&mut k_select) as *mut i32 as *mut core::ffi::c_void,
            (&mut out_vals_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut out_idx_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut out_count_ptr) as *mut u64 as *mut core::ffi::c_void,
        ];
        rvllm_fused::launch_raw(kernel, (1, 1, 1), (1024, 1, 1), 0, stream, &args)
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "sample_topk validate",
            stream: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cands(vals: &[(f32, u32)]) -> Vec<(f32, u32)> {
        vals.to_vec()
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = SampleRng::new(42);
        let mut b = SampleRng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = SampleRng::new(43);
        assert_ne!(SampleRng::new(42).next_u64(), c.next_u64());
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut rng = SampleRng::new(7);
        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut rng = SampleRng::new(0);
        for _ in 0..32 {
            let mut c = cands(&[(1.0, 10), (3.0, 20), (2.0, 30)]);
            assert_eq!(
                sample_from_candidates(&mut c, 1, 1.0, &mut rng).unwrap(),
                20
            );
        }
    }

    #[test]
    fn top_p_zero_keeps_argmax_only() {
        let mut rng = SampleRng::new(1);
        for _ in 0..32 {
            let mut c = cands(&[(0.0, 1), (0.5, 2), (-1.0, 3)]);
            assert_eq!(sample_from_candidates(&mut c, 0, 0.0, &mut rng).unwrap(), 2);
        }
    }

    #[test]
    fn arrival_order_does_not_matter() {
        // Same candidate SET in two different arrival orders + same seed
        // must draw the same token (canonical sort inside).
        let set = [(0.3f32, 7u32), (0.9, 4), (0.9, 11), (-0.2, 5), (0.1, 2)];
        let mut fwd = set.to_vec();
        let mut rev: Vec<_> = set.iter().rev().copied().collect();
        for seed in 0..256u64 {
            let a = sample_from_candidates(&mut fwd, 0, 0.8, &mut SampleRng::new(seed)).unwrap();
            let b = sample_from_candidates(&mut rev, 0, 0.8, &mut SampleRng::new(seed)).unwrap();
            assert_eq!(a, b, "seed {seed}");
        }
    }

    #[test]
    fn empirical_frequency_tracks_softmax() {
        // 3 candidates at T=1: p = softmax([2, 1, 0]) ~ [0.665, 0.245, 0.090].
        let mut counts = [0usize; 3];
        let mut rng = SampleRng::new(123);
        let n = 200_000;
        for _ in 0..n {
            let mut c = cands(&[(2.0, 0), (1.0, 1), (0.0, 2)]);
            counts[sample_from_candidates(&mut c, 0, 1.0, &mut rng).unwrap() as usize] += 1;
        }
        let p = [0.66524096, 0.24472847, 0.09003057];
        for i in 0..3 {
            let freq = counts[i] as f64 / n as f64;
            let rel = (freq - p[i]).abs() / p[i];
            assert!(rel < 0.03, "candidate {i}: freq {freq} vs p {}", p[i]);
        }
    }

    #[test]
    fn nucleus_cut_excludes_tail() {
        // p ~ [0.665, 0.245, 0.090]; top_p=0.7 keeps {0, 1} (0.665 < 0.7).
        let mut rng = SampleRng::new(9);
        for _ in 0..10_000 {
            let mut c = cands(&[(2.0, 0), (1.0, 1), (0.0, 2)]);
            let t = sample_from_candidates(&mut c, 0, 0.7, &mut rng).unwrap();
            assert_ne!(t, 2, "tail token must be cut by top_p=0.7");
        }
    }

    #[test]
    fn launch_validation() {
        let ok = SampleTopKLaunch {
            vocab: 262144,
            k_select: 1024,
            out_capacity: 1024,
            inv_temp: 1.0,
        };
        assert!(ok.validate().is_ok());
        assert!(SampleTopKLaunch { vocab: 0, ..ok }.validate().is_err());
        assert!(SampleTopKLaunch { k_select: 0, ..ok }.validate().is_err());
        assert!(SampleTopKLaunch {
            k_select: 1025,
            ..ok
        }
        .validate()
        .is_err());
        assert!(SampleTopKLaunch {
            inv_temp: 0.0,
            ..ok
        }
        .validate()
        .is_err());
        assert!(SampleTopKLaunch {
            vocab: 512,
            k_select: 1024,
            inv_temp: 1.0,
            out_capacity: 1024,
        }
        .validate()
        .is_err());
        assert!(SampleTopKLaunch {
            out_capacity: 1023,
            ..ok
        }
        .validate()
        .is_err());
    }

    #[test]
    fn candidates_fail_closed() {
        let mut rng = SampleRng::new(0);
        assert!(sample_from_candidates(&mut [], 0, 1.0, &mut rng).is_err());
        let mut nan = [(f32::NAN, 1)];
        assert!(sample_from_candidates(&mut nan, 0, 1.0, &mut rng).is_err());
        let mut one = [(1.0, 1)];
        assert!(sample_from_candidates(&mut one, 2, 1.0, &mut rng).is_err());
        assert!(sample_from_candidates(&mut one, 0, f32::NAN, &mut rng).is_err());
    }

    #[test]
    fn bounded_sampling_rejects_device_index_corruption() {
        let mut rng = SampleRng::new(0);
        let mut negative_cast = [(1.0, (-1_i32) as u32)];
        assert!(sample_from_candidates(&mut negative_cast, 0, 1.0, &mut rng).is_err());

        let mut at_vocab = [(1.0, 8)];
        assert!(sample_from_candidates_in_vocab(&mut at_vocab, 8, 0, 1.0, &mut rng).is_err());

        let mut valid = [(1.0, 7)];
        assert_eq!(
            sample_from_candidates_in_vocab(&mut valid, 8, 0, 1.0, &mut rng).unwrap(),
            7
        );
    }
}
