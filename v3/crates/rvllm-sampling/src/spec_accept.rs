//! Greedy accept/reject for speculative decoding.
//!
//! Greedy serving emits the **target model's argmax** at every position. The
//! drafter only proposes; whether a draft token is accepted changes speed, not
//! identity. The implementation follows that public greedy invariant directly:
//!
//!   * scan the K draft tokens against the target argmax at the matching
//!     verify positions,
//!   * accept the longest prefix where `draft[i] == target_argmax[i]`,
//!   * stop at the first mismatch — and at that mismatch position emit the
//!     **target argmax** (not the draft token), so the emitted sequence is
//!     always the greedy target sequence,
//!   * if every draft token matched, append the **bonus** token (the target
//!     argmax computed at the K+1-th / bonus verify position).
//!
//! The verify batch is M = K+1 rows: rows `0..K` are the per-draft positions
//! whose argmax is compared to the K draft tokens, and row K is the bonus
//! position whose argmax is the token to emit after a full accept. The target
//! argmax is taken over raw BF16 logits. `bf16 -> fp32` preserves every BF16
//! value exactly, so argmax matches a reference operating on the same BF16
//! values; ties break left (lowest token id).
//!
//! The fused pass returns, in ONE scan, the accepted token row, the
//! `next_token` (what to feed the next decode/draft step), and `valid_count`
//! (how many tokens were emitted). `valid_count == i+1` on a first mismatch at
//! position `i`; `valid_count == K+1` on a full accept.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError};

/// Result of one request's greedy accept scan.
///
/// `tokens` holds exactly `valid_count` emitted token ids — the accepted
/// prefix of target argmaxes plus the trailing token (the target argmax at the
/// first-unaccepted position, or the bonus token on a full accept). `tokens[0]`
/// is always present (verify always emits at least the row-0 argmax).
/// `next_token == tokens[valid_count - 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accepted {
    /// Emitted token ids (length == `valid_count`). All TARGET argmaxes.
    pub tokens: Vec<u32>,
    /// The token to feed the next step (== `tokens.last()`).
    pub next_token: u32,
    /// Number of tokens emitted this verify step (1..=K+1).
    pub valid_count: u32,
}

/// Pure-Rust reference for the fused greedy accept/reject scan.
///
/// * `target_argmax`: the target model's argmax at verify rows `0..K`
///   (one per draft token), in position order.
/// * `draft_tokens`: the K tokens the drafter proposed.
/// * `bonus_token`: the target argmax at the bonus verify row (row K), emitted
///   only when all K drafts were accepted.
///
/// The input lengths must match. Emit target argmax at each scanned position, stop
/// after the first mismatch (still emitting that position's target argmax),
/// append the bonus only on a clean sweep.
pub fn greedy_accept(
    target_argmax: &[u32],
    draft_tokens: &[u32],
    bonus_token: u32,
) -> Result<Accepted> {
    if target_argmax.len() != draft_tokens.len() {
        return Err(invalid(
            "input lengths",
            "target_argmax and draft_tokens must match",
        ));
    }
    let k = draft_tokens.len();
    if k >= u32::MAX as usize {
        return Err(invalid("k", "output length must fit in u32"));
    }
    let capacity = k
        .checked_add(1)
        .ok_or_else(|| invalid("k", "output capacity overflow"))?;
    let mut tokens = Vec::with_capacity(capacity);
    let mut rejected = false;
    for pos in 0..k {
        // Emit the TARGET argmax at this position — never the draft. This is
        // what makes the emitted sequence the greedy target sequence.
        let target = target_argmax[pos];
        tokens.push(target);
        if draft_tokens[pos] != target {
            rejected = true;
            break; // first mismatch: stop, this position's target is the last token
        }
    }
    if !rejected {
        // Clean sweep over all K drafts: append the bonus (target argmax at the
        // K+1-th verify row).
        tokens.push(bonus_token);
    }
    let valid_count = u32::try_from(tokens.len())
        .map_err(|_| invalid("output", "valid_count does not fit in u32"))?;
    let next_token = *tokens
        .last()
        .ok_or_else(|| invalid("output", "must contain at least the bonus token"))?;
    Ok(Accepted {
        tokens,
        next_token,
        valid_count,
    })
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "spec_accept validate",
            stream: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_accept_k7_emits_eight_with_bonus() {
        // All 7 drafts match the target argmax -> 7 accepted + 1 bonus = 8.
        let target = [10, 11, 12, 13, 14, 15, 16];
        let draft = [10, 11, 12, 13, 14, 15, 16];
        let bonus = 99;
        let acc = greedy_accept(&target, &draft, bonus).unwrap();
        assert_eq!(acc.valid_count, 8);
        assert_eq!(acc.tokens, vec![10, 11, 12, 13, 14, 15, 16, 99]);
        assert_eq!(acc.next_token, 99);
    }

    #[test]
    fn first_mismatch_at_i_emits_i_plus_one_target_tokens() {
        // Mismatch at position 3: emit target argmaxes [0..=3], count = 4, no bonus.
        let target = [10, 11, 12, 99, 14, 15, 16];
        let draft = [10, 11, 12, 77, 14, 15, 16];
        let bonus = 500;
        let acc = greedy_accept(&target, &draft, bonus).unwrap();
        assert_eq!(acc.valid_count, 4);
        // Emitted tokens are the TARGET argmaxes of the accepted prefix +
        // the target argmax at the mismatch position (NOT the draft 77).
        assert_eq!(acc.tokens, vec![10, 11, 12, 99]);
        assert_eq!(acc.next_token, 99);
    }

    #[test]
    fn mismatch_at_position_zero_emits_one_target_token() {
        let target = [42, 11, 12];
        let draft = [7, 11, 12];
        let acc = greedy_accept(&target, &draft, 999).unwrap();
        assert_eq!(acc.valid_count, 1);
        assert_eq!(acc.tokens, vec![42]);
        assert_eq!(acc.next_token, 42);
    }

    #[test]
    fn mismatch_at_last_draft_position() {
        // K=7, all match except the last -> count 7, no bonus, last = target[6].
        let target = [10, 11, 12, 13, 14, 15, 16];
        let draft = [10, 11, 12, 13, 14, 15, 77];
        let acc = greedy_accept(&target, &draft, 99).unwrap();
        assert_eq!(acc.valid_count, 7);
        assert_eq!(acc.tokens, vec![10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(acc.next_token, 16);
    }

    #[test]
    fn emitted_tokens_are_always_target_never_draft() {
        // Even on a matching position the emitted token is the target value;
        // construct a case where draft==target so the distinction is exercised
        // only at the mismatch (covered above) — here assert the type contract:
        // every emitted token comes from `target`/`bonus`, never `draft`.
        let target = [1, 2, 3, 4, 5, 6, 7];
        let draft = [1, 2, 9, 4, 5, 6, 7]; // mismatch at 2
        let acc = greedy_accept(&target, &draft, 100).unwrap();
        for (i, &t) in acc.tokens.iter().enumerate() {
            assert_eq!(t, target[i], "emitted token {i} must be the target argmax");
        }
        assert!(
            !acc.tokens.contains(&9),
            "draft-only token leaked into output"
        );
    }

    #[test]
    fn k1_full_accept_emits_two() {
        let acc = greedy_accept(&[5], &[5], 6).unwrap();
        assert_eq!(acc.valid_count, 2);
        assert_eq!(acc.tokens, vec![5, 6]);
        assert_eq!(acc.next_token, 6);
    }

    #[test]
    fn k1_reject_emits_one() {
        let acc = greedy_accept(&[5], &[4], 6).unwrap();
        assert_eq!(acc.valid_count, 1);
        assert_eq!(acc.tokens, vec![5]);
        assert_eq!(acc.next_token, 5);
    }

    #[test]
    fn valid_count_matches_kernel_semantics_across_all_mismatch_positions() {
        // Golden: for each first-mismatch position i in 0..K, valid_count == i+1
        // and emitted == target[0..=i]; full accept -> count K+1 with bonus.
        let target: [u32; 7] = [20, 21, 22, 23, 24, 25, 26];
        let bonus = 777;
        for mismatch in 0..7usize {
            let mut draft = target;
            draft[mismatch] = 9999; // force a mismatch exactly at `mismatch`
            let acc = greedy_accept(&target, &draft, bonus).unwrap();
            assert_eq!(acc.valid_count, (mismatch + 1) as u32);
            assert_eq!(acc.tokens, target[..=mismatch].to_vec());
            assert_eq!(acc.next_token, target[mismatch]);
        }
        // Full accept.
        let acc = greedy_accept(&target, &target, bonus).unwrap();
        assert_eq!(acc.valid_count, 8);
        assert_eq!(*acc.tokens.last().unwrap(), bonus);
    }

    #[test]
    fn empty_and_mismatched_inputs_are_safe() {
        let empty = greedy_accept(&[], &[], 9).unwrap();
        assert_eq!(empty.tokens, vec![9]);
        assert_eq!(empty.next_token, 9);
        assert_eq!(empty.valid_count, 1);
        assert!(greedy_accept(&[1], &[], 9).is_err());
    }
}
