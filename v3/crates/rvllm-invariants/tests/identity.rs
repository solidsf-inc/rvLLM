//! Numerical / output-identity invariants.
//!
//! The rest of this crate enforces the build DAG and LoC budgets. This file
//! pins the *behaviour* of the host reference functions that define greedy
//! serving identity, so a refactor that breaks greedy decoding fails loudly in
//! CI rather than silently changing emitted tokens.
//!
//! Host-only: imports the pure-Rust reference fns (`greedy_accept`,
//! `argmax_ref`) — no `cuda` feature, no device. The golden vectors are fixed
//! so the asserts double as a spec the references must keep satisfying.

use rvllm_fused::argmax_ref;
use rvllm_sampling::greedy_accept;

// ---------------------------------------------------------------------------
// rvllm_sampling::greedy_accept — spec-decode greedy accept/reject identity.
//
// Contract (from spec_accept.rs): emit the TARGET argmax at every scanned
// position, stop at the first position where draft != target (still emitting
// that position's target), and append the bonus token only on a clean sweep of
// all K drafts. Emitted length == valid_count; next_token == tokens.last().
// ---------------------------------------------------------------------------

#[test]
fn greedy_accept_stops_at_first_mismatch() {
    // Golden vector: K=5, the draft diverges at position 2. Everything from
    // position 2 onward must be dropped, and the emitted token at the mismatch
    // is the TARGET argmax (30), never the draft token (77).
    let target = [10u32, 20, 30, 40, 50];
    let draft = [10u32, 20, 77, 40, 50];
    let bonus = 999u32;

    let acc = greedy_accept(&target, &draft, bonus).expect("greedy_accept on valid inputs");

    // First mismatch at index 2 -> exactly 3 emitted tokens (indices 0..=2).
    assert_eq!(
        acc.valid_count, 3,
        "first-mismatch-stop must cut at index+1"
    );
    assert_eq!(
        acc.tokens,
        vec![10, 20, 30],
        "emitted prefix must be the TARGET argmaxes through the mismatch",
    );
    assert_eq!(
        acc.next_token, 30,
        "next_token is the target argmax at the mismatch, not the draft (77) or bonus",
    );
    assert!(
        !acc.tokens.contains(&77),
        "the draft-only token must never leak into the emitted sequence",
    );
    // The bonus is appended only on a clean sweep — not here.
    assert!(
        !acc.tokens.contains(&999),
        "bonus must not appear after a mismatch"
    );
}

#[test]
fn greedy_accept_bonus_on_clean_sweep() {
    // Golden vector: K=4, every draft matches the target argmax. The emitted
    // sequence is all K target argmaxes PLUS the bonus token, valid_count K+1.
    let target = [101u32, 202, 303, 404];
    let draft = [101u32, 202, 303, 404];
    let bonus = 555u32;

    let acc = greedy_accept(&target, &draft, bonus).expect("greedy_accept on valid inputs");

    assert_eq!(acc.valid_count, 5, "clean sweep emits K+1 tokens (K=4)");
    assert_eq!(
        acc.tokens,
        vec![101, 202, 303, 404, 555],
        "clean sweep emits the K target argmaxes then the bonus token",
    );
    assert_eq!(
        acc.next_token, 555,
        "after a full accept the next token fed forward is the bonus",
    );
}

// ---------------------------------------------------------------------------
// rvllm_fused::argmax_ref — greedy argmax with LEFT tie-break.
//
// Greedy serving emits `logits.argmax(dim=-1)`, which breaks ties toward the
// LOWEST token id. The reference scans with a strict `>` so the first maximum
// wins. These golden rows pin that tie-break so a refactor to `>=` (which would
// pick the LAST maximum) fails in CI.
// ---------------------------------------------------------------------------

#[test]
fn argmax_ref_left_tiebreak_all_equal() {
    // A single row where every logit is identical: the lowest index (0) wins.
    let logits = [1.5f32, 1.5, 1.5, 1.5];
    let mut out = [-1i32];
    argmax_ref(&logits, 1, 4, &mut out);
    assert_eq!(
        out[0], 0,
        "all-equal row must pick the lowest index (left tie-break)"
    );
}

#[test]
fn argmax_ref_left_tiebreak_two_equal_maxima() {
    // Two equal maxima (9.0) at indices 1 and 3, with smaller values elsewhere.
    // Left tie-break must pick index 1, the first occurrence — not 3.
    let logits = [2.0f32, 9.0, 4.0, 9.0, 1.0];
    let mut out = [-1i32];
    argmax_ref(&logits, 1, 5, &mut out);
    assert_eq!(
        out[0], 1,
        "two equal maxima must resolve to the first (lowest) index, not the last",
    );
}

#[test]
fn argmax_ref_normal_unique_max() {
    // Two rows, each with a single unambiguous maximum, to pin the ordinary
    // (non-tie) path and the per-row independence of the scan.
    let logits = [
        0.1f32, 0.2, 9.9, 0.3, // row 0: max at index 2
        7.0, 0.0, 1.0, 2.0, //    row 1: max at index 0
    ];
    let mut out = [-1i32, -1];
    argmax_ref(&logits, 2, 4, &mut out);
    assert_eq!(out[0], 2, "row 0 argmax is the unique maximum at index 2");
    assert_eq!(out[1], 0, "row 1 argmax is the unique maximum at index 0");
}
