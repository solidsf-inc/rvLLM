//! Input plan that feeds `pack::upload`.
//!
//! Decoupled from the scheduler so this crate doesn't depend upstream.
//! The scheduler (`rvllm-runtime::scheduler`) populates one of these and
//! hands it to the metadata layer.

use rvllm_core::{BlockId, TokenId};

/// One decoded step's worth of scheduler output, in the shape the
/// metadata packer expects. All slices are borrowed from scheduler-owned
/// storage; this struct never allocates.
#[derive(Debug)]
pub struct BatchPlan<'s> {
    /// Number of active sequences in this step (≤ bucket).
    pub num_seqs: u32,
    /// Per-seq current token (to be re-embedded). Padded with 0 to bucket.
    pub token_ids: &'s [TokenId],
    /// Per-seq position in its sequence.
    pub positions: &'s [u32],
    /// Per-seq current context length (# of valid KV tokens).
    pub context_lens: &'s [u32],
    /// Per-seq row-major block table, flattened. `block_tables.len() == num_seqs * max_blocks_input`.
    pub block_tables_flat: &'s [BlockId],
    pub max_blocks_input: u32,
    /// Per-seq KV slot for the new token. -1 for padded slots.
    pub slot_mapping: &'s [i32],
    /// `[0, 1, 2, ..., num_seqs]` for decode; scheduler fills in for
    /// prefill with per-seq query lengths.
    pub seq_start_pos: &'s [u32],
}

impl<'s> BatchPlan<'s> {
    /// Sanity-check that the plan fits the layout's bucket.
    pub fn fits_layout(&self, layout: &crate::layout::MetadataLayout) -> bool {
        let Ok(num_seqs) = usize::try_from(self.num_seqs) else {
            return false;
        };
        let Ok(max_blocks_input) = usize::try_from(self.max_blocks_input) else {
            return false;
        };
        let Some(block_table_elements) = num_seqs.checked_mul(max_blocks_input) else {
            return false;
        };
        let Some(seq_start_len) = num_seqs.checked_add(1) else {
            return false;
        };
        self.num_seqs > 0
            && self.num_seqs <= layout.bucket
            && self.max_blocks_input > 0
            && self.max_blocks_input <= layout.max_blocks
            && self.token_ids.len() == num_seqs
            && self.positions.len() == num_seqs
            && self.context_lens.len() == num_seqs
            && self.slot_mapping.len() == num_seqs
            && self.block_tables_flat.len() == block_table_elements
            && self.seq_start_pos.len() == seq_start_len
            && self.seq_start_pos.first() == Some(&0)
            && self
                .seq_start_pos
                .windows(2)
                .all(|window| window[0] < window[1])
            && self
                .seq_start_pos
                .last()
                .is_some_and(|&value| value <= i32::MAX as u32)
            && self
                .token_ids
                .iter()
                .all(|token| token.0 <= i32::MAX as u32)
            && self.positions.iter().all(|&value| value <= i32::MAX as u32)
            && self
                .context_lens
                .iter()
                .all(|&value| value > 0 && value <= i32::MAX as u32)
            && self
                .block_tables_flat
                .iter()
                .all(|block| block.0 <= i32::MAX as u32)
            && self.slot_mapping.iter().all(|&slot| slot >= 0)
    }
}
