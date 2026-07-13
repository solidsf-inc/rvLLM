//! `MetadataLayout`: the frozen per-bucket packed-buffer layout.
//!
//! Keyed on `(bucket, max_blocks_per_seq)`. Computed once at engine
//! init for every bucket in the graph-capture set, stored as a
//! `BTreeMap<(bucket, max_blocks), MetadataLayout>`. Captured graphs
//! bind the exact device offsets in this struct; replays write into
//! those offsets. There is NO second layout — prefill and decode have
//! separate entry points in `rvllm-runtime` that each produce their
//! own `BatchPlan`, and each plan goes through exactly one upload path
//! (`pack::upload`) keyed by its layout.

use rvllm_core::{ConfigError, MetaLayoutHash, Result, RvllmError};
use sha2::{Digest, Sha256};

/// Byte offsets (in i32 elements, not bytes) into the packed metadata
/// buffer. Every field is padded to `bucket` entries; `block_tables`
/// is `bucket * max_blocks`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MetadataLayout {
    pub bucket: u32,
    pub max_blocks: u32,
    /// Offsets into the packed buffer (units = i32 elements).
    pub token_ids_off: u32,
    pub positions_off: u32,
    pub context_lens_off: u32,
    pub block_tables_off: u32,
    pub slot_mapping_off: u32,
    pub seq_start_pos_off: u32,
    /// Total length of the packed buffer (i32 elements).
    pub total_elements: u32,
}

impl MetadataLayout {
    /// Compute the canonical layout for a given bucket + max_blocks.
    pub fn compute(bucket: u32, max_blocks: u32) -> Result<Self> {
        if bucket == 0 || max_blocks == 0 {
            return Err(invalid("bucket and max_blocks must be greater than zero"));
        }
        let token_ids_off = 0u32;
        let positions_off = token_ids_off
            .checked_add(bucket)
            .ok_or_else(|| invalid("positions offset overflow"))?;
        let context_lens_off = positions_off
            .checked_add(bucket)
            .ok_or_else(|| invalid("context lengths offset overflow"))?;
        let block_tables_off = context_lens_off
            .checked_add(bucket)
            .ok_or_else(|| invalid("block tables offset overflow"))?;
        let block_table_elements = bucket
            .checked_mul(max_blocks)
            .ok_or_else(|| invalid("block table extent overflow"))?;
        let slot_mapping_off = block_tables_off
            .checked_add(block_table_elements)
            .ok_or_else(|| invalid("slot mapping offset overflow"))?;
        let seq_start_pos_off = slot_mapping_off
            .checked_add(bucket)
            .ok_or_else(|| invalid("sequence starts offset overflow"))?;
        // seq_start_pos is bucket+1 entries (prefix sums + total).
        let total_elements = seq_start_pos_off
            .checked_add(bucket)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("metadata extent overflow"))?;
        Ok(Self {
            bucket,
            max_blocks,
            token_ids_off,
            positions_off,
            context_lens_off,
            block_tables_off,
            slot_mapping_off,
            seq_start_pos_off,
            total_elements,
        })
    }

    /// sha256 of the layout descriptor. Captured graphs carry this
    /// hash so replay can assert the bucket's layout hasn't drifted.
    pub fn hash(&self) -> MetaLayoutHash {
        let mut h = Sha256::new();
        h.update(self.bucket.to_le_bytes());
        h.update(self.max_blocks.to_le_bytes());
        h.update(self.token_ids_off.to_le_bytes());
        h.update(self.positions_off.to_le_bytes());
        h.update(self.context_lens_off.to_le_bytes());
        h.update(self.block_tables_off.to_le_bytes());
        h.update(self.slot_mapping_off.to_le_bytes());
        h.update(self.seq_start_pos_off.to_le_bytes());
        h.update(self.total_elements.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        MetaLayoutHash(out)
    }

    /// Bytes needed to hold the packed buffer.
    pub fn bytes(&self) -> Result<usize> {
        usize::try_from(self.total_elements)
            .ok()
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<i32>()))
            .ok_or_else(|| invalid("metadata byte extent overflow"))
    }
}

fn invalid(reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "metadata_layout",
            reason: reason.into(),
        },
        "metadata_layout",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_deterministic_for_bucket_maxblocks() {
        let a = MetadataLayout::compute(128, 129).unwrap();
        let b = MetadataLayout::compute(128, 129).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn layout_differs_across_buckets() {
        let a = MetadataLayout::compute(1, 129).unwrap();
        let b = MetadataLayout::compute(128, 129).unwrap();
        assert_ne!(a.hash(), b.hash());
        // block_tables_off scales with bucket
        assert!(b.block_tables_off > a.block_tables_off);
    }

    #[test]
    fn qwen_decode_128_fits_in_under_100kb() {
        let l = MetadataLayout::compute(128, 129).unwrap();
        assert!(l.bytes().unwrap() < 100 * 1024);
    }

    #[test]
    fn rejects_zero_and_overflowing_layouts() {
        assert!(MetadataLayout::compute(0, 1).is_err());
        assert!(MetadataLayout::compute(1, 0).is_err());
        assert!(MetadataLayout::compute(u32::MAX, u32::MAX).is_err());
    }
}
