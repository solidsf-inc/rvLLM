//! KV-cache layout: `[2, num_blocks, block_size, num_kv_heads, head_dim]`.
//!
//! K at offset 0, V at `num_blocks * block_size * num_kv_heads * head_dim`.
//! This matches the FA3 paged-decode page-table descriptor byte-for-byte,
//! so the kernel never does pointer math beyond `block_table[i]`.

use rvllm_core::{ConfigError, DType, Result, RvllmError};

/// Per-layer KV layout. One instance per model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvLayout {
    pub num_blocks: u32,
    pub block_size: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

impl KvLayout {
    /// Bytes per block (K *or* V, one of the two).
    pub fn block_bytes(&self) -> Result<usize> {
        self.validate()?;
        [
            self.block_size as usize,
            self.num_kv_heads as usize,
            self.head_dim as usize,
            self.dtype.bytes(),
        ]
        .into_iter()
        .try_fold(1usize, checked_mul)
    }

    /// Bytes for one layer (K + V combined). The factor-of-two is the
    /// first axis of the layout.
    pub fn layer_bytes(&self) -> Result<usize> {
        checked_mul(2, self.num_blocks as usize).and_then(|n| checked_mul(n, self.block_bytes()?))
    }

    /// Byte offset of the start of V within a layer (K starts at 0).
    pub fn v_offset(&self) -> Result<usize> {
        checked_mul(self.num_blocks as usize, self.block_bytes()?)
    }

    /// Row-major strides in elements (not bytes), by axis index:
    /// `[K_or_V, block, token_in_block, kv_head, head_dim]`.
    pub fn strides(&self) -> Result<[usize; 5]> {
        self.validate()?;
        let d = self.head_dim as usize;
        let h = self.num_kv_heads as usize;
        let b = self.block_size as usize;
        let n = self.num_blocks as usize;
        let hd = checked_mul(h, d)?;
        let bhd = checked_mul(b, hd)?;
        Ok([checked_mul(n, bhd)?, bhd, hd, d, 1])
    }

    fn validate(&self) -> Result<()> {
        if self.num_blocks == 0
            || self.block_size == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(invalid_layout("all KV-cache dimensions must be > 0"));
        }
        Ok(())
    }
}

fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid_layout("KV-cache size arithmetic overflow"))
}

fn invalid_layout(reason: &str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "kv_layout",
            reason: reason.into(),
        },
        "kv_layout",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_kv() -> KvLayout {
        KvLayout {
            num_blocks: 1024,
            block_size: 64,
            num_kv_heads: 4,
            head_dim: 128,
            dtype: DType::F16,
        }
    }

    #[test]
    fn sizes_round_trip() {
        let l = qwen_kv();
        // One block: 64 tokens * 4 heads * 128 dim * 2 bytes = 65536
        assert_eq!(l.block_bytes().unwrap(), 64 * 4 * 128 * 2);
        // One layer: 2 * 1024 blocks * 65536 bytes = 128 MiB
        assert_eq!(l.layer_bytes().unwrap(), 2 * 1024 * 65536);
        assert_eq!(l.v_offset().unwrap(), 1024 * 65536);
    }

    #[test]
    fn strides_are_row_major() {
        let l = qwen_kv();
        let s = l.strides().unwrap();
        // fastest axis is head_dim
        assert_eq!(s[4], 1);
        assert_eq!(s[3], 128);
        assert_eq!(s[2], 128 * 4);
        assert_eq!(s[1], 128 * 4 * 64);
        assert_eq!(s[0], 128 * 4 * 64 * 1024);
    }
}
