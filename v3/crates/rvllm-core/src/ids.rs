//! Identifier newtypes used across the stack.
//!
//! Newtypes prevent mixing e.g. block ids with token ids at the type
//! level. All ids are `Copy`, `Eq`, `Hash` and cheap to pass by value.

use core::fmt;

macro_rules! id_newtype {
    ($(#[$doc:meta])* $name:ident($inner:ty)) => {
        $(#[$doc])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            #[inline]
            pub const fn raw(self) -> $inner { self.0 }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

id_newtype!(
    /// Identifier of an in-flight request assigned by the engine.
    ReqId(u64)
);
id_newtype!(
    /// Identifier of a sequence (one request may have multiple, e.g. beam).
    SeqId(u64)
);
id_newtype!(
    /// Identifier of a KV-cache block in the paged allocator.
    BlockId(u32)
);
id_newtype!(
    /// Vocabulary token id. Always non-negative.
    ///
    /// Use `u32` rather than `i32` — `-1` sentinels belong to slot_mapping,
    /// not to token ids, and mixing them at the kernel boundary caused
    /// real bugs in v2.
    TokenId(u32)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_are_copy_eq_hash() {
        fn assert_all<T: Copy + Eq + std::hash::Hash>() {}
        assert_all::<ReqId>();
        assert_all::<SeqId>();
        assert_all::<BlockId>();
        assert_all::<TokenId>();
    }

    #[test]
    fn display_matches_inner() {
        assert_eq!(format!("{}", ReqId(42)), "42");
        assert_eq!(format!("{:?}", ReqId(42)), "ReqId(42)");
    }

    #[test]
    fn different_ids_are_not_interchangeable() {
        // This test documents intent; the compiler enforces it.
        let _req = ReqId(1);
        // let _seq: SeqId = _req; // would not compile
    }
}
