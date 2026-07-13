//! Authenticated autotune policy loading and shape dispatch.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use rvllm_core::{ConfigError, CutlassCtx, CutlassError, DType, IoError, Result, RvllmError};

use crate::variants::{canonical_variants, VariantDescriptor, VariantId};

const MAX_POLICY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_VARIANTS: usize = 1024;
const MAX_ENTRIES: usize = 65_536;
const MAX_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn deserialize_unique_entries<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, PolicyEntry>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueEntries;

    impl<'de> Visitor<'de> for UniqueEntries {
        type Value = BTreeMap<String, PolicyEntry>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map with unique shape keys")
        }

        fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut entries = BTreeMap::new();
            while let Some((key, entry)) = map.next_entry::<String, PolicyEntry>()? {
                if entries.insert(key.clone(), entry).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate policy entry {key:?}"
                    )));
                }
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(UniqueEntries)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyEntry {
    pub variant: VariantId,
    pub workspace_bytes: u64,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum GemmMode {
    Plain,
    Residual,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[serde(deny_unknown_fields)]
pub struct ShapeKey {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: DType,
    pub mode: GemmMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub revision: String,
    pub arch: String,
    pub variants: Vec<VariantDescriptor>,
    #[serde(deserialize_with = "deserialize_unique_entries")]
    pub entries: BTreeMap<String, PolicyEntry>,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Self> {
        let digest = required_env("RVLLM_POLICY_SHA256")?;
        let revision = required_env("RVLLM_RELEASE_REVISION")?;
        let arch = required_env("RVLLM_KERNEL_ARCH")?;
        Self::load_trusted(path, &digest, &revision, &arch)
    }

    pub fn load_trusted(
        path: &Path,
        expected_sha256: &str,
        expected_revision: &str,
        expected_arch: &str,
    ) -> Result<Self> {
        validate_digest(expected_sha256)?;
        let body = read_bounded(path)?;
        let actual = sha256_hex(&body);
        if actual != expected_sha256 {
            return Err(invalid_policy(format!(
                "policy digest {actual} does not match trusted pin {expected_sha256}"
            )));
        }
        let policy: Policy = serde_json::from_slice(&body)
            .map_err(|e| invalid_policy(format!("policy.json is not valid JSON: {e}")))?;
        if policy.revision != expected_revision {
            return Err(invalid_policy(format!(
                "policy revision {:?} does not match trusted revision {:?}",
                policy.revision, expected_revision
            )));
        }
        if policy.arch != expected_arch {
            return Err(invalid_policy(format!(
                "policy architecture {:?} does not match selected architecture {:?}",
                policy.arch, expected_arch
            )));
        }
        policy.validate_catalog(false)?;
        Ok(policy)
    }

    /// Validate an in-memory policy before using it for dispatch.
    pub fn validate(&self) -> Result<()> {
        self.validate_catalog(true)
    }

    fn validate_catalog(&self, allow_empty: bool) -> Result<()> {
        if self.variants.len() > MAX_VARIANTS || self.entries.len() > MAX_ENTRIES {
            return Err(invalid_policy(format!(
                "policy exceeds limits ({MAX_VARIANTS} variants, {MAX_ENTRIES} entries)"
            )));
        }
        if !allow_empty && (self.variants.is_empty() || self.entries.is_empty()) {
            return Err(invalid_policy(
                "authenticated policy must contain variants and entries".into(),
            ));
        }
        if self.revision.is_empty() || self.revision.len() > 64 {
            return Err(invalid_policy("invalid policy revision".into()));
        }
        if !matches!(
            self.arch.as_str(),
            "sm_80" | "sm_89" | "sm_90" | "sm_100" | "sm_121"
        ) {
            return Err(invalid_policy(format!(
                "unsupported policy architecture {:?}",
                self.arch
            )));
        }

        let canonical: BTreeMap<_, _> = canonical_variants()
            .into_iter()
            .map(|variant| (variant.id, variant))
            .collect();
        let mut ids = BTreeSet::new();
        for variant in &self.variants {
            if !ids.insert(variant.id) {
                return Err(invalid_policy(format!(
                    "duplicate variant id {}",
                    variant.id.0
                )));
            }
            if !variant.validate() {
                return Err(invalid_policy(format!(
                    "variant {} has invalid geometry or schedules",
                    variant.id.0
                )));
            }
            if canonical.get(&variant.id) != Some(variant) {
                return Err(invalid_policy(format!(
                    "variant {} does not match the compiled catalog",
                    variant.id.0
                )));
            }
        }

        for (key, entry) in &self.entries {
            parse_entry_key(key)?;
            if !ids.contains(&entry.variant) {
                return Err(invalid_policy(format!(
                    "entry {key:?} references unknown variant {}",
                    entry.variant.0
                )));
            }
            if entry.workspace_bytes > MAX_WORKSPACE_BYTES
                || usize::try_from(entry.workspace_bytes).is_err()
            {
                return Err(invalid_policy(format!(
                    "entry {key:?} has unsupported workspace size {}",
                    entry.workspace_bytes
                )));
            }
        }
        Ok(())
    }

    pub fn lookup(&self, m: usize, n: usize, k: usize, dtype: DType) -> Result<&PolicyEntry> {
        self.lookup_mode(m, n, k, dtype, GemmMode::Plain)
    }

    pub fn lookup_residual(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
    ) -> Result<&PolicyEntry> {
        self.lookup_mode(m, n, k, dtype, GemmMode::Residual)
    }

    fn lookup_mode(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        mode: GemmMode,
    ) -> Result<&PolicyEntry> {
        validate_shape(m, n, k, dtype)?;
        let key = Self::entry_key_for(m, n, k, dtype, mode);
        self.entries.get(&key).ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::AutotuneCacheMiss { m, n, k, dtype },
                CutlassCtx {
                    kernel: "Policy::lookup",
                    stream: 0,
                },
            )
        })
    }

    pub fn entry_key(m: usize, n: usize, k: usize, dtype: DType) -> String {
        Self::entry_key_for(m, n, k, dtype, GemmMode::Plain)
    }

    pub fn residual_entry_key(m: usize, n: usize, k: usize, dtype: DType) -> String {
        Self::entry_key_for(m, n, k, dtype, GemmMode::Residual)
    }

    pub fn entry_key_for(m: usize, n: usize, k: usize, dtype: DType, mode: GemmMode) -> String {
        let mode = match mode {
            GemmMode::Plain => "plain",
            GemmMode::Residual => "residual",
        };
        format!("{m}_{n}_{k}_{dtype:?}_{mode}")
    }
}

fn validate_shape(m: usize, n: usize, k: usize, dtype: DType) -> Result<()> {
    if m == 0
        || n == 0
        || k == 0
        || m > i32::MAX as usize
        || n > i32::MAX as usize
        || k > i32::MAX as usize
        || dtype != DType::Fp8E4M3
    {
        return Err(invalid_policy(format!(
            "invalid FP8 GEMM shape ({m}, {n}, {k}, {dtype:?})"
        )));
    }
    Ok(())
}

fn parse_entry_key(key: &str) -> Result<ShapeKey> {
    let parts: Vec<_> = key.split('_').collect();
    if parts.len() != 5 || parts[3] != "Fp8E4M3" {
        return Err(invalid_policy(format!("malformed shape key {key:?}")));
    }
    let parse = |value: &str| {
        value
            .parse::<u32>()
            .map_err(|_| invalid_policy(format!("malformed shape key {key:?}")))
    };
    let shape = ShapeKey {
        m: parse(parts[0])?,
        n: parse(parts[1])?,
        k: parse(parts[2])?,
        dtype: DType::Fp8E4M3,
        mode: match parts[4] {
            "plain" => GemmMode::Plain,
            "residual" => GemmMode::Residual,
            _ => return Err(invalid_policy(format!("malformed shape key {key:?}"))),
        },
    };
    validate_shape(
        shape.m as usize,
        shape.n as usize,
        shape.k as usize,
        shape.dtype,
    )?;
    Ok(shape)
}

fn required_env(name: &'static str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            RvllmError::config(ConfigError::MissingField { name }, "autotune policy trust")
        })
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
        err: IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| RvllmError::Io {
        err: IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_POLICY_BYTES {
        return Err(invalid_policy(format!(
            "policy must be a regular file of 1..={MAX_POLICY_BYTES} bytes"
        )));
    }
    let mut body = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_POLICY_BYTES + 1)
        .read_to_end(&mut body)
        .map_err(|source| RvllmError::Io {
            err: IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    if body.len() as u64 > MAX_POLICY_BYTES {
        return Err(invalid_policy("policy grew while reading".into()));
    }
    Ok(body)
}

fn validate_digest(digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_policy(
            "trusted policy digest must be 64 lowercase hex characters".into(),
        ));
    }
    Ok(())
}

fn invalid_policy(reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "policy.json",
            reason,
        },
        "policy.json",
    )
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().expect("four-byte chunk"));
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ScheduleTag;
    use crate::variants::{ClusterShape, TileShape};

    fn policy_with_one_entry(mode: GemmMode) -> Policy {
        let mut entries = BTreeMap::new();
        entries.insert(
            Policy::entry_key_for(128, 152064, 3584, DType::Fp8E4M3, mode),
            PolicyEntry {
                variant: VariantId(0),
                workspace_bytes: 1 << 20,
            },
        );
        Policy {
            revision: "0123456".into(),
            arch: "sm_90".into(),
            variants: vec![VariantDescriptor {
                id: VariantId(0),
                tile: TileShape::new(128, 128, 128),
                cluster: ClusterShape::one(),
                mainloop: ScheduleTag::Coop,
                epilogue: ScheduleTag::Coop,
            }],
            entries,
        }
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn lookup_modes_do_not_collide() {
        let plain = policy_with_one_entry(GemmMode::Plain);
        assert!(plain.lookup(128, 152064, 3584, DType::Fp8E4M3).is_ok());
        assert!(plain
            .lookup_residual(128, 152064, 3584, DType::Fp8E4M3)
            .is_err());
    }

    #[test]
    fn rejects_unknown_variant() {
        let mut policy = policy_with_one_entry(GemmMode::Plain);
        policy.entries.values_mut().next().unwrap().variant = VariantId(999);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn trusted_load_checks_digest_and_identity() {
        let policy = policy_with_one_entry(GemmMode::Plain);
        let bytes = serde_json::to_vec(&policy).unwrap();
        let path = std::env::temp_dir().join(format!(
            "rvllm-policy-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, &bytes).unwrap();
        let loaded = Policy::load_trusted(&path, &sha256_hex(&bytes), "0123456", "sm_90").unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.entries.len(), 1);
    }
}
