//! `manifest.json`: the SHA-pinned catalog of every kernel artifact.
//!
//! A release artifact bundle ships this file next to its kernels. At engine
//! init, `KernelManifest::load_and_verify` reads
//! `manifest.json`, then recomputes sha256 of every listed file and
//! aborts if any digest drifts. There is no lookup path that bypasses
//! this; `KernelLoader::new` takes a `VerifiedManifest` and refuses to
//! read anything not in it.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use rvllm_core::{ConfigError, IoError, Result, RvllmError};

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 4096;

fn deserialize_unique_entries<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, ArtifactEntry>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueEntries;

    impl<'de> Visitor<'de> for UniqueEntries {
        type Value = BTreeMap<String, ArtifactEntry>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map with unique artifact names")
        }

        fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut entries = BTreeMap::new();
            while let Some((name, entry)) = map.next_entry::<String, ArtifactEntry>()? {
                if entries.insert(name.clone(), entry).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate artifact name {name:?}"
                    )));
                }
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(UniqueEntries)
}

/// Manifest entry for one artifact.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactEntry {
    /// Path relative to the manifest file's directory.
    pub path: String,
    /// sha256 hex digest (lowercase, 64 chars).
    pub sha256: String,
    /// Size in bytes.
    pub bytes: u64,
    /// Artifact container interpreted by the loader.
    pub kind: ArtifactKind,
    /// Versioned ABI contract, for example `cuda-ptx-v1`.
    pub abi: String,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Ptx,
    SharedObject,
}

/// The full deploy manifest.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KernelManifest {
    /// Build SHA that produced this manifest. Engine init verifies it against
    /// the operator-supplied `RVLLM_RELEASE_REVISION` trust pin.
    pub revision: String,
    /// GPU arch the kernels were built for (e.g. `sm_90`).
    pub arch: String,
    /// Entries keyed by logical name (e.g. `libfa3_kernels.so`, `argmax`).
    #[serde(deserialize_with = "deserialize_unique_entries")]
    pub entries: BTreeMap<String, ArtifactEntry>,
}

/// A `KernelManifest` whose on-disk checksums have been re-verified.
/// Only this type unlocks `KernelLoader`.
#[derive(Clone, Debug)]
pub struct VerifiedManifest {
    manifest: KernelManifest,
    root: PathBuf,
    artifacts: BTreeMap<String, VerifiedArtifact>,
}

#[derive(Clone, Debug)]
pub struct VerifiedArtifact {
    path: PathBuf,
    bytes: Arc<[u8]>,
    kind: ArtifactKind,
    abi: String,
}

impl VerifiedArtifact {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn kind(&self) -> ArtifactKind {
        self.kind
    }

    pub fn abi(&self) -> &str {
        &self.abi
    }
}

impl VerifiedManifest {
    pub fn manifest(&self) -> &KernelManifest {
        &self.manifest
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Resolve a logical name to its on-disk absolute path.
    /// Returns `None` if the name is not in the manifest.
    pub fn path_of(&self, logical_name: &str) -> Option<PathBuf> {
        Some(self.artifacts.get(logical_name)?.path.clone())
    }
    pub fn artifact(&self, logical_name: &str) -> Option<&VerifiedArtifact> {
        self.artifacts.get(logical_name)
    }
    pub fn revision(&self) -> &str {
        &self.manifest.revision
    }
    pub fn arch(&self) -> &str {
        &self.manifest.arch
    }
}

impl KernelManifest {
    /// Verify against deployment trust pins supplied by the operator.
    pub fn load_and_verify(manifest_path: &Path) -> Result<VerifiedManifest> {
        let digest = required_env("RVLLM_KERNEL_MANIFEST_SHA256")?;
        let revision = required_env("RVLLM_RELEASE_REVISION")?;
        let arch = required_env("RVLLM_KERNEL_ARCH")?;
        Self::load_and_verify_trusted(manifest_path, &digest, &revision, &arch)
    }

    /// Load and authenticate the manifest, then read and verify every
    /// artifact into immutable owned bytes. Compute paths consume these bytes
    /// directly and never reopen the verified path.
    pub fn load_and_verify_trusted(
        manifest_path: &Path,
        expected_manifest_sha256: &str,
        expected_revision: &str,
        expected_arch: &str,
    ) -> Result<VerifiedManifest> {
        validate_digest(expected_manifest_sha256, "trusted manifest digest")?;
        let body = read_bounded(manifest_path, MAX_MANIFEST_BYTES)?;
        let got_manifest_digest = sha256_hex(&body);
        if got_manifest_digest != expected_manifest_sha256 {
            return Err(invalid_manifest(format!(
                "manifest digest {got_manifest_digest} does not match trusted pin {expected_manifest_sha256}"
            )));
        }
        let manifest: KernelManifest = serde_json::from_slice(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("manifest.json is not valid JSON: {e}")],
                },
                "manifest.json",
            )
        })?;
        validate_manifest_identity(&manifest, expected_revision, expected_arch)?;
        if manifest.entries.is_empty() || manifest.entries.len() > MAX_ARTIFACTS {
            return Err(invalid_manifest(format!(
                "artifact count must be in 1..={MAX_ARTIFACTS}, got {}",
                manifest.entries.len()
            )));
        }

        let root = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()
            .map_err(|source| RvllmError::Io {
                err: IoError::from(&source),
                path: manifest_path.to_path_buf(),
                source,
            })?;
        let mut artifacts = BTreeMap::new();
        let mut total_bytes = 0u64;
        for (name, entry) in &manifest.entries {
            validate_name(name)?;
            validate_digest(&entry.sha256, name)?;
            validate_entry_contract(entry, name)?;
            if entry.bytes == 0 || entry.bytes > MAX_ARTIFACT_BYTES {
                return Err(invalid_manifest(format!(
                    "{name}: bytes must be in 1..={MAX_ARTIFACT_BYTES}"
                )));
            }
            total_bytes = total_bytes
                .checked_add(entry.bytes)
                .ok_or_else(|| invalid_manifest("total artifact byte count overflow".into()))?;
            if total_bytes > MAX_ARTIFACT_BYTES * 4 {
                return Err(invalid_manifest(
                    "total artifact bytes exceed release limit".into(),
                ));
            }
            let relative = validate_relative_path(&entry.path, name)?;
            let path = root
                .join(relative)
                .canonicalize()
                .map_err(|source| RvllmError::Io {
                    err: IoError::from(&source),
                    path: root.join(&entry.path),
                    source,
                })?;
            if !path.starts_with(&root) {
                return Err(invalid_manifest(format!(
                    "{name}: artifact escapes manifest root"
                )));
            }
            let bytes = read_bounded(&path, entry.bytes)?;
            if bytes.len() as u64 != entry.bytes {
                return Err(invalid_manifest(format!(
                    "{name}: size {} does not match manifest {}",
                    bytes.len(),
                    entry.bytes
                )));
            }
            let got = sha256_hex(&bytes);
            if got != entry.sha256 {
                return Err(invalid_manifest(format!(
                    "{name}: sha256 {got} does not match manifest {}",
                    entry.sha256
                )));
            }
            artifacts.insert(
                name.clone(),
                VerifiedArtifact {
                    path,
                    bytes: Arc::from(bytes),
                    kind: entry.kind,
                    abi: entry.abi.clone(),
                },
            );
        }

        Ok(VerifiedManifest {
            manifest,
            root,
            artifacts,
        })
    }
}

fn required_env(name: &'static str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            RvllmError::config(ConfigError::MissingField { name }, "kernel manifest trust")
        })
}

fn validate_manifest_identity(
    manifest: &KernelManifest,
    expected_revision: &str,
    expected_arch: &str,
) -> Result<()> {
    if manifest.revision != expected_revision {
        return Err(invalid_manifest(format!(
            "revision {:?} does not match trusted revision {:?}",
            manifest.revision, expected_revision
        )));
    }
    if manifest.arch != expected_arch {
        return Err(invalid_manifest(format!(
            "architecture {:?} does not match selected target {:?}",
            manifest.arch, expected_arch
        )));
    }
    if manifest.revision.len() < 7
        || manifest.revision.len() > 64
        || !manifest
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid_manifest(
            "revision must be a 7..=64 character hex commit ID".into(),
        ));
    }
    if !matches!(
        manifest.arch.as_str(),
        "sm_80" | "sm_89" | "sm_90" | "sm_100" | "sm_121"
    ) {
        return Err(invalid_manifest(format!(
            "unsupported kernel architecture {:?}",
            manifest.arch
        )));
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(invalid_manifest(format!("invalid artifact name {name:?}")));
    }
    Ok(())
}

fn validate_digest(digest: &str, owner: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_manifest(format!(
            "{owner}: sha256 must be 64 lowercase hex characters"
        )));
    }
    Ok(())
}

fn validate_entry_contract(entry: &ArtifactEntry, name: &str) -> Result<()> {
    if entry.abi.is_empty()
        || entry.abi.len() > 64
        || !entry
            .abi
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(invalid_manifest(format!("{name}: invalid ABI identifier")));
    }
    let extension = Path::new(&entry.path)
        .extension()
        .and_then(|value| value.to_str());
    match entry.kind {
        ArtifactKind::Ptx if extension == Some("ptx") && entry.abi == "cuda-ptx-v1" => Ok(()),
        ArtifactKind::SharedObject
            if extension == Some("so") && entry.abi.starts_with("rvllm-cuda-so-v") =>
        {
            Ok(())
        }
        _ => Err(invalid_manifest(format!(
            "{name}: artifact kind, extension, and ABI do not agree"
        ))),
    }
}

fn validate_relative_path<'a>(path: &'a str, name: &str) -> Result<&'a Path> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(invalid_manifest(format!(
            "{name}: artifact path must be a contained relative path"
        )));
    }
    Ok(path)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|source| RvllmError::Io {
        err: IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| RvllmError::Io {
        err: IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(invalid_manifest(format!(
            "{} is not a regular file within the {limit}-byte limit",
            path.display()
        )));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| invalid_manifest(format!("{} length exceeds usize", path.display())))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| RvllmError::Io {
            err: IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > limit {
        return Err(invalid_manifest(format!(
            "{} grew beyond the {limit}-byte limit while reading",
            path.display()
        )));
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn invalid_manifest(reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: "manifest.json",
            reason,
        },
        "manifest.json",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(body).unwrap();
        p
    }

    #[test]
    fn roundtrip_verify() {
        let tmp = tempdir();
        let artifact = write_tmp(&tmp, "kern.ptx", b"PTX CONTENT");
        let digest = {
            let mut h = Sha256::new();
            h.update(b"PTX CONTENT");
            hex::encode(h.finalize())
        };
        let mut entries = BTreeMap::new();
        entries.insert(
            "argmax".into(),
            ArtifactEntry {
                path: "kern.ptx".into(),
                sha256: digest,
                bytes: 11,
                kind: ArtifactKind::Ptx,
                abi: "cuda-ptx-v1".into(),
            },
        );
        let manifest = KernelManifest {
            revision: "a".repeat(40),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&mp, &manifest_bytes).unwrap();
        let root_digest = sha256_hex(&manifest_bytes);
        let verified =
            KernelManifest::load_and_verify_trusted(&mp, &root_digest, &"a".repeat(40), "sm_90")
                .unwrap();
        assert_eq!(verified.revision(), "a".repeat(40));
        assert_eq!(verified.arch(), "sm_90");
        assert_eq!(
            verified.path_of("argmax").unwrap(),
            artifact.canonicalize().unwrap()
        );
    }

    #[test]
    fn drift_rejected() {
        let tmp = tempdir();
        write_tmp(&tmp, "kern.ptx", b"PTX CONTENT");
        let bogus = "0".repeat(64);
        let mut entries = BTreeMap::new();
        entries.insert(
            "argmax".into(),
            ArtifactEntry {
                path: "kern.ptx".into(),
                sha256: bogus,
                bytes: 11,
                kind: ArtifactKind::Ptx,
                abi: "cuda-ptx-v1".into(),
            },
        );
        let manifest = KernelManifest {
            revision: "a".repeat(40),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&mp, &manifest_bytes).unwrap();
        let root_digest = sha256_hex(&manifest_bytes);
        let err =
            KernelManifest::load_and_verify_trusted(&mp, &root_digest, &"a".repeat(40), "sm_90")
                .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("sha256"));
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-kernels-manifest-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
}
