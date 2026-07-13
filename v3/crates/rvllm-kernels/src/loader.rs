//! `KernelLoader`: the only path to open PTX modules and `.so` handles.
//!
//! Construction takes a `VerifiedManifest`, so the SHA-pinned invariant
//! propagates: no artifact is touched unless its digest matched at
//! verification time. Requests for a logical name that is not in the
//! manifest return `Err(RvllmError::config(MissingField, ...))` — the
//! engine refuses to start rather than fall back.

use rvllm_core::{ConfigError, Result, RvllmError};

use crate::manifest::{ArtifactKind, VerifiedArtifact, VerifiedManifest};

pub struct KernelLoader {
    manifest: VerifiedManifest,
    context: rvllm_mem::CudaContextHandle,
}

impl KernelLoader {
    /// Build a loader from a verified manifest. The manifest must
    /// already have passed `KernelManifest::load_and_verify`.
    pub fn new(manifest: VerifiedManifest, context: &rvllm_mem::CudaContextHandle) -> Self {
        Self {
            manifest,
            context: context.clone(),
        }
    }

    pub fn manifest(&self) -> &VerifiedManifest {
        &self.manifest
    }

    /// Return the absolute path of an artifact by logical name, or
    /// `Err` if the manifest has no such entry. Engine refuses to
    /// start on `Err`.
    pub fn path(&self, logical_name: &str) -> Result<std::path::PathBuf> {
        self.manifest.path_of(logical_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "manifest.entries",
                    reason: format!("missing artifact {logical_name:?}"),
                },
                "manifest.entries",
            )
        })
    }

    /// Load a PTX module by logical name into a `LoadedModule`. Under
    /// feature `cuda` this is `cuModuleLoad` on the file. Under no-cuda
    /// it verifies the file is readable and returns a stub module so
    /// type-level tests compose.
    pub fn load_ptx(&self, logical_name: &str) -> Result<crate::module::LoadedModule> {
        let artifact = self.artifact(logical_name)?;
        if artifact.kind() != ArtifactKind::Ptx || artifact.abi() != "cuda-ptx-v1" {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "manifest.entries",
                    reason: format!("{logical_name:?} is not a cuda-ptx-v1 artifact"),
                },
                "manifest.entries",
            ));
        }
        crate::module::LoadedModule::load_from_bytes(
            &self.context,
            artifact.path().to_path_buf(),
            artifact.bytes(),
        )
    }

    /// Convenience: raw PTX bytes for a logical name (for host-side
    /// inspection, not the compute path).
    pub fn read_ptx_bytes(&self, logical_name: &str) -> Result<PtxBytes> {
        let artifact = self.artifact(logical_name)?;
        if artifact.kind() != ArtifactKind::Ptx {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "manifest.entries",
                    reason: format!("{logical_name:?} is not PTX"),
                },
                "manifest.entries",
            ));
        }
        Ok(PtxBytes {
            bytes: artifact.bytes().to_vec(),
        })
    }

    /// Return the already-open, verified shared-object bytes and metadata.
    /// Dynamic loaders must materialize these bytes through a contained,
    /// immutable deployment mechanism rather than reopening an untrusted path.
    pub fn so_artifact(&self, logical_name: &str) -> Result<VerifiedArtifact> {
        let artifact = self.artifact(logical_name)?;
        if artifact.kind() != ArtifactKind::SharedObject {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "manifest.entries",
                    reason: format!("{logical_name:?} is not a shared object"),
                },
                "manifest.entries",
            ));
        }
        Ok(artifact.clone())
    }

    fn artifact(&self, logical_name: &str) -> Result<&VerifiedArtifact> {
        self.manifest.artifact(logical_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "manifest.entries",
                    reason: format!("missing artifact {logical_name:?}"),
                },
                "manifest.entries",
            )
        })
    }
}

/// Opaque host-side representation of PTX module bytes. Real runtime
/// replaces with `cudarc::CudaModule` under feature `cuda`.
#[derive(Debug)]
pub struct PtxBytes {
    pub bytes: Vec<u8>,
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::manifest::{ArtifactEntry, ArtifactKind, KernelManifest};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    fn verified(tmp: &PathBuf, body: &[u8], name: &str) -> VerifiedManifest {
        let file = tmp.join(format!("{name}.ptx"));
        fs::write(&file, body).unwrap();
        let digest = {
            let mut h = Sha256::new();
            h.update(body);
            hex::encode(h.finalize())
        };
        let mut entries = BTreeMap::new();
        entries.insert(
            name.to_string(),
            ArtifactEntry {
                path: format!("{name}.ptx"),
                sha256: digest,
                bytes: body.len() as u64,
                kind: ArtifactKind::Ptx,
                abi: "cuda-ptx-v1".into(),
            },
        );
        let m = KernelManifest {
            revision: "b".repeat(40),
            arch: "sm_90".into(),
            entries,
        };
        let mp = tmp.join("manifest.json");
        let manifest_bytes = serde_json::to_vec_pretty(&m).unwrap();
        fs::write(&mp, &manifest_bytes).unwrap();
        let mut h = Sha256::new();
        h.update(&manifest_bytes);
        let digest = hex::encode(h.finalize());
        KernelManifest::load_and_verify_trusted(&mp, &digest, &"b".repeat(40), "sm_90").unwrap()
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-kernels-loader-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn load_ptx_roundtrip() {
        let tmp = tempdir();
        let vm = verified(&tmp, b"HELLO PTX", "argmax");
        let loader = KernelLoader::new(vm, &rvllm_mem::CudaContextHandle::host_stub());
        assert!(loader.load_ptx("argmax").is_err());
        let bytes = loader.read_ptx_bytes("argmax").unwrap();
        assert_eq!(bytes.bytes, b"HELLO PTX");
    }

    #[test]
    fn missing_name_is_err() {
        let tmp = tempdir();
        let vm = verified(&tmp, b"HELLO PTX", "argmax");
        let loader = KernelLoader::new(vm, &rvllm_mem::CudaContextHandle::host_stub());
        let err = loader.load_ptx("silu").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("missing artifact"));
        assert!(s.contains("silu"));
    }
}
