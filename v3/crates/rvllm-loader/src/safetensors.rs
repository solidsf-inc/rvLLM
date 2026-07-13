//! Safetensors shard parsing.
//!
//! Layout: `[u64 header_bytes][JSON header][tensor bytes...]`. The JSON
//! header maps `name -> { dtype, shape, data_offsets: [start,end] }`
//! where offsets are RELATIVE to the start of the tensor payload region
//! (i.e. after header_bytes + 8).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 100 * 1024 * 1024;
pub const MAX_SAFETENSORS_SHARD_BYTES: u64 = 1 << 40;
pub const MAX_SAFETENSORS_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS_PER_SHARD: usize = 1_000_000;
const MAX_TENSOR_RANK: usize = 16;
const MAX_TENSOR_NAME_BYTES: usize = 4096;

struct UniqueObject(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueObject {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueObjectVisitor;

        impl<'de> Visitor<'de> for UniqueObjectVisitor {
            type Value = UniqueObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object with unique keys")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut out = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if out.insert(key.clone(), value).is_some() {
                        return Err(A::Error::custom(format!("duplicate key {key:?}")));
                    }
                }
                Ok(UniqueObject(out))
            }
        }

        deserializer.deserialize_map(UniqueObjectVisitor)
    }
}

struct UniqueStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueStringMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueStringMapVisitor;

        impl<'de> Visitor<'de> for UniqueStringMapVisitor {
            type Value = UniqueStringMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string map with unique keys")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut out = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    if out.insert(key.clone(), value).is_some() {
                        return Err(A::Error::custom(format!("duplicate weight {key:?}")));
                    }
                }
                Ok(UniqueStringMap(out))
            }
        }

        deserializer.deserialize_map(UniqueStringMapVisitor)
    }
}

#[derive(Deserialize)]
struct IndexDocument {
    weight_map: UniqueStringMap,
}

/// Tensor entry in a shard.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset inside the shard file (relative to file start, i.e.
    /// already includes the `8 + header_bytes` prefix).
    pub file_offset: u64,
    pub nbytes: u64,
}

/// A parsed shard's header. Backing file is mmap'd by the caller.
#[derive(Clone, Debug)]
pub struct ShardHeader {
    pub path: PathBuf,
    pub total_bytes: u64,
    pub tensors: BTreeMap<String, TensorEntry>,
}

impl ShardHeader {
    pub fn parse(path: &Path, file_bytes: &[u8]) -> Result<Self> {
        let loader_err = |detail: String| -> RvllmError {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ctx: LoaderCtx {
                    path: path.to_path_buf(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        };

        let total_bytes = u64::try_from(file_bytes.len())
            .map_err(|_| loader_err("file length does not fit u64".into()))?;
        if total_bytes > MAX_SAFETENSORS_SHARD_BYTES {
            return Err(loader_err(format!(
                "shard is {total_bytes} bytes; limit is {MAX_SAFETENSORS_SHARD_BYTES}"
            )));
        }
        if file_bytes.len() < 8 {
            return Err(loader_err("shorter than 8-byte header prefix".into()));
        }
        let header_u64 = u64::from_le_bytes(file_bytes[..8].try_into().expect("8-byte prefix"));
        let header_bytes = usize::try_from(header_u64)
            .map_err(|_| loader_err("header length does not fit usize".into()))?;
        if header_bytes > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(loader_err(format!(
                "header is {header_bytes} bytes; limit is {MAX_SAFETENSORS_HEADER_BYTES}"
            )));
        }
        let payload_start = 8usize
            .checked_add(header_bytes)
            .ok_or_else(|| loader_err("header offset overflow".into()))?;
        if payload_start > file_bytes.len() {
            return Err(loader_err(format!(
                "header claims {header_bytes} bytes but file is only {}",
                file_bytes.len()
            )));
        }
        let header_str = std::str::from_utf8(&file_bytes[8..payload_start])
            .map_err(|_| loader_err("header is not valid utf-8".into()))?;
        let header: UniqueObject = serde_json::from_str(header_str)
            .map_err(|e| loader_err(format!("header json: {e}")))?;
        if header.0.len() > MAX_TENSORS_PER_SHARD + 1 {
            return Err(loader_err(format!(
                "header contains too many entries: {}",
                header.0.len()
            )));
        }

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, meta) in header.0 {
            if name == "__metadata__" {
                continue;
            }
            if name.is_empty()
                || name.len() > MAX_TENSOR_NAME_BYTES
                || name.chars().any(char::is_control)
            {
                return Err(loader_err(
                    "tensor name is empty, too long, or contains controls".into(),
                ));
            }
            let obj = meta
                .as_object()
                .ok_or_else(|| loader_err(format!("{name}: meta not an object")))?;
            let dtype_str = obj
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| loader_err(format!("{name}: missing dtype")))?;
            let dtype = map_dtype(dtype_str)
                .ok_or_else(|| loader_err(format!("{name}: unsupported dtype {dtype_str}")))?;
            let shape_values = obj
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing shape")))?;
            if shape_values.len() > MAX_TENSOR_RANK {
                return Err(loader_err(format!(
                    "{name}: rank {} exceeds {MAX_TENSOR_RANK}",
                    shape_values.len()
                )));
            }
            let shape: Vec<usize> = shape_values
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| loader_err(format!("{name}: bad shape element")))
                })
                .collect::<Result<Vec<_>>>()?;
            let offsets = obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing data_offsets")))?;
            if offsets.len() != 2 {
                return Err(loader_err(format!(
                    "{name}: expected 2 offsets got {}",
                    offsets.len()
                )));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| loader_err(format!("{name}: start offset is not u64")))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| loader_err(format!("{name}: end offset is not u64")))?;
            let nbytes = end
                .checked_sub(start)
                .ok_or_else(|| loader_err(format!("{name}: end offset precedes start")))?;
            let elements = shape.iter().try_fold(1usize, |acc, dim| {
                acc.checked_mul(*dim)
                    .ok_or_else(|| loader_err(format!("{name}: shape product overflow")))
            })?;
            let expected_usize = elements
                .checked_mul(dtype_bytes(dtype))
                .ok_or_else(|| loader_err(format!("{name}: byte size overflow")))?;
            let expected = u64::try_from(expected_usize)
                .map_err(|_| loader_err(format!("{name}: byte size does not fit u64")))?;
            if expected != nbytes {
                return Err(loader_err(format!(
                    "{name}: offset range {nbytes} != dtype*shape {expected}"
                )));
            }
            let payload_len = u64::try_from(file_bytes.len() - payload_start)
                .map_err(|_| loader_err(format!("{name}: payload length does not fit u64")))?;
            if end > payload_len {
                return Err(loader_err(format!(
                    "{name}: data range {start}..{end} exceeds payload length {payload_len}"
                )));
            }
            let file_offset = u64::try_from(payload_start)
                .ok()
                .and_then(|base| base.checked_add(start))
                .ok_or_else(|| loader_err(format!("{name}: file offset overflow")))?;
            ranges.push((start, end, name.clone()));
            if tensors
                .insert(
                    name.clone(),
                    TensorEntry {
                        name: name.clone(),
                        dtype,
                        shape,
                        file_offset,
                        nbytes,
                    },
                )
                .is_some()
            {
                return Err(loader_err(format!("duplicate tensor {name:?}")));
            }
        }
        ranges.sort_by_key(|range| (range.0, range.1));
        for pair in ranges.windows(2) {
            let (_, previous_end, previous_name) = &pair[0];
            let (next_start, _, next_name) = &pair[1];
            if *next_start < *previous_end {
                return Err(loader_err(format!(
                    "tensor ranges overlap: {previous_name:?} ends at {previous_end}, {next_name:?} starts at {next_start}"
                )));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            total_bytes,
            tensors,
        })
    }
}

fn map_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F64" => DType::F64,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F8_E4M3" | "F8E4M3" => DType::Fp8E4M3,
        "F8_E5M2" | "F8E5M2" => DType::Fp8E5M2,
        // Integer dtypes used by compressed-tensors pack-quantized INT4:
        //   weight_packed: I32 (8 int4 nibbles per lane),
        //   weight_shape:  I64 ([out, in] logical dims).
        "I32" => DType::I32,
        "I64" => DType::I64,
        "U32" => DType::U32,
        "U8" => DType::U8,
        _ => return None,
    })
}

fn dtype_bytes(d: DType) -> usize {
    // Single canonical size source (DType::bytes); main's paged_attention.rs also
    // depends on DType::bytes(), and it covers all variants incl. F64/I64/Fp8E5M2/U8.
    d.bytes()
}

/// HF often ships sharded models: `model.safetensors.index.json`
/// maps `weight_name -> "model-00001-of-00004.safetensors"`.
#[derive(Clone, Debug)]
pub struct ShardIndex {
    pub shards: Vec<PathBuf>,
    pub weight_to_shard: BTreeMap<String, PathBuf>,
}

impl ShardIndex {
    /// Resolve the shard set under `model_dir`.
    ///
    /// - If `model.safetensors.index.json` exists, parses the map.
    /// - Else falls back to a single `model.safetensors`.
    pub fn resolve(model_dir: &Path) -> Result<Self> {
        let model_root = model_dir.canonicalize().map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: model_dir.to_path_buf(),
            source,
        })?;
        let index_path = model_root.join("model.safetensors.index.json");
        let err_ctx = |detail: String| -> RvllmError {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ctx: LoaderCtx {
                    path: index_path.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        };
        if index_path.exists() {
            let canonical_index = index_path.canonicalize().map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: index_path.clone(),
                source,
            })?;
            if !canonical_index.starts_with(&model_root) {
                return Err(err_ctx("index resolves outside model directory".into()));
            }
            let bytes = read_bounded(&canonical_index, MAX_SAFETENSORS_INDEX_BYTES)?;
            let obj: IndexDocument =
                serde_json::from_slice(&bytes).map_err(|e| err_ctx(format!("index json: {e}")))?;
            let mut weight_to_shard = BTreeMap::new();
            let mut shards_set = BTreeSet::new();
            if obj.weight_map.0.is_empty() {
                return Err(err_ctx("index weight_map is empty".into()));
            }
            for (k, shard) in obj.weight_map.0 {
                if k.is_empty()
                    || k.len() > MAX_TENSOR_NAME_BYTES
                    || k.chars().any(char::is_control)
                {
                    return Err(err_ctx("index contains an invalid tensor name".into()));
                }
                let relative = Path::new(&shard);
                if relative.as_os_str().is_empty()
                    || relative.is_absolute()
                    || relative
                        .components()
                        .any(|component| !matches!(component, std::path::Component::Normal(_)))
                {
                    return Err(err_ctx(format!("{k}: invalid shard path {shard:?}")));
                }
                let p =
                    model_root
                        .join(relative)
                        .canonicalize()
                        .map_err(|source| RvllmError::Io {
                            err: rvllm_core::IoError::from(&source),
                            path: model_root.join(relative),
                            source,
                        })?;
                if !p.starts_with(&model_root) || !p.is_file() {
                    return Err(err_ctx(format!(
                        "{k}: shard resolves outside model directory"
                    )));
                }
                shards_set.insert(p.clone());
                weight_to_shard.insert(k, p);
            }
            Ok(Self {
                shards: shards_set.into_iter().collect(),
                weight_to_shard,
            })
        } else {
            let single = model_root.join("model.safetensors");
            if !single.exists() {
                return Err(err_ctx(format!(
                    "no index at {} and no model.safetensors at {}",
                    index_path.display(),
                    single.display()
                )));
            }
            let canonical_single = single.canonicalize().map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: single.clone(),
                source,
            })?;
            if !canonical_single.starts_with(&model_root) || !canonical_single.is_file() {
                return Err(err_ctx(
                    "model.safetensors resolves outside model directory".into(),
                ));
            }
            Ok(Self {
                shards: vec![canonical_single],
                weight_to_shard: BTreeMap::new(),
            })
        }
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let len = file
        .metadata()
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if len > limit {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("{} is {len} bytes; limit is {limit}", path.display()),
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > limit {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("{} grew beyond {limit} bytes while reading", path.display()),
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_shard(dir: &Path, tensors: &[(&str, DType, &[usize], &[u8])]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = payload.len();
            payload.extend_from_slice(data);
            let end = payload.len();
            let mut meta = serde_json::Map::new();
            let dt = match dtype {
                DType::F32 => "F32",
                DType::F64 => "F64",
                DType::F16 => "F16",
                DType::Bf16 => "BF16",
                DType::Fp8E4M3 => "F8_E4M3",
                DType::Fp8E5M2 => "F8_E5M2",
                DType::I32 => "I32",
                DType::I64 => "I64",
                DType::U32 => "U32",
                DType::U8 => "U8",
            };
            meta.insert("dtype".into(), serde_json::Value::String(dt.into()));
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|n| serde_json::Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number((start as u64).into()),
                    serde_json::Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), serde_json::Value::Object(meta));
        }
        let hjson = serde_json::to_string(&header).unwrap();
        let hb = hjson.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        f.write_all(&payload).unwrap();
        path
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-loader-st-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn parses_minimal_shard() {
        let dir = tempdir();
        let data_f32 = (0u32..4)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let path = write_shard(&dir, &[("w", DType::F32, &[4], &data_f32)]);
        let body = std::fs::read(&path).unwrap();
        let hdr = ShardHeader::parse(&path, &body).unwrap();
        let w = hdr.tensors.get("w").unwrap();
        assert_eq!(w.shape, vec![4]);
        assert!(matches!(w.dtype, DType::F32));
        assert_eq!(w.nbytes, 16);
    }

    #[test]
    fn rejects_wrong_offset_length() {
        let dir = tempdir();
        // Payload is 3 bytes but shape says 4 f32 = 16 bytes.
        let path = write_shard(&dir, &[("w", DType::F32, &[4], &[0u8, 1, 2])]);
        let body = std::fs::read(&path).unwrap();
        let err = ShardHeader::parse(&path, &body).unwrap_err();
        assert!(matches!(
            err,
            RvllmError::Loader {
                err: LoaderError::Corrupt { ref detail },
                ..
            } if detail.contains("offset range")
        ));
    }

    #[test]
    fn fallback_to_single_shard() {
        let dir = tempdir();
        let _ = write_shard(&dir, &[("w", DType::F32, &[1], &[0u8; 4])]);
        let idx = ShardIndex::resolve(&dir).unwrap();
        assert_eq!(idx.shards.len(), 1);
        assert!(idx.weight_to_shard.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn single_shard_must_resolve_inside_model_root() {
        use std::os::unix::fs::symlink;

        let root = tempdir();
        let model = root.join("model");
        let outside = root.join("outside");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let external = write_shard(&outside, &[("w", DType::F32, &[1], &[0u8; 4])]);
        symlink(external, model.join("model.safetensors")).unwrap();

        let error = ShardIndex::resolve(&model).unwrap_err();
        assert!(matches!(
            error,
            RvllmError::Loader {
                err: LoaderError::Corrupt { ref detail },
                ..
            } if detail.contains("outside model directory")
        ));
    }

    #[test]
    fn parses_metadata_dtypes_used_by_baked_models() {
        let dir = tempdir();
        let u8_data = [0u8; 4];
        let i64_data = [0u8; 16];
        let path = write_shard(
            &dir,
            &[
                ("packed", DType::U8, &[4], &u8_data),
                ("shape", DType::I64, &[2], &i64_data),
            ],
        );
        let body = std::fs::read(&path).unwrap();
        let hdr = ShardHeader::parse(&path, &body).unwrap();
        assert!(matches!(hdr.tensors["packed"].dtype, DType::U8));
        assert!(matches!(hdr.tensors["shape"].dtype, DType::I64));
        assert_eq!(hdr.tensors["shape"].nbytes, 16);
    }
}
