//! rvllm-attention: architecture-specific paged decode + prefill.
//!
//! SM90 can load paged decode and prefill exports from
//! `libfa3_kernels.so`, built from FlashAttention-3 Hopper sources. SM80/SM89
//! use the rvLLM fallback shared object, and SM100/SM121 use authenticated PTX.
//!
//! The invariants:
//! - `head_dim` must be one of `{128, 256, 512}` at construction
//! - GQA ratio sanity (`num_heads` divisible by `num_kv_heads`)
//! - context_lens[i] == 0 valid padded-slot marker; kernel must predicate

pub mod decode;
pub mod prefill;

pub use decode::{PagedDecodeFp8Launcher, PagedDecodeLauncher, PagedDecodeParams};
pub use prefill::{PagedPrefillFp8Launcher, PagedPrefillLauncher, PagedPrefillParams};

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

#[cfg(feature = "cuda")]
const ATTENTION_SO_ABI_VERSION: i32 = 2;
#[cfg(feature = "cuda")]
const FA3_UPSTREAM_REVISION: &[u8] = b"1233b73b6c95340c65c9edfe929611838354fc6e";
#[cfg(feature = "cuda")]
const FP8_OUTPUT_DTYPE_F16: i32 = 1;
#[cfg(feature = "cuda")]
const FP8_OUTPUT_ELEMENT_BYTES: i32 = 2;

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct AuthenticatedLibrary {
    library: libloading::Library,
    #[cfg(target_os = "linux")]
    _backing: std::fs::File,
}

#[cfg(feature = "cuda")]
impl AuthenticatedLibrary {
    unsafe fn get<T>(
        &self,
        symbol: &[u8],
    ) -> std::result::Result<libloading::Symbol<'_, T>, libloading::Error> {
        self.library.get(symbol)
    }
}

#[cfg(feature = "cuda")]
fn load_authenticated_library(path: &std::path::Path) -> Result<AuthenticatedLibrary> {
    let canonical = path
        .canonicalize()
        .map_err(|error| authenticated_so_error(path, format!("canonicalize: {error}")))?;
    let manifest_path = path
        .parent()
        .ok_or_else(|| {
            authenticated_so_error(path, "shared object has no parent directory".into())
        })?
        .join("manifest.json");
    let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
    let mut matches = manifest
        .manifest()
        .entries
        .keys()
        .filter(|name| manifest.path_of(name).as_deref() == Some(canonical.as_path()));
    let name = matches.next().ok_or_else(|| {
        authenticated_so_error(path, "shared object is not in the verified manifest".into())
    })?;
    if matches.next().is_some() {
        return Err(authenticated_so_error(
            path,
            "shared object appears more than once in the verified manifest".into(),
        ));
    }
    let artifact = manifest
        .artifact(name)
        .ok_or_else(|| authenticated_so_error(path, "verified artifact disappeared".into()))?;
    if artifact.kind() != rvllm_kernels::manifest::ArtifactKind::SharedObject
        || !artifact.abi().starts_with("rvllm-cuda-so-v")
    {
        return Err(authenticated_so_error(
            path,
            format!("artifact has incompatible ABI {:?}", artifact.abi()),
        ));
    }
    load_verified_bytes(path, artifact.bytes())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn load_verified_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<AuthenticatedLibrary> {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};

    const MFD_CLOEXEC: u32 = 0x0001;
    const MFD_ALLOW_SEALING: u32 = 0x0002;
    const F_ADD_SEALS: i32 = 1033;
    const F_SEAL_SEAL: i32 = 0x0001;
    const F_SEAL_SHRINK: i32 = 0x0002;
    const F_SEAL_GROW: i32 = 0x0004;
    const F_SEAL_WRITE: i32 = 0x0008;

    unsafe extern "C" {
        fn memfd_create(name: *const std::ffi::c_char, flags: u32) -> i32;
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }

    let name = std::ffi::CString::new("rvllm-authenticated-attention").expect("static string");
    let fd = unsafe { memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(authenticated_so_error(
            path,
            format!("memfd_create: {}", std::io::Error::last_os_error()),
        ));
    }
    let mut backing = unsafe { std::fs::File::from_raw_fd(fd) };
    backing
        .write_all(bytes)
        .and_then(|_| backing.flush())
        .map_err(|error| authenticated_so_error(path, format!("materialize: {error}")))?;
    let seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if unsafe { fcntl(backing.as_raw_fd(), F_ADD_SEALS, seals) } != 0 {
        return Err(authenticated_so_error(
            path,
            format!("seal: {}", std::io::Error::last_os_error()),
        ));
    }
    let fd_path = std::path::PathBuf::from(format!("/proc/self/fd/{}", backing.as_raw_fd()));
    let library = unsafe { libloading::Library::new(&fd_path) }
        .map_err(|error| authenticated_so_error(path, format!("dlopen: {error}")))?;
    Ok(AuthenticatedLibrary {
        library,
        _backing: backing,
    })
}

#[cfg(all(feature = "cuda", not(target_os = "linux")))]
fn load_verified_bytes(path: &std::path::Path, _bytes: &[u8]) -> Result<AuthenticatedLibrary> {
    Err(authenticated_so_error(
        path,
        "authenticated CUDA shared-object loading requires Linux memfd sealing".into(),
    ))
}

#[cfg(feature = "cuda")]
fn authenticated_so_error(path: &std::path::Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: rvllm_core::LoaderError::Corrupt { detail },
        ctx: rvllm_core::LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

/// Runtime-constructed wrapper around `libfa3_kernels.so`. The wrapper
/// refuses to exist if the .so is missing or its manifest-verified
/// exports don't include the entry points. Callers obtain launchers
/// from the wrapper.
/// Function pointer types for authenticated attention .so exports.
#[cfg(feature = "cuda")]
pub(crate) type Sm89WorkspaceSizeFn =
    unsafe extern "C" fn(batch_size: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32) -> u64;

#[cfg(feature = "cuda")]
pub(crate) type AbiVersionFn = unsafe extern "C" fn() -> i32;

#[cfg(feature = "cuda")]
pub(crate) type OutputContractFn = unsafe extern "C" fn() -> i32;

#[cfg(feature = "cuda")]
pub(crate) type UpstreamRevisionFn = unsafe extern "C" fn() -> *const std::ffi::c_char;

#[cfg(feature = "cuda")]
pub(crate) type Fa3DecodeWorkspaceSizeFn = unsafe extern "C" fn(
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    is_fp8: i32,
    window_size_left: i32,
) -> u64;

#[cfg(feature = "cuda")]
pub(crate) type Fa3PrefillWorkspaceSizeFn = unsafe extern "C" fn(
    total_q: i32,
    max_seqlen_q: i32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    is_fp8: i32,
    window_size_left: i32,
) -> u64;

#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Debug)]
enum WorkspaceSizeDispatch {
    Sm89(Sm89WorkspaceSizeFn),
    Sm90 {
        decode: Fa3DecodeWorkspaceSizeFn,
        prefill: Fa3PrefillWorkspaceSizeFn,
    },
}

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedDecodeFn = unsafe extern "C" fn(
    q_ptr: *mut std::ffi::c_void,
    k_cache_ptr: *mut std::ffi::c_void,
    v_cache_ptr: *mut std::ffi::c_void,
    o_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    workspace_bytes: usize,
    scale: f32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    window_size_left: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

// FP8 E4M3 paged decode: Q / K cache / V cache are FP8 (1 byte/elem).
// q_descale / k_descale / v_descale point at f32 per-tensor scale scalars
// on the device. The authenticated output contract is F16, 2 bytes/element.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedDecodeFp8Fn = unsafe extern "C" fn(
    q_fp8_ptr: *mut std::ffi::c_void,
    k_cache_fp8_ptr: *mut std::ffi::c_void,
    v_cache_fp8_ptr: *mut std::ffi::c_void,
    o_f16_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    workspace_bytes: usize,
    k_scale_cache_ptr: *mut std::ffi::c_void,
    v_scale_cache_ptr: *mut std::ffi::c_void,
    q_scale_cache_ptr: *mut std::ffi::c_void,
    q_descale_ptr: *mut f32,
    k_descale_ptr: *mut f32,
    v_descale_ptr: *mut f32,
    scale: f32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    window_size_left: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

// FP8 E4M3 paged PREFILL: multi-query causal self-attention. Q layout is
// [total_q, num_heads, head_dim] indexed via cu_seqlens_q. K / V cache
// are paged FP8. Causal mask applied per-seq.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedPrefillFp8Fn = unsafe extern "C" fn(
    q_fp8_ptr: *mut std::ffi::c_void,
    k_cache_fp8_ptr: *mut std::ffi::c_void,
    v_cache_fp8_ptr: *mut std::ffi::c_void,
    o_f16_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    cu_seqlens_q_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    workspace_bytes: usize,
    k_scale_cache_ptr: *mut std::ffi::c_void,
    v_scale_cache_ptr: *mut std::ffi::c_void,
    q_scale_cache_ptr: *mut std::ffi::c_void,
    q_descale_ptr: *mut f32,
    k_descale_ptr: *mut f32,
    v_descale_ptr: *mut f32,
    scale: f32,
    total_q: i32,
    max_seqlen_q: i32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    window_size_left: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

#[derive(Debug)]
pub struct Fa3Kernels {
    pub so_path: std::path::PathBuf,
    pub head_dim: u32,
    #[cfg(feature = "cuda")]
    _lib: AuthenticatedLibrary,
    #[cfg(feature = "cuda")]
    workspace_size_dispatch: WorkspaceSizeDispatch,
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_decode: PagedDecodeFn,
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_decode_fp8: PagedDecodeFp8Fn,
    /// Optional prefill export. Callers must reject or explicitly route an
    /// unsupported prefill request when this is `None`.
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_prefill_fp8: Option<PagedPrefillFp8Fn>,
    /// True when the shared object exposes the `fa_sm89_*` ABI family.
    pub is_sm89_backend: bool,
}

impl Fa3Kernels {
    /// Load the FA3 .so. Called once at engine init from a
    /// `KernelLoader`-produced path. Returns `Err` with explicit
    /// `AttentionError::Fa3SoMissing` if the path does not exist.
    pub fn load(path: std::path::PathBuf, head_dim: u32) -> Result<Self> {
        if !path.exists() {
            return Err(RvllmError::Attention {
                err: AttentionError::Fa3SoMissing { path: path.clone() },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            unsafe {
                let _lib = load_authenticated_library(&path)?;
                let has_sm90 = _lib
                    .get::<Fa3DecodeWorkspaceSizeFn>(b"fa3_sm90_decode_workspace_size\0")
                    .is_ok();
                let has_sm89 = _lib
                    .get::<Sm89WorkspaceSizeFn>(b"fa_sm89_decode_workspace_size\0")
                    .is_ok();
                let is_sm89 = match (has_sm90, has_sm89) {
                    (true, false) => false,
                    (false, true) => true,
                    _ => {
                        return Err(authenticated_so_error(
                            &path,
                            "attention ABI must expose exactly one architecture family".into(),
                        ));
                    }
                };
                let (abi_name, dec_name, fp8_name, prefill_name): (&[u8], &[u8], &[u8], &[u8]) =
                    if is_sm89 {
                        (
                            b"rvllm_fa_sm89_abi_version\0",
                            b"fa_sm89_paged_decode\0",
                            b"fa_sm89_paged_decode_fp8\0",
                            b"fa_sm89_paged_prefill_fp8\0",
                        )
                    } else {
                        (
                            b"rvllm_fa3_abi_version\0",
                            b"fa3_sm90_paged_decode\0",
                            b"fa3_sm90_paged_decode_fp8\0",
                            b"fa3_sm90_paged_prefill_fp8\0",
                        )
                    };
                let (output_dtype_name, output_size_name): (&[u8], &[u8]) = if is_sm89 {
                    (
                        b"fa_sm89_fp8_output_dtype\0",
                        b"fa_sm89_fp8_output_element_size\0",
                    )
                } else {
                    (
                        b"fa3_sm90_fp8_output_dtype\0",
                        b"fa3_sm90_fp8_output_element_size\0",
                    )
                };
                let sym_err = |name: &'static str| RvllmError::Attention {
                    err: AttentionError::Fa3SoMissing { path: path.clone() },
                    ctx: AttnCtx {
                        op: name,
                        stream: 0,
                        num_seqs: 0,
                        head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                };
                let abi_sym: libloading::Symbol<AbiVersionFn> = _lib
                    .get(abi_name)
                    .map_err(|_| sym_err("dlsym:abi_version"))?;
                let abi_version = abi_sym();
                if abi_version != ATTENTION_SO_ABI_VERSION {
                    return Err(authenticated_so_error(
                        &path,
                        format!(
                            "attention ABI version {abi_version} does not match required version {ATTENTION_SO_ABI_VERSION}"
                        ),
                    ));
                }
                let output_dtype =
                    _lib.get::<OutputContractFn>(output_dtype_name)
                        .map_err(|_| sym_err("dlsym:fp8_output_dtype"))?();
                let output_element_bytes = _lib
                    .get::<OutputContractFn>(output_size_name)
                    .map_err(|_| sym_err("dlsym:fp8_output_element_size"))?(
                );
                if output_dtype != FP8_OUTPUT_DTYPE_F16
                    || output_element_bytes != FP8_OUTPUT_ELEMENT_BYTES
                {
                    return Err(authenticated_so_error(
                        &path,
                        format!(
                            "attention FP8 output contract must be F16/{FP8_OUTPUT_ELEMENT_BYTES} bytes, got dtype {output_dtype}/size {output_element_bytes}"
                        ),
                    ));
                }
                if !is_sm89 {
                    let revision_sym: libloading::Symbol<UpstreamRevisionFn> = _lib
                        .get(b"rvllm_fa3_upstream_revision\0")
                        .map_err(|_| sym_err("dlsym:upstream_revision"))?;
                    let revision_ptr = revision_sym();
                    if revision_ptr.is_null()
                        || std::ffi::CStr::from_ptr(revision_ptr).to_bytes()
                            != FA3_UPSTREAM_REVISION
                    {
                        return Err(authenticated_so_error(
                            &path,
                            "FA3 upstream revision does not match the pinned source".into(),
                        ));
                    }
                }
                let workspace_size_dispatch = if is_sm89 {
                    WorkspaceSizeDispatch::Sm89(
                        *_lib
                            .get::<Sm89WorkspaceSizeFn>(b"fa_sm89_decode_workspace_size\0")
                            .map_err(|_| sym_err("dlsym:sm89_workspace_size"))?,
                    )
                } else {
                    WorkspaceSizeDispatch::Sm90 {
                        decode: *_lib
                            .get::<Fa3DecodeWorkspaceSizeFn>(b"fa3_sm90_decode_workspace_size\0")
                            .map_err(|_| sym_err("dlsym:fa3_decode_workspace_size"))?,
                        prefill: *_lib
                            .get::<Fa3PrefillWorkspaceSizeFn>(b"fa3_sm90_prefill_workspace_size\0")
                            .map_err(|_| sym_err("dlsym:fa3_prefill_workspace_size"))?,
                    }
                };
                let dec_sym: libloading::Symbol<PagedDecodeFn> = _lib
                    .get(dec_name)
                    .map_err(|_| sym_err("dlsym:paged_decode"))?;
                let dec_fp8_sym: libloading::Symbol<PagedDecodeFp8Fn> = _lib
                    .get(fp8_name)
                    .map_err(|_| sym_err("dlsym:paged_decode_fp8"))?;
                let fn_paged_prefill_fp8: Option<PagedPrefillFp8Fn> = Some(
                    *_lib
                        .get::<PagedPrefillFp8Fn>(prefill_name)
                        .map_err(|_| sym_err("dlsym:paged_prefill_fp8"))?,
                );
                let fn_paged_decode = *dec_sym;
                let fn_paged_decode_fp8 = *dec_fp8_sym;
                return Ok(Self {
                    so_path: path,
                    head_dim,
                    _lib,
                    workspace_size_dispatch,
                    fn_paged_decode,
                    fn_paged_decode_fp8,
                    fn_paged_prefill_fp8,
                    is_sm89_backend: is_sm89,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self {
            so_path: path,
            head_dim,
            is_sm89_backend: false,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn decode_workspace_size(&self, params: &PagedDecodeParams, is_fp8: bool) -> Result<usize> {
        params.validate()?;
        let bytes = unsafe {
            match self.workspace_size_dispatch {
                WorkspaceSizeDispatch::Sm89(query) => query(
                    params.num_seqs as i32,
                    params.num_heads as i32,
                    params.num_kv_heads as i32,
                    params.head_dim as i32,
                ),
                WorkspaceSizeDispatch::Sm90 { decode, .. } => decode(
                    params.num_seqs as i32,
                    params.num_heads as i32,
                    params.num_kv_heads as i32,
                    params.head_dim as i32,
                    params.block_size as i32,
                    params.max_blocks_per_seq as i32,
                    if is_fp8 { 1 } else { 0 },
                    params.window_size_left,
                ),
            }
        };
        workspace_extent(
            bytes,
            self.is_sm89_backend,
            params.num_seqs,
            params.head_dim,
            "decode workspace query",
        )
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_workspace_size(
        &self,
        params: &PagedPrefillParams,
        max_seqlen_q: u32,
    ) -> Result<usize> {
        params.validate()?;
        if max_seqlen_q == 0 || max_seqlen_q > params.num_tokens || max_seqlen_q > i32::MAX as u32 {
            return Err(RvllmError::Attention {
                err: AttentionError::InvalidParams {
                    reason: "max_seqlen_q must be in 1..=num_tokens and fit the CUDA ABI".into(),
                },
                ctx: AttnCtx {
                    op: "Fa3Kernels::prefill_workspace_size",
                    stream: 0,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let bytes = unsafe {
            match self.workspace_size_dispatch {
                WorkspaceSizeDispatch::Sm89(_) => return Ok(0),
                WorkspaceSizeDispatch::Sm90 { prefill, .. } => prefill(
                    params.num_tokens as i32,
                    max_seqlen_q as i32,
                    params.num_seqs as i32,
                    params.num_heads as i32,
                    params.num_kv_heads as i32,
                    params.head_dim as i32,
                    params.block_size as i32,
                    params.max_blocks_per_seq as i32,
                    1,
                    params.window_size_left,
                ),
            }
        };
        workspace_extent(
            bytes,
            false,
            params.num_seqs,
            params.head_dim,
            "prefill workspace query",
        )
    }
}

#[cfg(feature = "cuda")]
fn workspace_extent(
    bytes: u64,
    allow_zero: bool,
    num_seqs: u32,
    head_dim: u32,
    op: &'static str,
) -> Result<usize> {
    usize::try_from(bytes)
        .ok()
        .filter(|&extent| bytes != u64::MAX && (allow_zero || extent != 0))
        .ok_or_else(|| RvllmError::Attention {
            err: AttentionError::InvalidParams {
                reason: "attention workspace query returned an invalid extent".into(),
            },
            ctx: AttnCtx {
                op,
                stream: 0,
                num_seqs,
                head_dim,
            },
            bt: std::backtrace::Backtrace::capture(),
        })
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn require_workspace_capacity(
    available: usize,
    required: usize,
    op: &'static str,
    num_seqs: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    if available < required {
        return Err(RvllmError::Attention {
            err: AttentionError::InvalidParams {
                reason: format!(
                    "workspace is underallocated: need {required} bytes, got {available}"
                ),
            },
            ctx: AttnCtx {
                op,
                stream,
                num_seqs,
                head_dim,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod workspace_tests {
    use super::require_workspace_capacity;

    #[test]
    fn underallocated_workspace_fails_before_dispatch() {
        assert!(require_workspace_capacity(4095, 4096, "test", 1, 128, 0).is_err());
        assert!(require_workspace_capacity(4096, 4096, "test", 1, 128, 0).is_ok());
    }
}

// ============================================================================
// Fa2PtxKernels — Blackwell attention backend via PTX-launched FA2 kernels
// ============================================================================

/// PTX-based attention backend for Blackwell targets where
/// `libfa3_kernels.so` does not apply (FA3 requires WGMMA + TMA
/// multicast, both Hopper-only). Loads `flash_attention.ptx` via
/// `KernelLoader` and resolves the four entry points we compile:
/// `flash_attention_2_kernel`, `flash_attention_2_decode_kernel`,
/// `flash_attention_2_f16kv_kernel`,
/// `flash_attention_2_decode_f16kv_kernel`.
///
/// Unsupported operations return a typed error.
#[derive(Debug)]
pub struct Fa2PtxKernels {
    pub head_dim: u32,
    /// Tile width compiled into `flash_attention.ptx`: 64 on SM80-SM90,
    /// 32 on SM100/SM121. Host dynamic-shared-memory sizing must match.
    pub f16_tile_cols: u32,
    #[cfg(feature = "cuda")]
    pub flash_attention_mod: rvllm_kernels::LoadedModule,
    #[cfg(feature = "cuda")]
    pub fn_decode: rvllm_kernels::KernelFn,
    /// F16-I/O decode against an f16 paged KV cache. Head dimensions above
    /// 256 are rejected because this kernel exceeds its shared-memory budget.
    #[cfg(feature = "cuda")]
    pub fn_decode_f16io: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill_f16kv: rvllm_kernels::KernelFn,
    /// FP8-E4M3 KV-cache decode with f16 output for the sm_121 ABI.
    #[cfg(feature = "cuda")]
    pub fn_decode_fp8kv: rvllm_kernels::KernelFn,
}

impl Fa2PtxKernels {
    /// Load `flash_attention.ptx` (the FA2 source compiled for this
    /// arch) via the shared `KernelLoader`. Resolves all four entry
    /// points. `head_dim` must be one of the supported values —
    /// mirrors `Fa3Kernels::load` behaviour.
    pub fn load(loader: &rvllm_kernels::KernelLoader, head_dim: u32) -> Result<Self> {
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: AttnCtx {
                    op: "Fa2PtxKernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        let f16_tile_cols = match loader.manifest().arch() {
            "sm_80" | "sm_89" | "sm_90" => 64,
            "sm_100" | "sm_121" => 32,
            _ => {
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "FA2 PTX",
                        op: "unrecognized manifest architecture",
                    },
                    ctx: AttnCtx {
                        op: "Fa2PtxKernels::load",
                        stream: 0,
                        num_seqs: 0,
                        head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        };

        #[cfg(feature = "cuda")]
        {
            let flash_attention_mod = loader.load_ptx("flash_attention")?;
            let fn_decode = flash_attention_mod.get_function("flash_attention_2_decode_kernel")?;
            let fn_decode_f16io =
                flash_attention_mod.get_function("flash_attention_2_decode_f16io_kernel")?;
            let fn_prefill = flash_attention_mod.get_function("flash_attention_2_kernel")?;
            let fn_prefill_f16kv =
                flash_attention_mod.get_function("flash_attention_2_f16kv_kernel")?;
            let fn_decode_fp8kv =
                flash_attention_mod.get_function("flash_attention_2_decode_fp8kv_kernel")?;

            Ok(Self {
                head_dim,
                f16_tile_cols,
                flash_attention_mod,
                fn_decode,
                fn_decode_f16io,
                fn_prefill,
                fn_prefill_f16kv,
                fn_decode_fp8kv,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = loader;
            Ok(Self {
                head_dim,
                f16_tile_cols,
            })
        }
    }
}

// ============================================================================
// AttentionBackend — unifies Fa3 (SM90 dlopen) and Fa2Ptx (sm_121 PTX)
// ============================================================================

/// Which attention backend the runtime is using on the live device.
/// Selected once per `CompileTarget`:
///
///   * SM80 / SM89 → `Fa3` variant carrying the rvLLM `fa_sm89_*` ABI
///   * SM90 → `Fa3` (FA3 for sliding attention, rvLLM fallback for global)
///   * SM100 / SM121 → `Fa2Ptx` (PTX launch plus decode-per-query prefill)
///
/// Callers (launcher structs in `decode.rs` / `prefill.rs`) `match`
/// on this enum and route to the appropriate launch path. An attempt
/// to launch a path that a given backend doesn't implement returns
/// `AttentionError::FeatureNotAvailable` rather than silently
/// succeeding with wrong output.
///
/// `#[non_exhaustive]` so a future SM100-specific backend (or a
/// trait-based dispatch table) can be added without breaking
/// downstream external matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum AttentionBackend {
    Fa3(Fa3Kernels),
    Fa2Ptx(Fa2PtxKernels),
    /// Apple Silicon Metal backend. Constructed once at engine initialization
    /// on macOS aarch64 with the `metal` feature. Dispatched into
    /// `rvllm_metal::paged_attention::call_paged_attention_metal(...)`
    /// at launch time.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    Metal(MetalAttentionKernels),
}

impl AttentionBackend {
    #[cfg(feature = "cuda")]
    pub fn decode_workspace_size(&self, params: &PagedDecodeParams, is_fp8: bool) -> Result<usize> {
        match self {
            AttentionBackend::Fa3(kernels) => kernels.decode_workspace_size(params, is_fp8),
            AttentionBackend::Fa2Ptx(_) => Ok(0),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            AttentionBackend::Metal(_) => Ok(0),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_workspace_size(
        &self,
        params: &PagedPrefillParams,
        max_seqlen_q: u32,
    ) -> Result<usize> {
        match self {
            AttentionBackend::Fa3(kernels) => kernels.prefill_workspace_size(params, max_seqlen_q),
            AttentionBackend::Fa2Ptx(_) => Ok(0),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            AttentionBackend::Metal(_) => Ok(0),
        }
    }

    /// Head dim this backend was constructed for.
    #[must_use]
    pub fn head_dim(&self) -> u32 {
        match self {
            AttentionBackend::Fa3(fa3) => fa3.head_dim,
            AttentionBackend::Fa2Ptx(fa2) => fa2.head_dim,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            AttentionBackend::Metal(m) => m.head_dim,
        }
    }
}

impl From<Fa3Kernels> for AttentionBackend {
    fn from(fa3: Fa3Kernels) -> Self {
        AttentionBackend::Fa3(fa3)
    }
}

impl From<Fa2PtxKernels> for AttentionBackend {
    fn from(fa2: Fa2PtxKernels) -> Self {
        AttentionBackend::Fa2Ptx(fa2)
    }
}

// ============================================================================
// MetalAttentionKernels — Apple Silicon dispatch handle
// ============================================================================

/// Apple Silicon Metal attention dispatch handle. Holds the
/// `MetalDevice` (MTLDevice + primary MTLCommandQueue) and the
/// compiled `MetalKernels` library, both refcounted so multiple
/// launchers can share them without cloning the underlying Metal
/// objects.
///
/// `head_dim` is validated at construction against
/// `SUPPORTED_HEAD_DIMS`; the .metallib in `MetalKernels` is built
/// with the head-size template instantiated to a fixed set (currently
/// `hs128` for Gemma 4 31B — see `KERNEL_NAMES` in `rvllm-metal`),
/// and a mismatched `head_dim` at launch returns
/// `AttentionError::UnsupportedHeadDim`.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub struct MetalAttentionKernels {
    pub device: std::sync::Arc<rvllm_metal::MetalDevice>,
    pub kernels: std::sync::Arc<rvllm_metal::MetalKernels>,
    /// Bridge between the launcher's u64 device-pointer ABI and
    /// rvllm-metal's `&metal::Buffer` API. Every buffer the launcher
    /// references (Q, K-cache, V-cache, block tables, context lens,
    /// output, workspace) must be pre-registered with this registry
    /// before the launcher is called; the runtime owns that
    /// registration. See rvllm-mem/src/metal.rs for the registry API.
    pub registry: std::sync::Arc<rvllm_mem::MetalBufferRegistry>,
    pub head_dim: u32,
    /// I/O dtype for the matmul and cache path.
    pub dtype: rvllm_core::DType,
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl MetalAttentionKernels {
    /// Construct the handle. Validates `head_dim` against the same
    /// supported set as the CUDA backends so the launcher
    /// validation in decode.rs / prefill.rs stays uniform.
    pub fn new(
        device: std::sync::Arc<rvllm_metal::MetalDevice>,
        kernels: std::sync::Arc<rvllm_metal::MetalKernels>,
        registry: std::sync::Arc<rvllm_mem::MetalBufferRegistry>,
        head_dim: u32,
        dtype: rvllm_core::DType,
    ) -> Result<Self> {
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: AttnCtx {
                    op: "MetalAttentionKernels::new",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // This backend uses the BF16 kernel specialization.
        match dtype {
            rvllm_core::DType::Bf16 | rvllm_core::DType::F16 | rvllm_core::DType::F32 => {}
            _ => {
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "Metal",
                        op: "MetalAttentionKernels::new: only BF16/F16/F32 supported in v1 (no FP8 KV)",
                    },
                    ctx: AttnCtx {
                        op: "MetalAttentionKernels::new",
                        stream: 0,
                        num_seqs: 0,
                        head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        Ok(Self {
            device,
            kernels,
            registry,
            head_dim,
            dtype,
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl From<MetalAttentionKernels> for AttentionBackend {
    fn from(m: MetalAttentionKernels) -> Self {
        AttentionBackend::Metal(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_so_rejected_at_load() {
        let err = Fa3Kernels::load("/nonexistent/libfa3_kernels.so".into(), 128).unwrap_err();
        assert!(matches!(
            err,
            RvllmError::Attention {
                err: AttentionError::Fa3SoMissing { .. },
                ..
            }
        ));
    }

    #[test]
    fn unsupported_head_dim_rejected() {
        // Use a non-empty path so head-dimension validation runs first.
        let tmp = std::env::temp_dir().join("fa3-fake.so");
        std::fs::write(&tmp, b"fake").unwrap();
        let err = Fa3Kernels::load(tmp.clone(), 64).unwrap_err();
        std::fs::remove_file(&tmp).ok();
        assert!(matches!(
            err,
            RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim { got: 64, .. },
                ..
            }
        ));
    }
}
