use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
use std::path::PathBuf;
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64"),
    test
))]
use tokenizers::Tokenizer;

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64"),
    test
))]
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;

use crate::{Backend, ServeConfig};

#[derive(Clone)]
pub struct WorkerHandle {
    tx: mpsc::Sender<Job>,
    in_flight: Arc<AtomicUsize>,
    max_inflight: usize,
    max_model_len: usize,
    vocab_size: Option<usize>,
    prompt_logprobs_supported: bool,
    metrics: Arc<ServeMetrics>,
}

/// Engine-side serving counters for `GET /metrics`. Plain atomics, no
/// metrics framework; EMAs are f64 bits in an AtomicU64 (single-writer:
/// the engine worker thread).
#[derive(Default)]
pub struct ServeMetrics {
    pub requests_ok: AtomicU64,
    pub requests_err: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    pub decode_tok_s_ema: AtomicU64,
    pub ttft_ms_ema: AtomicU64,
    pub graph_captures_total: AtomicU64,
}

impl ServeMetrics {
    pub fn load_f64(cell: &AtomicU64) -> f64 {
        f64::from_bits(cell.load(Ordering::Relaxed))
    }

    fn ema_update(cell: &AtomicU64, x: f64) {
        let prev = Self::load_f64(cell);
        let next = if prev == 0.0 { x } else { 0.8 * prev + 0.2 * x };
        cell.store(next.to_bits(), Ordering::Relaxed);
    }

    fn record(&self, out: &GenerateOutput, timing: RequestTiming) {
        self.requests_ok.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(out.prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens_total
            .fetch_add(out.completion_tokens as u64, Ordering::Relaxed);
        if timing.ttft_ms > 0.0 {
            Self::ema_update(&self.ttft_ms_ema, timing.ttft_ms);
        }
        if timing.decode_tokens > 1 && timing.decode_ms > 0.0 {
            // First token belongs to TTFT; the decode rate is the rest.
            let rate = (timing.decode_tokens - 1) as f64 / (timing.decode_ms / 1000.0);
            Self::ema_update(&self.decode_tok_s_ema, rate);
        }
        self.graph_captures_total
            .store(timing.graph_captures_total, Ordering::Relaxed);
    }
}

/// Per-request engine timing, surfaced by the CUDA and Metal engines.
#[derive(Clone, Copy, Default)]
struct RequestTiming {
    ttft_ms: f64,
    decode_ms: f64,
    decode_tokens: usize,
    graph_captures_total: u64,
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// Rendered prompt. May contain `crate::openai::image_slot_marker(i)`
    /// sentinels — the worker substitutes the real
    /// `<boi>{soft × output_length}<eoi>` text at execution time once it
    /// has encoded image `i` and knows its `output_length`.
    pub prompt: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub max_tokens: usize,
    /// Whether to prepend BOS when tokenizing (branch flat decode flag).
    pub add_bos: bool,
    /// Suppress EOS / stop-token termination (branch flat decode flag).
    pub ignore_eos: bool,
    /// Return prompt token logprobs instead of generating. The worker rejects
    /// this unless the loaded engine has full-vocabulary scoring enabled.
    pub prompt_logprobs: bool,
    /// Per-request sampling params (already clamped by openai.rs). Greedy
    /// requests take the exact pre-sampling engine path.
    pub sampling: rvllm_runtime::gemma4_bring_up::SamplingParams,
    /// Bounded `data:` image sources corresponding to the slot markers in
    /// `prompt`. Empty for text-only requests.
    pub images: Vec<String>,
}

/// Why generation ended, in OpenAI `finish_reason` terms: `Stop` for an
/// EOS token, `Length` for hitting the `max_tokens` / context bound.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerateOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Branch path: prompt token logprobs when `prompt_logprobs` was requested;
    /// `None` for ordinary generation.
    pub prompt_logprobs: Option<Vec<Option<f64>>>,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Debug)]
pub struct WorkerStats {
    pub in_flight: usize,
    pub max_inflight: usize,
}

#[derive(Debug)]
pub enum GenerateError {
    Busy { max_inflight: usize },
    Invalid(String),
    Engine(String),
}

struct Job {
    req: GenerateRequest,
    tx: mpsc::Sender<Result<GenerateOutput, String>>,
}

struct EngineMetadata {
    vocab_size: Option<usize>,
    prompt_logprobs_supported: bool,
}

impl WorkerHandle {
    pub fn start(config: ServeConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (init_tx, init_rx) = mpsc::channel::<Result<EngineMetadata, String>>();
        let max_inflight = config.max_inflight_requests;
        let max_model_len = config.max_model_len;
        let metrics = Arc::new(ServeMetrics::default());
        let worker_metrics = Arc::clone(&metrics);

        thread::Builder::new()
            .name("rvllm-engine".into())
            .spawn(move || worker_loop(config, job_rx, init_tx, worker_metrics))
            .map_err(|e| format!("spawn engine worker: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(metadata)) => Ok(Self {
                tx: job_tx,
                in_flight: Arc::new(AtomicUsize::new(0)),
                max_inflight,
                max_model_len,
                vocab_size: metadata.vocab_size,
                prompt_logprobs_supported: metadata.prompt_logprobs_supported,
                metrics,
            }),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(format!("engine worker exited during init: {e}")),
        }
    }

    pub fn metrics(&self) -> &ServeMetrics {
        &self.metrics
    }

    pub fn generate(&self, req: GenerateRequest) -> Result<GenerateOutput, GenerateError> {
        validate_requested_max_tokens(req.max_tokens, self.max_model_len)
            .map_err(GenerateError::Invalid)?;
        if let (Some(ids), Some(vocab_size)) = (req.prompt_token_ids.as_deref(), self.vocab_size) {
            validate_prompt_token_ids(ids, vocab_size).map_err(GenerateError::Invalid)?;
        }
        validate_prompt_logprobs(req.prompt_logprobs, self.prompt_logprobs_supported)
            .map_err(GenerateError::Invalid)?;
        let _slot = self.acquire_slot()?;
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Job { req, tx })
            .map_err(|e| GenerateError::Engine(format!("engine worker is not running: {e}")))?;
        rx.recv()
            .map_err(|e| GenerateError::Engine(format!("engine worker dropped response: {e}")))?
            .map_err(GenerateError::Engine)
    }

    pub fn stats(&self) -> WorkerStats {
        WorkerStats {
            in_flight: self.in_flight.load(Ordering::SeqCst),
            max_inflight: self.max_inflight,
        }
    }

    fn acquire_slot(&self) -> Result<InFlightSlot, GenerateError> {
        let mut cur = self.in_flight.load(Ordering::SeqCst);
        loop {
            if cur >= self.max_inflight {
                return Err(GenerateError::Busy {
                    max_inflight: self.max_inflight,
                });
            }
            match self.in_flight.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    return Ok(InFlightSlot {
                        in_flight: Arc::clone(&self.in_flight),
                    })
                }
                Err(next) => cur = next,
            }
        }
    }
}

struct InFlightSlot {
    in_flight: Arc<AtomicUsize>,
}

impl Drop for InFlightSlot {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn worker_loop(
    config: ServeConfig,
    rx: mpsc::Receiver<Job>,
    init_tx: mpsc::Sender<Result<EngineMetadata, String>>,
    metrics: Arc<ServeMetrics>,
) {
    let mut engine = match EngineState::load(config.clone()) {
        Ok(engine) => {
            let metadata = EngineMetadata {
                vocab_size: engine.vocab_size(),
                prompt_logprobs_supported: engine.prompt_logprobs_supported(),
            };
            let _ = init_tx.send(Ok(metadata));
            engine
        }
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };

    for job in rx {
        let result = engine.generate(job.req);
        let result = match result {
            Ok((out, timing)) => {
                metrics.record(&out, timing);
                Ok(out)
            }
            Err(e) => {
                metrics.requests_err.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        };
        let _ = job.tx.send(result);
    }
}

enum EngineState {
    DryRun(DryRunEngine),
    #[cfg(feature = "cuda")]
    Gemma4(Box<CudaGemma4Engine>),
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    Gemma4Metal(Box<MetalGemma4Engine>),
}

impl EngineState {
    fn load(config: ServeConfig) -> Result<Self, String> {
        if config.dry_run {
            tracing::warn!(
                backend = config.backend.as_str(),
                "RVLLM_DRY_RUN enabled: engine will not be loaded"
            );
            return Ok(Self::DryRun(DryRunEngine { config }));
        }
        match config.backend {
            Backend::Cuda => load_cuda_engine(config),
            Backend::Metal => load_metal_engine(config),
        }
    }

    fn vocab_size(&self) -> Option<usize> {
        match self {
            Self::DryRun(_) => None,
            #[cfg(feature = "cuda")]
            Self::Gemma4(engine) => Some(engine.vocab_size),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Gemma4Metal(engine) => Some(engine.vocab_size),
        }
    }

    fn prompt_logprobs_supported(&self) -> bool {
        match self {
            Self::DryRun(_) => true,
            #[cfg(feature = "cuda")]
            Self::Gemma4(engine) => engine.prompt_logprobs_supported,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Gemma4Metal(_) => false,
        }
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
    ) -> Result<(GenerateOutput, RequestTiming), String> {
        match self {
            EngineState::DryRun(e) => Ok((e.generate(req)?, RequestTiming::default())),
            #[cfg(feature = "cuda")]
            EngineState::Gemma4(e) => e.generate(req),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            EngineState::Gemma4Metal(e) => e.generate(req),
        }
    }
}

struct DryRunEngine {
    config: ServeConfig,
}

impl DryRunEngine {
    fn generate(&self, req: GenerateRequest) -> Result<GenerateOutput, String> {
        if let Some(ms) = std::env::var("RVLLM_DRY_RUN_DELAY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|ms| *ms > 0)
        {
            thread::sleep(std::time::Duration::from_millis(ms));
        }

        let prompt_tokens = req
            .prompt_token_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_else(|| rough_token_count(&req.prompt));
        let completion_tokens =
            bounded_completion_tokens(self.config.max_model_len, prompt_tokens, req.max_tokens)?;
        let prompt_logprobs = if req.prompt_logprobs {
            let mut logprobs = vec![None; prompt_tokens];
            for entry in logprobs.iter_mut().skip(1) {
                *entry = Some(0.0);
            }
            Some(logprobs)
        } else {
            None
        };

        Ok(GenerateOutput {
            text: "RVLLM_DRY_RUN".into(),
            token_ids: vec![0; completion_tokens],
            prompt_tokens,
            completion_tokens,
            prompt_logprobs,
            finish_reason: FinishReason::Length,
        })
    }
}

#[cfg(feature = "cuda")]
struct CudaGemma4Engine {
    config: ServeConfig,
    tokenizer: Tokenizer,
    bringup: rvllm_runtime::gemma4_bring_up::Gemma4Bringup,
    _embedding_mod: rvllm_kernels::LoadedModule,
    fn_embed: rvllm_kernels::KernelFn,
    fn_argmax: rvllm_kernels::KernelFn,
    bos_id: Option<u32>,
    stop_token_ids: Vec<u32>,
    vocab_size: usize,
    prompt_logprobs_supported: bool,
    /// Persistent decode session: regions allocated once, captured graphs
    /// reused across requests, spec decode when RVLLM_SPEC_DECODE=1.
    /// `None` only with the RVLLM_SERVE_SESSION=0 escape hatch, which
    /// restores the legacy per-request `run_generate` path.
    session: Option<rvllm_runtime::gemma4_serve::Gemma4ServeSession>,
    /// Vision pipeline. Present when the binary was built with
    /// `--features vision` AND `--vision-weights-dir` was set at startup.
    /// `None` otherwise — any request with images returns 400.
    #[cfg(feature = "vision")]
    vision: Option<VisionState>,
}

#[cfg(all(feature = "cuda", feature = "vision"))]
struct VisionState {
    ctx: rvllm_vision::VisionContext,
    /// Cached string forms of the three multimodal special tokens, looked
    /// up once at startup from `tokenizer.json#added_tokens`. These are
    /// what the worker splices into the prompt where image slot markers
    /// were emitted by openai.rs.
    boi_str: String,
    soft_str: String,
    eoi_str: String,
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
struct MetalGemma4Engine {
    config: ServeConfig,
    tokenizer: Tokenizer,
    arch: rvllm_loader::gemma4_arch::Gemma4Arch,
    device: Arc<rvllm_metal::MetalDevice>,
    kernels: Arc<rvllm_metal::MetalKernels>,
    weight_cache: Arc<rvllm_loader::metal_loader::MetalWeightCache>,
    host_decoder: rvllm_loader::metal_host::Gemma4HostDecoder,
    bos_id: Option<u32>,
    stop_token_ids: Vec<u32>,
    vocab_size: usize,
}

#[cfg(feature = "cuda")]
fn load_cuda_engine(config: ServeConfig) -> Result<EngineState, String> {
    use rvllm_core::{ModelArch, ModelConfig};
    use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let model_cfg = ModelConfig::load_hf(&model_dir)
        .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?;
    if model_cfg.architecture != ModelArch::Gemma4 {
        return Err(format!(
            "rvllm-server only serves Gemma 4 here; config architecture is {:?}",
            model_cfg.architecture
        ));
    }
    let vocab_size = model_cfg.vocab_size;

    let tokenizer = load_tokenizer(&model_dir)?;

    let paths = Gemma4EnginePaths {
        model_dir,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path_or_placeholder("RVLLM_CUTLASS_SO"),
        fa3_so: env_path_or_placeholder("RVLLM_FA3_SO"),
        policy_json: env_path_or_placeholder("RVLLM_POLICY"),
        // `None` lets Gemma4Bringup resolve RVLLM_W4A8_SO itself.
        w4a8_so: std::env::var_os("RVLLM_W4A8_SO").map(std::path::PathBuf::from),
    };
    let arena_bytes = arena_bytes()?;

    tracing::info!(
        arena_gb = (arena_bytes as f64) / 1_073_741_824.0,
        "loading Gemma 4 engine"
    );
    let bringup =
        Gemma4Bringup::load(paths, arena_bytes).map_err(|e| format!("gemma4 bringup: {e}"))?;
    let prompt_logprobs_supported = bringup.model.pruned_vocab.is_none()
        || std::env::var("RVLLM_PPL_FULLHEAD").ok().as_deref() == Some("1");
    let embedding_mod = bringup
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embedding_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;
    let fn_argmax = bringup.fused.fn_argmax.clone();

    let bos_id = tokenizer.token_to_id("<bos>").or(Some(2));
    let stop_token_ids = stop_token_ids(&tokenizer);
    // Persistent serve session (graph reuse + spec decode). Must be
    // constructed before anything else touches the arena bump pointer.
    let session = if std::env::var("RVLLM_SERVE_SESSION").ok().as_deref() == Some("0") {
        tracing::warn!(
            "RVLLM_SERVE_SESSION=0: legacy per-request run_generate path \
             (graphs re-captured every request, no spec decode)"
        );
        None
    } else {
        Some(
            rvllm_runtime::gemma4_serve::Gemma4ServeSession::new(&bringup, config.max_model_len)
                .map_err(|e| format!("serve session init: {e}"))?,
        )
    };

    #[cfg(feature = "vision")]
    let vision = load_vision_state(&config, &tokenizer)?;

    tracing::info!(
        model = %config.served_model_name,
        stop_ids = ?stop_token_ids,
        spec_decode = session.as_ref().map(|s| s.spec_enabled()).unwrap_or(false),
        "Gemma 4 engine ready"
    );
    Ok(EngineState::Gemma4(Box::new(CudaGemma4Engine {
        config,
        tokenizer,
        bringup,
        _embedding_mod: embedding_mod,
        fn_embed,
        fn_argmax,
        bos_id,
        stop_token_ids,
        vocab_size,
        prompt_logprobs_supported,
        session,
        #[cfg(feature = "vision")]
        vision,
    })))
}

#[cfg(not(feature = "cuda"))]
fn load_cuda_engine(_config: ServeConfig) -> Result<EngineState, String> {
    Err(
        "rvllm-server was built without --features cuda; set RVLLM_DRY_RUN=1 for bind-only checks"
            .into(),
    )
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn load_metal_engine(config: ServeConfig) -> Result<EngineState, String> {
    use rvllm_core::{ModelArch, ModelConfig};

    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let model_cfg = ModelConfig::load_hf(&model_dir)
        .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?;
    if model_cfg.architecture != ModelArch::Gemma4 {
        return Err(format!(
            "rvllm-server only serves Gemma 4 here; config architecture is {:?}",
            model_cfg.architecture
        ));
    }
    let vocab_size = model_cfg.vocab_size;

    let tokenizer = load_tokenizer(&model_dir)?;

    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .map_err(|e| format!("gemma4 arch parse {}: {e}", model_dir.display()))?;
    let device = Arc::new(
        rvllm_metal::MetalDevice::system_default().map_err(|e| format!("metal device: {e}"))?,
    );
    let kernels = Arc::new(
        rvllm_metal::MetalKernels::new(&device).map_err(|e| format!("metal kernels: {e}"))?,
    );
    let weight_cache = Arc::new(
        rvllm_loader::metal_loader::MetalWeightCache::from_dir_env(
            &model_dir,
            Arc::clone(&device),
            arch.num_hidden_layers,
        )
        .map_err(|e| format!("metal weight cache: {e}"))?,
    );
    let host_decoder = rvllm_loader::metal_host::Gemma4HostDecoder::new_with_metal(
        arch.clone(),
        config.max_model_len,
        Arc::clone(&device),
        Arc::clone(&kernels),
    )
    .map_err(|e| format!("metal decoder: {e}"))?;
    let bos_id = tokenizer.token_to_id("<bos>").or(Some(2));
    let stop_token_ids = stop_token_ids(&tokenizer);

    tracing::info!(
        model = %config.served_model_name,
        stop_ids = ?stop_token_ids,
        device = ?device,
        layers = arch.num_hidden_layers,
        max_cached_layers = weight_cache.max_cached_layers(),
        resident_layers = weight_cache.resident_layer_count(),
        weight_prefix = weight_cache.weight_prefix(),
        "Gemma 4 Metal host decoder ready"
    );

    Ok(EngineState::Gemma4Metal(Box::new(MetalGemma4Engine {
        config,
        tokenizer,
        arch,
        device,
        kernels,
        weight_cache,
        host_decoder,
        bos_id,
        stop_token_ids,
        vocab_size,
    })))
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn load_metal_engine(_config: ServeConfig) -> Result<EngineState, String> {
    Err(
        "rvllm-server was built without Apple Metal support; build on macOS aarch64 with --features metal"
            .into(),
    )
}

#[cfg(all(feature = "cuda", feature = "vision"))]
fn load_vision_state(
    config: &ServeConfig,
    tokenizer: &Tokenizer,
) -> Result<Option<VisionState>, String> {
    let Some(dir) = config.vision_weights_dir.as_ref() else {
        return Ok(None);
    };
    // CPU device is the safe baseline; CUDA/Metal can be enabled by the
    // vision-feature build via candle features. Selecting per-build is V1's
    // call — we just hand it whatever default it picks.
    let device = candle_core::Device::Cpu;
    let ctx = rvllm_vision::VisionContext::load(dir, device)
        .map_err(|e| format!("vision context load {}: {e}", dir.display()))?;

    let boi_id = ctx.boi_token_id();
    let soft_id = ctx.image_token_id();
    let eoi_id = ctx.eoi_token_id();
    let boi_str = lookup_special_token_str(tokenizer, boi_id, "boi")?;
    let soft_str = lookup_special_token_str(tokenizer, soft_id, "image_soft_token")?;
    let eoi_str = lookup_special_token_str(tokenizer, eoi_id, "eoi")?;
    tracing::info!(
        boi_id, soft_id, eoi_id, %boi_str, %soft_str, %eoi_str,
        "vision context loaded"
    );
    Ok(Some(VisionState {
        ctx,
        boi_str,
        soft_str,
        eoi_str,
    }))
}

/// Reverse-look-up a token id to its rendered string. The Gemma 4 tokenizer
/// has the three multimodal special tokens (`<boi>` = 255999,
/// `<image_soft_token>` = 258880, `<eoi>` = 258882) in its added_tokens; we
/// resolve them by id rather than by hard-coded string so a tokenizer
/// re-export with renamed surface forms still works.
#[cfg(all(feature = "cuda", feature = "vision"))]
fn lookup_special_token_str(tokenizer: &Tokenizer, id: u32, name: &str) -> Result<String, String> {
    tokenizer
        .id_to_token(id)
        .ok_or_else(|| format!("tokenizer has no token with id {id} (expected {name})"))
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl MetalGemma4Engine {
    fn generate(
        &mut self,
        req: GenerateRequest,
    ) -> Result<(GenerateOutput, RequestTiming), String> {
        if !req.images.is_empty() {
            return Err(
                "Gemma 4 Metal backend has safetensors + OpenAI chat wiring, but image embedding injection is not wired yet"
                    .into(),
            );
        }
        if req.prompt_logprobs {
            return Err("Gemma 4 Metal backend does not expose prompt_logprobs yet".into());
        }
        if !req.sampling.is_greedy() {
            return Err(
                "Gemma 4 Metal host decoder currently supports greedy generation only; set temperature=0"
                    .into(),
            );
        }

        let mut prompt_ids = if let Some(ids) = req.prompt_token_ids {
            ids
        } else {
            let encoding = self
                .tokenizer
                .encode(req.prompt.as_str(), false)
                .map_err(|e| format!("tokenize: {e}"))?;
            encoding.get_ids().to_vec()
        };
        if req.add_bos {
            if let Some(bos) = self.bos_id {
                if prompt_ids.first().copied() != Some(bos) {
                    prompt_ids.insert(0, bos);
                }
            }
        }
        validate_prompt_token_ids(&prompt_ids, self.vocab_size)?;
        if prompt_ids.len() >= self.config.max_model_len {
            return Err(format!(
                "prompt has {} tokens, max_model_len is {}",
                prompt_ids.len(),
                self.config.max_model_len
            ));
        }
        if req
            .max_tokens
            .min(self.config.max_model_len - prompt_ids.len())
            == 0
        {
            return Err("max_tokens leaves no decode room under max_model_len".into());
        }

        let max_new = req
            .max_tokens
            .min(self.config.max_model_len - prompt_ids.len());
        let stop_token_ids = if req.ignore_eos {
            &[][..]
        } else {
            self.stop_token_ids.as_slice()
        };
        let (output_ids, timing) = self
            .host_decoder
            .generate_timed(&self.weight_cache, &prompt_ids, max_new, stop_token_ids)
            .map_err(|e| format!("gemma4 metal host decode: {e}"))?;

        let text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| format!("detokenize: {e}"))?;
        let hit_eos = !req.ignore_eos
            && output_ids
                .last()
                .is_some_and(|t| self.stop_token_ids.contains(t));
        let finish_reason = if hit_eos {
            FinishReason::Stop
        } else if output_ids.len() >= max_new {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };

        let _keep_alive = (&self.device, &self.kernels, &self.arch);
        Ok((
            GenerateOutput {
                text,
                token_ids: output_ids.clone(),
                prompt_tokens: prompt_ids.len(),
                completion_tokens: output_ids.len(),
                prompt_logprobs: None,
                finish_reason,
            },
            RequestTiming {
                ttft_ms: timing.ttft_ms,
                decode_ms: timing.decode_ms,
                decode_tokens: timing.decode_tokens,
                graph_captures_total: 0,
            },
        ))
    }
}

#[cfg(feature = "cuda")]
impl CudaGemma4Engine {
    fn generate(
        &mut self,
        req: GenerateRequest,
    ) -> Result<(GenerateOutput, RequestTiming), String> {
        if req.prompt_logprobs {
            if !req.images.is_empty() {
                return Err("prompt_logprobs is only supported for text prompts".into());
            }
            let token_ids = req
                .prompt_token_ids
                .as_deref()
                .ok_or("prompt_logprobs requires prompt_token_ids")?;
            validate_prompt_token_ids(token_ids, self.vocab_size)?;
            return Ok((self.prompt_logprobs(token_ids)?, RequestTiming::default()));
        }

        // Text-only fast path: prompt is already free of slot markers
        // because openai.rs only emits them when an image_url part was
        // present. We tokenize the prompt verbatim and pass empty
        // image_embeds to the runtime — same byte-for-byte path as before
        // the vision graft.
        if req.images.is_empty() {
            return self.generate_text_only(&req);
        }

        #[cfg(not(feature = "vision"))]
        {
            return Err(
                "request contains images but rvllm-server was built without --features vision"
                    .into(),
            );
        }

        #[cfg(feature = "vision")]
        {
            self.generate_with_images(req)
        }
    }

    /// Run the persistent session (graph reuse across requests, spec
    /// decode when enabled) or, behind RVLLM_SERVE_SESSION=0, the legacy
    /// per-request run_generate path. `stop_token_ids` is the caller-resolved
    /// stop set (empty when the branch `ignore_eos` flag is set), so the
    /// session/legacy/sampled tails all honour ignore_eos uniformly.
    fn run_engine(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        image_embeds: &[(usize, Vec<f32>)],
        stop_token_ids: &[u32],
        sampling: rvllm_runtime::gemma4_bring_up::SamplingParams,
    ) -> Result<(Vec<u32>, RequestTiming), String> {
        validate_prompt_token_ids(prompt_ids, self.vocab_size)?;
        // Sampled requests take the eager sampled tail (no persistent
        // session: the captured greedy graph ends in argmax + device
        // token feedback, which sampling replaces). Greedy requests ride
        // the session: graph reuse across requests + optional spec.
        if !sampling.is_greedy() {
            let ids = unsafe {
                self.bringup.run_generate_sampled(
                    &self.fn_embed,
                    &self.fn_argmax,
                    prompt_ids,
                    max_new,
                    stop_token_ids,
                    image_embeds,
                    sampling,
                )
            }
            .map_err(|e| format!("gemma4 generate (sampled): {e}"))?;
            return Ok((ids, RequestTiming::default()));
        }
        if let Some(session) = self.session.as_mut() {
            let out = unsafe {
                session.generate(
                    &self.bringup,
                    &self.fn_embed,
                    &self.fn_argmax,
                    prompt_ids,
                    max_new,
                    stop_token_ids,
                    image_embeds,
                )
            }
            .map_err(|e| format!("gemma4 serve session: {e}"))?;
            let timing = RequestTiming {
                ttft_ms: out.prefill_ms,
                decode_ms: out.decode_ms,
                decode_tokens: out.ids.len(),
                graph_captures_total: session.graph_captures_total(),
            };
            Ok((out.ids, timing))
        } else {
            let ids = unsafe {
                self.bringup.run_generate(
                    &self.fn_embed,
                    &self.fn_argmax,
                    prompt_ids,
                    max_new,
                    stop_token_ids,
                    image_embeds,
                )
            }
            .map_err(|e| format!("gemma4 generate: {e}"))?;
            Ok((ids, RequestTiming::default()))
        }
    }

    fn generate_text_only(
        &mut self,
        req: &GenerateRequest,
    ) -> Result<(GenerateOutput, RequestTiming), String> {
        let prompt_ids = self.prompt_ids_for(req)?;
        let max_new = self.bound_max_new(&prompt_ids, req.max_tokens)?;
        // Branch ignore_eos: suppress the stop set so generation runs to the
        // token budget. Otherwise use the configured EOS / stop tokens.
        let stop_token_ids: Vec<u32> = if req.ignore_eos {
            Vec::new()
        } else {
            self.stop_token_ids.clone()
        };
        let (output_ids, timing) =
            self.run_engine(&prompt_ids, max_new, &[], &stop_token_ids, req.sampling)?;
        Ok((
            self.finalize_output(&prompt_ids, &output_ids, &stop_token_ids, max_new)?,
            timing,
        ))
    }

    /// Use caller-supplied token IDs verbatim or tokenize the rendered prompt.
    fn prompt_ids_for(&self, req: &GenerateRequest) -> Result<Vec<u32>, String> {
        if let Some(ids) = req.prompt_token_ids.as_ref() {
            validate_prompt_token_ids(ids, self.vocab_size)?;
            return Ok(ids.clone());
        }
        self.tokenize_text_prompt(&req.prompt, req.add_bos)
    }

    fn prompt_logprobs(&mut self, prompt_ids: &[u32]) -> Result<GenerateOutput, String> {
        let result = unsafe { self.bringup.run_ppl(&self.fn_embed, prompt_ids, 0) }
            .map_err(|e| format!("gemma4 prompt_logprobs: {e}"))?;
        Ok(GenerateOutput {
            text: String::new(),
            token_ids: Vec::new(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            prompt_logprobs: Some(result.token_logprobs),
            finish_reason: FinishReason::Stop,
        })
    }

    #[cfg(feature = "vision")]
    fn generate_with_images(
        &mut self,
        req: GenerateRequest,
    ) -> Result<(GenerateOutput, RequestTiming), String> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            "request contains images but --vision-weights-dir was not configured at startup; \
             pass --vision-weights-dir <PATH> to enable image input"
                .to_string()
        })?;

        // Encode every image up front: we need the per-image output_length
        // before we can splice the right number of soft tokens into the
        // prompt, and we need the concatenated embedding buffer before we
        // can pair embedding rows with their post-tokenization positions.
        let mut all_embeds: Vec<f32> = Vec::new();
        let mut per_image_token_str: Vec<String> = Vec::with_capacity(req.images.len());
        for (i, source) in req.images.iter().enumerate() {
            let dynamic = rvllm_imageio::parse_image_url(source)
                .map_err(|e| format!("image {i} decode: {e}"))?;
            let (embed, output_length) = vision
                .ctx
                .encode_image(&dynamic)
                .map_err(|e| format!("image {i} encode: {e}"))?;
            let expected = output_length
                .checked_mul(vision.ctx.text_hidden_size)
                .ok_or_else(|| format!("image {i}: output_length overflow"))?;
            if embed.len() != expected {
                return Err(format!(
                    "image {i}: encoder returned {} floats but expected output_length ({}) * text_hidden_size ({}) = {}",
                    embed.len(),
                    output_length,
                    vision.ctx.text_hidden_size,
                    expected
                ));
            }
            all_embeds.extend_from_slice(&embed);
            per_image_token_str.push(
                rvllm_vision::chat::build_image_token_string(
                    &vision.boi_str,
                    &vision.soft_str,
                    &vision.eoi_str,
                    output_length,
                )
                .map_err(|e| format!("image {i}: {e}"))?,
            );
        }

        // Splice the real `<boi>{soft × N}<eoi>` strings into the prompt
        // at the slot markers openai.rs emitted. Each marker appears
        // exactly once, in slot order.
        let mut spliced = req.prompt.clone();
        for (i, replacement) in per_image_token_str.iter().enumerate() {
            let marker = crate::openai::image_slot_marker(i);
            let pos = spliced.find(&marker).ok_or_else(|| {
                format!(
                    "image slot marker {i} not found in rendered prompt; \
                     openai.rs and worker.rs are out of sync"
                )
            })?;
            spliced.replace_range(pos..pos + marker.len(), replacement);
        }
        // Sanity: every slot must have been substituted. If a marker
        // remains, fail loudly so a bug here can't silently corrupt
        // generation.
        if spliced.contains(crate::openai::IMAGE_SLOT_MARK) {
            return Err(
                "rendered prompt still contains an image slot marker after substitution"
                    .to_string(),
            );
        }

        let prompt_ids = self.tokenize_text_prompt(&spliced, req.add_bos)?;
        let max_new = self.bound_max_new(&prompt_ids, req.max_tokens)?;
        // Owned stop set: cloned so it does not borrow `self` across the
        // `&mut self` `run_engine` call. Branch ignore_eos => empty set.
        let stop_token_ids: Vec<u32> = if req.ignore_eos {
            Vec::new()
        } else {
            self.stop_token_ids.clone()
        };

        // Build per-position embedding chunks. We walk the token id stream
        // once; every time we hit the image-soft-token id we pop the next
        // text_hidden_size floats off the head of `all_embeds`.
        let text_hidden = vision.ctx.text_hidden_size;
        let soft_id = vision.ctx.image_token_id();
        let mut image_embeds: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut cursor = 0usize;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            if tok == soft_id {
                let end = cursor + text_hidden;
                if end > all_embeds.len() {
                    return Err(format!(
                        "ran out of image embedding floats at token position {i}: \
                         have {} floats, need {} more (text_hidden_size = {})",
                        all_embeds.len() - cursor,
                        text_hidden,
                        text_hidden
                    ));
                }
                let chunk = all_embeds[cursor..end].to_vec();
                cursor = end;
                image_embeds.push((i, chunk));
            }
        }
        if cursor != all_embeds.len() {
            return Err(format!(
                "image embeddings have {} leftover floats after pairing with soft-token positions; \
                 token id {} appears fewer times in the tokenized prompt than expected",
                all_embeds.len() - cursor,
                soft_id
            ));
        }

        let (output_ids, timing) = self
            .run_engine(
                &prompt_ids,
                max_new,
                &image_embeds,
                &stop_token_ids,
                req.sampling,
            )
            .map_err(|e| format!("vision generation: {e}"))?;

        Ok((
            self.finalize_output(&prompt_ids, &output_ids, &stop_token_ids, max_new)?,
            timing,
        ))
    }

    fn tokenize_text_prompt(&self, prompt: &str, add_bos: bool) -> Result<Vec<u32>, String> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut prompt_ids = encoding.get_ids().to_vec();
        if add_bos {
            if let Some(bos) = self.bos_id {
                if prompt_ids.first().copied() != Some(bos) {
                    prompt_ids.insert(0, bos);
                }
            }
        }
        validate_prompt_token_ids(&prompt_ids, self.vocab_size)?;
        Ok(prompt_ids)
    }

    fn bound_max_new(&self, prompt_ids: &[u32], max_tokens: usize) -> Result<usize, String> {
        if prompt_ids.len() >= self.config.max_model_len {
            return Err(format!(
                "prompt has {} tokens, max_model_len is {}",
                prompt_ids.len(),
                self.config.max_model_len
            ));
        }
        let max_new = max_tokens.min(self.config.max_model_len.saturating_sub(prompt_ids.len()));
        if max_new == 0 {
            return Err("max_tokens leaves no decode room under max_model_len".into());
        }
        Ok(max_new)
    }

    fn finalize_output(
        &self,
        prompt_ids: &[u32],
        output_ids: &[u32],
        stop_token_ids: &[u32],
        max_new: usize,
    ) -> Result<GenerateOutput, String> {
        let prompt_len = prompt_ids.len();
        let text = self
            .tokenizer
            .decode(output_ids, true)
            .map_err(|e| format!("detokenize: {e}"))?;
        // EOS wins even at the max_tokens boundary; only an unstopped run
        // into the token budget is "length".
        let hit_eos = output_ids
            .last()
            .is_some_and(|t| stop_token_ids.contains(t));
        let finish_reason = if hit_eos {
            FinishReason::Stop
        } else if output_ids.len() >= max_new {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        Ok(GenerateOutput {
            text,
            token_ids: output_ids.to_vec(),
            prompt_tokens: prompt_len,
            completion_tokens: output_ids.len(),
            prompt_logprobs: None,
            finish_reason,
        })
    }
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|_| format!("missing env var: {name}"))
}

#[cfg(feature = "cuda")]
fn env_path_or_placeholder(name: &str) -> PathBuf {
    std::env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
}

#[cfg(feature = "cuda")]
fn arena_bytes() -> Result<usize, String> {
    if let Ok(value) = std::env::var("RVLLM_ARENA_GB") {
        let gb = value
            .parse::<usize>()
            .map_err(|_| "RVLLM_ARENA_GB must be a positive integer")?;
        return arena_bytes_from_gb(gb);
    }

    let mut free: usize = 0;
    let mut total: usize = 0;
    let probe_ctx =
        rvllm_mem::context::CudaContextHandle::init(0).map_err(|e| format!("probe ctx: {e}"))?;
    let status =
        unsafe { cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemGetInfo_v2 failed: {status:?}"));
    }
    drop(probe_ctx);

    let reserve = 512 * 1024 * 1024;
    Ok(if free > reserve { free - reserve } else { free })
}

#[cfg(any(feature = "cuda", test))]
fn arena_bytes_from_gb(gb: usize) -> Result<usize, String> {
    if gb == 0 {
        return Err("RVLLM_ARENA_GB must be greater than zero".into());
    }
    gb.checked_mul(1usize << 30)
        .ok_or_else(|| "RVLLM_ARENA_GB byte count overflow".into())
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn stop_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    if let Ok(raw) = std::env::var("RVLLM_EOS") {
        let ids: Vec<u32> = raw
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        if !ids.is_empty() {
            return ids;
        }
    }

    let mut ids = Vec::new();
    for token in ["<turn|>", "<eos>", "<|tool_response>", "</s>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    if ids.is_empty() {
        ids.push(107);
    }
    ids
}

fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn validate_requested_max_tokens(max_tokens: usize, max_model_len: usize) -> Result<(), String> {
    if !(1..=max_model_len).contains(&max_tokens) {
        return Err(format!(
            "max_tokens must be in 1..={max_model_len} for this model"
        ));
    }
    Ok(())
}

fn bounded_completion_tokens(
    max_model_len: usize,
    prompt_tokens: usize,
    max_tokens: usize,
) -> Result<usize, String> {
    if prompt_tokens >= max_model_len {
        return Err(format!(
            "prompt has {prompt_tokens} tokens, max_model_len is {max_model_len}"
        ));
    }
    let completion_tokens = max_tokens.min(max_model_len - prompt_tokens);
    if completion_tokens == 0 {
        return Err("max_tokens leaves no decode room under max_model_len".into());
    }
    Ok(completion_tokens)
}

fn validate_prompt_token_ids(ids: &[u32], vocab_size: usize) -> Result<(), String> {
    if ids.is_empty() {
        return Err("prompt token IDs must not be empty".into());
    }
    if vocab_size == 0 || ids.iter().any(|&id| id as usize >= vocab_size) {
        return Err("prompt token ID is outside the loaded model vocabulary".into());
    }
    Ok(())
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64"),
    test
))]
fn load_tokenizer(model_dir: &std::path::Path) -> Result<Tokenizer, String> {
    use std::io::Read;

    let root = model_dir
        .canonicalize()
        .map_err(|e| format!("model root {}: {e}", model_dir.display()))?;
    let path = root
        .join("tokenizer.json")
        .canonicalize()
        .map_err(|e| format!("tokenizer path {}: {e}", root.display()))?;
    if !path.starts_with(&root) {
        return Err(format!(
            "tokenizer path {} escapes model root {}",
            path.display(),
            root.display()
        ));
    }
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("tokenizer open {}: {e}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("tokenizer metadata {}: {e}", path.display()))?;
    if !metadata.is_file() || metadata.len() > MAX_TOKENIZER_BYTES {
        return Err(format!(
            "tokenizer {} must be a file no larger than {MAX_TOKENIZER_BYTES} bytes",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_TOKENIZER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("tokenizer read {}: {e}", path.display()))?;
    if bytes.len() as u64 > MAX_TOKENIZER_BYTES {
        return Err(format!(
            "tokenizer {} exceeds {MAX_TOKENIZER_BYTES} bytes",
            path.display()
        ));
    }
    Tokenizer::from_bytes(bytes).map_err(|e| format!("tokenizer parse {}: {e}", path.display()))
}

fn validate_prompt_logprobs(requested: bool, supported: bool) -> Result<(), String> {
    if requested && !supported {
        return Err(
            "prompt_logprobs is unavailable for the loaded model without full-vocabulary scoring"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        arena_bytes_from_gb, bounded_completion_tokens, load_tokenizer, validate_prompt_logprobs,
        validate_prompt_token_ids, validate_requested_max_tokens, DryRunEngine, FinishReason,
        GenerateRequest, MAX_TOKENIZER_BYTES,
    };
    use crate::{Backend, ServeConfig};
    use rvllm_runtime::gemma4_bring_up::SamplingParams;

    #[test]
    fn dry_run_budget_exhaustion_reports_length() {
        let engine = DryRunEngine {
            config: ServeConfig {
                backend: Backend::Cuda,
                host: "127.0.0.1".into(),
                port: 8080,
                served_model_name: "dry-run".into(),
                default_system_prompt: None,
                max_model_len: 8,
                max_num_seqs: 1,
                max_inflight_requests: 1,
                max_num_batched_tokens: 8,
                max_prefill_chunk: 8,
                dry_run: true,
                api_key: None,
                vision_weights_dir: None,
            },
        };
        let output = engine
            .generate(GenerateRequest {
                prompt: "one two".into(),
                prompt_token_ids: None,
                max_tokens: 2,
                add_bos: true,
                ignore_eos: false,
                prompt_logprobs: false,
                sampling: SamplingParams::greedy(),
                images: Vec::new(),
            })
            .unwrap();
        assert_eq!(output.completion_tokens, 2);
        assert_eq!(output.finish_reason, FinishReason::Length);
    }

    #[test]
    fn arena_gib_conversion_is_checked() {
        assert_eq!(arena_bytes_from_gb(1).unwrap(), 1usize << 30);
        assert!(arena_bytes_from_gb(0).is_err());
        assert!(arena_bytes_from_gb(usize::MAX).is_err());
    }

    #[test]
    fn request_token_budget_is_bounded_before_generation() {
        assert!(validate_requested_max_tokens(usize::MAX, 8192).is_err());
        assert!(validate_requested_max_tokens(0, 8192).is_err());
        assert_eq!(bounded_completion_tokens(8, 6, 8).unwrap(), 2);
        assert!(bounded_completion_tokens(8, 8, 1).is_err());
    }

    #[test]
    fn pretokenized_prompts_must_fit_loaded_vocabulary() {
        assert!(validate_prompt_token_ids(&[0, 7], 8).is_ok());
        assert!(validate_prompt_token_ids(&[8], 8).is_err());
        assert_eq!(
            validate_prompt_token_ids(&[], 8).unwrap_err(),
            "prompt token IDs must not be empty"
        );
    }

    #[test]
    fn prompt_logprobs_require_full_vocabulary_scoring() {
        assert!(validate_prompt_logprobs(false, false).is_ok());
        assert!(validate_prompt_logprobs(true, true).is_ok());
        assert!(validate_prompt_logprobs(true, false).is_err());
    }

    #[test]
    fn tokenizer_load_is_bounded_and_root_contained() {
        let root = tempdir("root");
        let tokenizer =
            tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        tokenizer.save(root.join("tokenizer.json"), false).unwrap();
        load_tokenizer(&root).unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        let root = tempdir("oversized");
        let file = std::fs::File::create(root.join("tokenizer.json")).unwrap();
        file.set_len(MAX_TOKENIZER_BYTES + 1).unwrap();
        assert!(load_tokenizer(&root).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn tokenizer_symlink_cannot_escape_model_root() {
        let root = tempdir("contained");
        let outside = tempdir("outside");
        let tokenizer =
            tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        tokenizer
            .save(outside.join("tokenizer.json"), false)
            .unwrap();
        std::os::unix::fs::symlink(outside.join("tokenizer.json"), root.join("tokenizer.json"))
            .unwrap();
        assert!(load_tokenizer(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(outside).unwrap();
    }

    fn tempdir(label: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rvllm-tokenizer-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
