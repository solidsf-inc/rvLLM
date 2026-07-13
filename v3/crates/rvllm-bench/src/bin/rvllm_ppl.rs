//! Sliding-window Gemma 4 perplexity measurement.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rvllm_core::{ModelArch, ModelConfig};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use sha2::{Digest, Sha256};

const MAX_TEXT_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_ARENA_GIB: usize = 256;
const MAX_METADATA_TEXT: usize = 256;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(error) = run() {
        eprintln!("rvllm-ppl: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let config = ModelConfig::load_hf(&model_dir)
        .map_err(|error| format!("model config {}: {error}", model_dir.display()))?;
    if config.architecture != ModelArch::Gemma4 {
        return Err("rvllm-ppl supports Gemma 4 models only".into());
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_bytes = read_bounded(&tokenizer_path, MAX_TOKENIZER_BYTES, "tokenizer")?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| format!("tokenizer {}: {error}", tokenizer_path.display()))?;
    let bos_id = bos_token_id(&model_dir, &tokenizer)?;
    if bos_id as usize >= config.vocab_size {
        return Err(format!(
            "BOS token {bos_id} is outside vocabulary {}",
            config.vocab_size
        ));
    }

    let text = read_text()?;
    let input_sha256 = sha256_bytes(text.as_bytes());
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let raw = encoding.get_ids();
    if raw.is_empty() {
        return Err("input produced no scoreable tokens".into());
    }
    if raw.iter().any(|token| *token as usize >= config.vocab_size) {
        return Err("tokenizer emitted an ID outside the model vocabulary".into());
    }

    let mut sequence = Vec::with_capacity(raw.len().saturating_add(1));
    sequence.push(bos_id);
    sequence.extend_from_slice(raw);
    let max_window = config.max_position_embeddings;
    let requested_window = env_usize("RVLLM_PPL_CHUNK", 2048)?;
    if requested_window < 2 || requested_window > max_window {
        return Err(format!("RVLLM_PPL_CHUNK must be in 2..={max_window}"));
    }
    let window = requested_window.min(sequence.len());
    let default_stride = (window / 2).max(1);
    let stride = env_usize("RVLLM_PPL_STRIDE", default_stride)?;
    if !(1..=window).contains(&stride) {
        return Err(format!("RVLLM_PPL_STRIDE must be in 1..={window}"));
    }
    if sequence.len() > window && stride == window {
        return Err(
            "RVLLM_PPL_STRIDE must be less than the window for multi-window scoring".into(),
        );
    }
    let max_windows = env_usize("RVLLM_PPL_CHUNKS", 0)?;

    let paths = Gemma4EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
        w4a8_so: None,
    };
    let metadata = receipt_metadata()?;
    let artifacts = artifact_digests(&paths)?;
    let arena_bytes = arena_bytes()?;
    let load_started = Instant::now();
    let engine = Gemma4Bringup::load(paths, arena_bytes)
        .map_err(|error| format!("Gemma 4 bringup: {error}"))?;
    if engine.arch.is_e4b() {
        return Err("this Gemma 4 variant is not supported by the public evaluator".into());
    }
    eprintln!("bringup_s={:.3}", load_started.elapsed().as_secs_f64());

    let embedding_module = engine
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|error| format!("load embedding kernel: {error}"))?;
    let embedding = embedding_module
        .get_function("embedding_gather_f16_kernel")
        .map_err(|error| format!("resolve embedding kernel: {error}"))?;

    let started = Instant::now();
    let mut total_nll = 0.0f64;
    let mut total_tokens = 0usize;
    let plan = ppl_windows(sequence.len(), window, stride, max_windows)?;

    for (window_index, &(begin, end, score_from)) in plan.iter().enumerate() {
        let tokens = &sequence[begin..end];
        let result = unsafe { engine.run_ppl(&embedding, tokens, score_from) }
            .map_err(|error| format!("run_ppl window {window_index}: {error}"))?;
        if result.n_evaluated == 0 || !result.total_nll.is_finite() {
            return Err(format!("window {window_index} returned an invalid score"));
        }
        total_nll += result.total_nll;
        total_tokens = total_tokens
            .checked_add(result.n_evaluated)
            .ok_or("evaluated token count overflow")?;
        if !total_nll.is_finite() {
            return Err("accumulated NLL is not finite".into());
        }
    }
    let windows = plan.len();

    if total_tokens == 0 {
        return Err("no tokens were evaluated".into());
    }
    let perplexity = (total_nll / total_tokens as f64).exp();
    if !perplexity.is_finite() {
        return Err("perplexity is not finite".into());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let result = serde_json::json!({
        "mode": "sliding_window_perplexity",
        "perplexity": perplexity,
        "total_nll": total_nll,
        "tokens": total_tokens,
        "window": window,
        "stride": stride,
        "windows": windows,
        "elapsed_s": elapsed,
        "tokens_per_second": if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 },
        "input_tokens": sequence.len(),
        "input_sha256": input_sha256,
        "max_windows": max_windows,
        "arena_bytes": arena_bytes,
    });
    println!("{}", finalize_receipt(result, metadata, artifacts)?);
    Ok(())
}

fn read_text() -> Result<String, String> {
    if let Ok(path) = std::env::var("RVLLM_PPL_TEXT") {
        let path = PathBuf::from(path);
        let text = read_bounded_utf8(&path, MAX_TEXT_BYTES, "input")?;
        if text.is_empty() {
            return Err("input text is empty".into());
        }
        return Ok(text);
    }
    if let Ok(text) = std::env::var("RVLLM_PROMPT") {
        if text.len() as u64 > MAX_TEXT_BYTES {
            return Err(format!("RVLLM_PROMPT exceeds {MAX_TEXT_BYTES} bytes"));
        }
        if text.is_empty() {
            return Err("RVLLM_PROMPT is empty".into());
        }
        return Ok(text);
    }
    let mut text = String::new();
    std::io::stdin()
        .take(MAX_TEXT_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| format!("stdin: {error}"))?;
    if text.len() as u64 > MAX_TEXT_BYTES {
        return Err(format!("stdin exceeds {MAX_TEXT_BYTES} bytes"));
    }
    if text.is_empty() {
        return Err("empty text (set RVLLM_PPL_TEXT, RVLLM_PROMPT, or stdin)".into());
    }
    Ok(text)
}

fn ppl_windows(
    sequence_len: usize,
    window: usize,
    stride: usize,
    max_windows: usize,
) -> Result<Vec<(usize, usize, usize)>, String> {
    if sequence_len < 2 || window < 2 || window > sequence_len || stride == 0 || stride > window {
        return Err("invalid perplexity window geometry".into());
    }
    if sequence_len > window && stride == window {
        return Err("perplexity windows must retain at least one prediction-context token".into());
    }
    let mut plan = Vec::new();
    let mut begin = 0usize;
    let mut previous_end = 0usize;
    loop {
        let end = begin
            .checked_add(window)
            .ok_or("perplexity window index overflow")?
            .min(sequence_len);
        let score_from = if plan.is_empty() {
            0
        } else {
            previous_end
                .checked_sub(begin)
                .and_then(|offset| offset.checked_sub(1))
                .ok_or("perplexity windows do not retain prediction context")?
        };
        plan.push((begin, end, score_from));
        previous_end = end;
        if end == sequence_len || (max_windows > 0 && plan.len() >= max_windows) {
            break;
        }
        begin = begin
            .checked_add(stride)
            .ok_or("perplexity stride overflow")?;
    }
    Ok(plan)
}

fn bos_token_id(model_dir: &Path, tokenizer: &tokenizers::Tokenizer) -> Result<u32, String> {
    let path = model_dir.join("tokenizer_config.json");
    let bytes = read_bounded(&path, MAX_TOKENIZER_CONFIG_BYTES, "tokenizer metadata")?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let bos = match value.get("bos_token") {
        Some(serde_json::Value::String(token)) => Some(token.as_str()),
        Some(serde_json::Value::Object(token)) => token.get("content").and_then(|v| v.as_str()),
        _ => None,
    }
    .ok_or("tokenizer metadata does not define a BOS token")?;
    tokenizer
        .token_to_id(bos)
        .ok_or_else(|| format!("BOS token {bos:?} is absent from tokenizer.json"))
}

fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("{label} {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{label} {} exceeds {limit} bytes", path.display()));
    }
    Ok(bytes)
}

fn read_bounded_utf8(path: &Path, limit: u64, label: &str) -> Result<String, String> {
    String::from_utf8(read_bounded(path, limit, label)?)
        .map_err(|error| format!("{label} {} is not UTF-8: {error}", path.display()))
}

fn receipt_metadata() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "schema": "rvllm.local-benchmark.v1",
        "source_sha": required_hex("RVLLM_SOURCE_SHA", 40)?,
        "model": required_text("RVLLM_MODEL_ID")?,
        "model_sha256": required_hex("RVLLM_MODEL_SHA256", 64)?,
        "hardware": required_text("RVLLM_HARDWARE")?,
        "driver": required_text("RVLLM_DRIVER")?,
        "toolchain": required_text("RVLLM_TOOLCHAIN")?,
        "backend": "cuda",
        "command": command(),
        "unix_time_seconds": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_secs(),
    }))
}

fn artifact_digests(paths: &Gemma4EnginePaths) -> Result<serde_json::Value, String> {
    let manifest = paths
        .kernels_dir
        .join(required_text("RVLLM_KERNEL_ARCH")?)
        .join("manifest.json");
    Ok(serde_json::json!({
        "kernel_manifest_sha256": sha256_file(&manifest)?,
        "cutlass_sha256": sha256_file(&paths.cutlass_so)?,
        "flash_attention_sha256": sha256_file(&paths.fa3_so)?,
        "policy_sha256": sha256_file(&paths.policy_json)?,
    }))
}

fn finalize_receipt(
    mut result: serde_json::Value,
    metadata: serde_json::Value,
    artifacts: serde_json::Value,
) -> Result<String, String> {
    {
        let object = result
            .as_object_mut()
            .ok_or("benchmark result must be a JSON object")?;
        for (key, value) in metadata
            .as_object()
            .ok_or("receipt metadata must be a JSON object")?
        {
            object.insert(key.clone(), value.clone());
        }
        object.insert("artifacts".into(), artifacts);
    }
    let canonical =
        serde_json::to_vec(&result).map_err(|error| format!("receipt JSON: {error}"))?;
    result.as_object_mut().unwrap().insert(
        "receipt_sha256".into(),
        serde_json::Value::String(sha256_bytes(&canonical)),
    );
    serde_json::to_string(&result).map_err(|error| format!("receipt JSON: {error}"))
}

fn required_text(name: &str) -> Result<String, String> {
    let value = std::env::var(name).map_err(|_| format!("missing env var: {name}"))?;
    if value.is_empty()
        || value.len() > MAX_METADATA_TEXT
        || value.trim() != value
        || value.chars().any(|character| character.is_control())
    {
        return Err(format!(
            "{name} must be 1..={MAX_METADATA_TEXT} characters without controls or outer whitespace"
        ));
    }
    Ok(value)
}

fn required_hex(name: &str, length: usize) -> Result<String, String> {
    let value = required_text(name)?;
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{name} must be {length} lowercase hexadecimal characters"
        ));
    }
    Ok(value)
}

fn command() -> Vec<String> {
    let mut arguments = std::env::args_os();
    let _ = arguments.next();
    let binary = std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "rvllm-ppl".into());
    std::iter::once(binary)
        .chain(arguments.map(|argument| argument.to_string_lossy().into_owned()))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("hash {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_windows_score_every_target_once_without_reinserting_bos() {
        let plan = ppl_windows(10, 4, 2, 0).unwrap();
        assert_eq!(plan, vec![(0, 4, 0), (2, 6, 1), (4, 8, 1), (6, 10, 1)]);
        assert_eq!(
            plan.iter()
                .map(|(begin, end, score_from)| end - begin - 1 - score_from)
                .sum::<usize>(),
            9
        );
        assert!(plan.iter().skip(1).all(|(begin, _, _)| *begin > 0));
    }

    #[test]
    fn non_overlapping_windows_fail_closed() {
        assert!(ppl_windows(10, 4, 4, 0).is_err());
    }

    #[test]
    fn max_windows_is_explicitly_bounded() {
        assert_eq!(ppl_windows(10, 4, 2, 2).unwrap().len(), 2);
    }

    #[test]
    fn receipt_hash_covers_every_other_field() {
        let encoded = finalize_receipt(
            serde_json::json!({"result": 1}),
            serde_json::json!({"schema": "test"}),
            serde_json::json!({"artifact": "abc"}),
        )
        .unwrap();
        let mut receipt: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        let expected = receipt
            .as_object_mut()
            .unwrap()
            .remove("receipt_sha256")
            .unwrap();
        assert_eq!(
            expected,
            sha256_bytes(&serde_json::to_vec(&receipt).unwrap())
        );
    }
}

fn arena_bytes() -> Result<usize, String> {
    let gib = env_usize("RVLLM_ARENA_GB", 32)?;
    if !(1..=MAX_ARENA_GIB).contains(&gib) {
        return Err(format!("RVLLM_ARENA_GB must be in 1..={MAX_ARENA_GIB}"));
    }
    gib.checked_mul(1usize << 30)
        .ok_or("RVLLM_ARENA_GB byte count overflow".into())
}

fn env_path(name: &str) -> Result<PathBuf, String> {
    let path = std::env::var_os(name).ok_or_else(|| format!("missing env var: {name}"))?;
    if path.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(PathBuf::from(path))
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}")),
    }
}
