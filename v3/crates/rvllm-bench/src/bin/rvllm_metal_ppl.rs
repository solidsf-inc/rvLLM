#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rvllm_loader::gemma4_arch::Gemma4Arch;
use rvllm_loader::metal_host::Gemma4HostDecoder;
use rvllm_loader::metal_loader::MetalWeightCache;
use sha2::{Digest, Sha256};

const MAX_TEXT_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_TEXT: usize = 256;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(e) = run() {
        eprintln!("rvllm-metal-ppl: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let text = read_text()?;
    let input_sha256 = sha256_bytes(text.as_bytes());
    let metadata = receipt_metadata()?;
    let artifacts = artifact_digests(&model_dir)?;

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer_bytes = read_bounded(&tok_path, MAX_TOKENIZER_BYTES, "tokenizer")?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| format!("tokenizer load {}: {e}", tok_path.display()))?;
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let bos_id = bos_token_id(&model_dir, &tokenizer)?;
    let token_ids: Vec<u32> = std::iter::once(bos_id)
        .chain(encoding.get_ids().iter().copied())
        .collect();
    if token_ids.len() < 2 {
        return Err(format!("not enough tokens ({}) to score", token_ids.len()));
    }

    let arch = Gemma4Arch::from_dir(&model_dir)
        .map_err(|e| format!("gemma4 arch parse {}: {e}", model_dir.display()))?;
    if token_ids
        .iter()
        .any(|token| *token as usize >= arch.vocab_size)
    {
        return Err("tokenizer emitted an ID outside the model vocabulary".into());
    }
    let max_model_len = env_usize("RVLLM_PPL_MAX_MODEL_LEN", arch.max_position_embeddings)?;
    if max_model_len == 0 || max_model_len > arch.max_position_embeddings {
        return Err(format!(
            "RVLLM_PPL_MAX_MODEL_LEN must be in 1..={}",
            arch.max_position_embeddings
        ));
    }
    if token_ids.len() > max_model_len {
        return Err(format!(
            "token count {} exceeds RVLLM_PPL_MAX_MODEL_LEN={max_model_len}",
            token_ids.len()
        ));
    }

    let t_load = Instant::now();
    let device = Arc::new(
        rvllm_metal::MetalDevice::system_default().map_err(|e| format!("metal device: {e}"))?,
    );
    let kernels = Arc::new(
        rvllm_metal::MetalKernels::new(&device).map_err(|e| format!("metal kernels: {e}"))?,
    );
    let weights =
        MetalWeightCache::from_dir_env(&model_dir, Arc::clone(&device), arch.num_hidden_layers)
            .map_err(|e| format!("metal weight cache: {e}"))?;
    let mut decoder =
        Gemma4HostDecoder::new_with_metal(arch, max_model_len, Arc::clone(&device), kernels)
            .map_err(|e| format!("metal decoder: {e}"))?;
    eprintln!("load_s={:.3}", t_load.elapsed().as_secs_f64());

    let t_eval = Instant::now();
    let score = decoder
        .score_nll(&weights, &token_ids, 0)
        .map_err(|e| format!("score_nll: {e}"))?;
    if score.tokens == 0 || !score.perplexity.is_finite() || !score.total_nll.is_finite() {
        return Err("score_nll returned an invalid result".into());
    }
    let elapsed = t_eval.elapsed().as_secs_f64();
    let result = serde_json::json!({
        "mode": "perplexity",
        "perplexity": score.perplexity,
        "total_nll": score.total_nll,
        "tokens": score.tokens,
        "input_tokens": token_ids.len(),
        "elapsed_s": elapsed,
        "tokens_per_second": if elapsed > 0.0 { score.tokens as f64 / elapsed } else { 0.0 },
        "input_sha256": input_sha256,
        "max_model_len": max_model_len,
    });
    println!("{}", finalize_receipt(result, metadata, artifacts)?);
    Ok(())
}

fn env_path(name: &str) -> Result<PathBuf, String> {
    let value = std::env::var_os(name).ok_or_else(|| format!("missing env var: {name}"))?;
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(PathBuf::from(value))
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
        "backend": "metal",
        "command": command(),
        "unix_time_seconds": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_secs(),
    }))
}

fn artifact_digests(model_dir: &Path) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "config_sha256": sha256_file(&model_dir.join("config.json"))?,
        "tokenizer_sha256": sha256_file(&model_dir.join("tokenizer.json"))?,
        "tokenizer_config_sha256": sha256_file(&model_dir.join("tokenizer_config.json"))?,
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
        .unwrap_or_else(|| "rvllm-metal-ppl".into());
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
