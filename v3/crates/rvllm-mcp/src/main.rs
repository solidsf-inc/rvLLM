//! Stdio Model Context Protocol bridge for an rvLLM chat endpoint.

use std::io::{self, BufRead, Read, Write};
use std::sync::Mutex;
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH};
use reqwest::{Method, Url};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "rvllm-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_INPUT_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 1024 * 1024;
const MAX_SYSTEM_BYTES: usize = 256 * 1024;
const MAX_MODEL_BYTES: usize = 1024;
const MAX_TOKENS: u32 = 1_000_000;

#[derive(Clone, Debug)]
struct Config {
    base_url: Url,
    model: Option<String>,
    api_key: Option<String>,
    default_max_tokens: u32,
    request_timeout: Duration,
    client: Client,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let raw_url =
            std::env::var("RVLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
        let base_url = validate_base_url(&raw_url)?;
        let model = bounded_optional_env("RVLLM_MODEL", MAX_MODEL_BYTES)?;
        let api_key = bounded_optional_env("RVLLM_API_KEY", 4096)?;
        if api_key
            .as_deref()
            .is_some_and(|key| key.chars().any(char::is_control))
        {
            return Err("RVLLM_API_KEY contains control characters".into());
        }
        let default_max_tokens =
            parse_bounded_env("RVLLM_DEFAULT_MAX_TOKENS", 512u32, 1u32, MAX_TOKENS)?;
        let timeout_secs = parse_bounded_env("RVLLM_REQUEST_TIMEOUT_SECS", 120u64, 1u64, 3600u64)?;
        let request_timeout = Duration::from_secs(timeout_secs);
        let client = Client::builder()
            .timeout(request_timeout)
            .connect_timeout(request_timeout.min(Duration::from_secs(30)))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| "failed to initialize HTTP client".to_string())?;
        Ok(Self {
            base_url,
            model,
            api_key,
            default_max_tokens,
            request_timeout,
            client,
        })
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = match Config::from_env() {
        Ok(config) => config,
        Err(message) => {
            error!("configuration rejected: {message}");
            std::process::exit(2);
        }
    };
    info!("rvllm-mcp starting");

    let stdout = io::stdout();
    let stdout_lock = Mutex::new(stdout.lock());
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    loop {
        let line = match read_bounded_line(&mut reader, MAX_INPUT_LINE_BYTES) {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(error) if error.kind() == io::ErrorKind::InvalidData => {
                warn!("oversized MCP input rejected");
                continue;
            }
            Err(error) => {
                error!("stdin read error: {error}");
                break;
            }
        };
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let msg: Value = match serde_json::from_slice(&line) {
            Ok(value) => value,
            Err(_) => {
                warn!("invalid JSON-RPC input rejected");
                continue;
            }
        };
        let Some(response) = handle_message(&config, &msg) else {
            continue;
        };
        let mut encoded = match serde_json::to_vec(&response) {
            Ok(encoded) if encoded.len() <= MAX_RESPONSE_BYTES + MAX_INPUT_LINE_BYTES => encoded,
            Ok(_) => {
                error!("MCP response exceeded output limit");
                continue;
            }
            Err(error) => {
                error!("serialize response: {error}");
                continue;
            }
        };
        encoded.push(b'\n');
        let mut guard = stdout_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.write_all(&encoded).is_err() || guard.flush().is_err() {
            error!("stdout write failed");
            break;
        }
    }
}

fn handle_message(config: &Config, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "initialize" => Some(rpc_ok(
            id?,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
                "capabilities": { "tools": { "listChanged": false } },
                "instructions": "Use `complete` to run a chat completion on the configured rvLLM endpoint."
            }),
        )),
        "notifications/initialized" | "notifications/cancelled" => None,
        "tools/list" => Some(rpc_ok(
            id?,
            json!({ "tools": [tool_descriptor_complete()] }),
        )),
        "tools/call" => {
            let id = id?;
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Value::Null);
            if name != "complete" {
                return Some(rpc_err(id, -32601, "unknown tool"));
            }
            Some(match call_complete(config, &args) {
                Ok(text) => rpc_ok(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }),
                ),
                Err(message) => rpc_ok(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": format!("rvLLM request failed: {message}") }],
                        "isError": true
                    }),
                ),
            })
        }
        "ping" => Some(rpc_ok(id?, json!({}))),
        "" => None,
        _ => id.map(|id| rpc_err(id, -32601, "method not found")),
    }
}

fn tool_descriptor_complete() -> Value {
    json!({
        "name": "complete",
        "description": "Run a chat completion against the configured rvLLM endpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": { "type": "string" },
                "system": { "type": "string" },
                "max_tokens": { "type": "integer", "minimum": 1, "maximum": MAX_TOKENS },
                "temperature": { "type": "number", "minimum": 0.0, "maximum": 2.0 },
                "model": { "type": "string", "maxLength": MAX_MODEL_BYTES }
            },
            "required": ["prompt"],
            "additionalProperties": false
        }
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompleteArgs {
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    model: Option<String>,
}

fn call_complete(config: &Config, args: &Value) -> Result<String, String> {
    let parsed: CompleteArgs =
        serde_json::from_value(args.clone()).map_err(|_| "invalid arguments".to_string())?;
    validate_text("prompt", &parsed.prompt, 1, MAX_PROMPT_BYTES)?;
    if let Some(system) = &parsed.system {
        validate_text("system", system, 0, MAX_SYSTEM_BYTES)?;
    }
    if let Some(model) = &parsed.model {
        validate_identifier("model", model, MAX_MODEL_BYTES)?;
    }
    let max_tokens = parsed.max_tokens.unwrap_or(config.default_max_tokens);
    if !(1..=MAX_TOKENS).contains(&max_tokens) {
        return Err("max_tokens is out of range".into());
    }
    let temperature = parsed.temperature.unwrap_or(0.7);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err("temperature is out of range".into());
    }
    let model = match parsed.model.or_else(|| config.model.clone()) {
        Some(model) => model,
        None => probe_default_model(config)?,
    };
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = parsed.system.filter(|value| !value.is_empty()) {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": parsed.prompt}));
    let body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false
    });
    let response = request_json(config, Method::POST, "v1/chat/completions", Some(&body))?;
    response
        .get("choices")
        .and_then(|value| value.get(0))
        .and_then(|value| value.get("message"))
        .and_then(|value| value.get("content"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "endpoint returned an unexpected response shape".into())
}

fn probe_default_model(config: &Config) -> Result<String, String> {
    let response = request_json(config, Method::GET, "v1/models", None)?;
    let model = response
        .get("data")
        .and_then(Value::as_array)
        .and_then(|models| models.first())
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| "endpoint returned no model".to_string())?;
    validate_identifier("model id", model, MAX_MODEL_BYTES)?;
    Ok(model.to_owned())
}

fn request_json(
    config: &Config,
    method: Method,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let url = config
        .base_url
        .join(path)
        .map_err(|_| "invalid endpoint path".to_string())?;
    let mut request = config
        .client
        .request(method, url)
        .timeout(config.request_timeout);
    if let Some(api_key) = &config.api_key {
        request = request.header(AUTHORIZATION, format!("Bearer {api_key}"));
    }
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request
        .send()
        .map_err(|_| "endpoint request failed".to_string())?;
    decode_response(response)
}

fn decode_response(mut response: Response) -> Result<Value, String> {
    let status = response.status();
    if !status.is_success() {
        return Err(format!("endpoint returned HTTP {}", status.as_u16()));
    }
    if response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err("endpoint response exceeded size limit".into());
    }
    let mut bytes = Vec::new();
    response
        .by_ref()
        .take(MAX_RESPONSE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "failed to read endpoint response".to_string())?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err("endpoint response exceeded size limit".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "endpoint returned invalid JSON".into())
}

fn validate_base_url(raw: &str) -> Result<Url, String> {
    let mut url = Url::parse(raw).map_err(|_| "RVLLM_BASE_URL is invalid".to_string())?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("RVLLM_BASE_URL must not contain credentials".into());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("RVLLM_BASE_URL must not contain a query or fragment".into());
    }
    if url.path() != "/" && !url.path().is_empty() {
        return Err("RVLLM_BASE_URL must not contain a path".into());
    }
    if url.host().is_none() {
        return Err("RVLLM_BASE_URL is missing a host".into());
    }
    match url.scheme() {
        "https" => {}
        "http" if is_loopback_url(&url) => {}
        "http" => return Err("plain HTTP is restricted to a literal loopback host".into()),
        _ => return Err("RVLLM_BASE_URL must use http or https".into()),
    }
    url.set_path("/");
    Ok(url)
}

fn is_loopback_url(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn validate_text(name: &str, value: &str, min: usize, max: usize) -> Result<(), String> {
    if value.len() < min || value.len() > max {
        return Err(format!("{name} length is out of range"));
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str, max: usize) -> Result<(), String> {
    validate_text(name, value, 1, max)?;
    if value.chars().any(char::is_control) {
        return Err(format!("{name} contains control characters"));
    }
    Ok(())
}

fn bounded_optional_env(name: &str, max: usize) -> Result<Option<String>, String> {
    let Ok(value) = std::env::var(name) else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(None);
    }
    validate_identifier(name, &value, max)?;
    Ok(Some(value))
}

fn parse_bounded_env<T>(name: &str, default: T, min: T, max: T) -> Result<T, String>
where
    T: Copy + Ord + std::str::FromStr,
{
    let Ok(raw) = std::env::var(name) else {
        return Ok(default);
    };
    let value = raw.parse::<T>().map_err(|_| format!("{name} is invalid"))?;
    if value < min || value > max {
        return Err(format!("{name} is out of range"));
    }
    Ok(value)
}

fn read_bounded_line<R: BufRead>(reader: &mut R, max: usize) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let mut oversized = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if line.is_empty() && !oversized {
                return Ok(None);
            }
            break;
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if !oversized {
            if line.len().saturating_add(take) > max {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(&available[..take]);
            }
        }
        let ended = available[take - 1] == b'\n';
        reader.consume(take);
        if ended {
            break;
        }
    }
    if oversized {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "line too long"));
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(Some(line))
}

fn rpc_ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_err(id: Value, code: i32, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_requires_loopback() {
        assert!(validate_base_url("http://127.0.0.1:8000").is_ok());
        assert!(validate_base_url("http://[::1]:8000").is_ok());
        assert!(validate_base_url("http://example.com").is_err());
        assert!(validate_base_url("https://example.com").is_ok());
    }

    #[test]
    fn url_rejects_credentials_and_paths() {
        assert!(validate_base_url("https://user:secret@example.com").is_err());
        assert!(validate_base_url("https://example.com/api").is_err());
    }

    #[test]
    fn bounded_line_drains_oversized_input() {
        let input = b"123456\nok\n";
        let mut reader = io::BufReader::new(&input[..]);
        assert_eq!(
            read_bounded_line(&mut reader, 4).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            read_bounded_line(&mut reader, 4).unwrap(),
            Some(b"ok".to_vec())
        );
    }

    #[test]
    fn tools_list_exposes_complete() {
        let config = test_config();
        let request = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"});
        let response = handle_message(&config, &request).unwrap();
        assert_eq!(response["result"]["tools"][0]["name"], "complete");
    }

    fn test_config() -> Config {
        Config {
            base_url: Url::parse("http://127.0.0.1:8000/").unwrap(),
            model: Some("test".into()),
            api_key: None,
            default_max_tokens: 64,
            request_timeout: Duration::from_secs(1),
            client: Client::builder().build().unwrap(),
        }
    }
}
