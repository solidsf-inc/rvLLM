use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const MAX_ERROR_BODY: usize = 4096;
const MAX_MODELS_BODY: usize = 1024 * 1024;
const MAX_COMPLETION_BODY: usize = 16 * 1024 * 1024;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Content(String),
    Done { elapsed_secs: f64 },
    Error(String),
}

pub fn fetch_models(endpoint: String, tx: mpsc::Sender<Vec<String>>) {
    tokio::spawn(async move {
        let result = async {
            let url = models_url(&endpoint)?;
            let response = authorized(client()?.get(url))?
                .send()
                .await?
                .error_for_status()?;
            let body: ModelListResponse =
                serde_json::from_slice(&read_bounded(response, MAX_MODELS_BODY).await?)?;
            Ok::<_, BoxError>(body.data.into_iter().map(|model| model.id).collect())
        }
        .await;
        let _ = tx.send(result.unwrap_or_default());
    });
}

pub fn request_chat(endpoint: String, request: ChatRequest, tx: mpsc::Sender<ChatEvent>) {
    tokio::spawn(async move {
        if let Err(error) = request_inner(endpoint, request, tx.clone()).await {
            let _ = tx.send(ChatEvent::Error(error.to_string()));
        }
    });
}

async fn request_inner(
    endpoint: String,
    request: ChatRequest,
    tx: mpsc::Sender<ChatEvent>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = endpoint_url(&endpoint)?;
    let started = Instant::now();
    let response = authorized(client()?.post(url))?
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = read_prefix(response, MAX_ERROR_BODY)
            .await
            .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
        return Err(format!("API returned {status}: {body}").into());
    }

    let body = read_bounded(response, MAX_COMPLETION_BODY).await?;
    let content = parse_chat_completion(&body)?;
    let _ = tx.send(ChatEvent::Content(content));
    let _ = tx.send(ChatEvent::Done {
        elapsed_secs: started.elapsed().as_secs_f64(),
    });
    Ok(())
}

fn parse_chat_completion(body: &[u8]) -> Result<String, BoxError> {
    let completion: ChatCompletionResponse = serde_json::from_slice(body)?;
    let choice = completion
        .choices
        .into_iter()
        .next()
        .ok_or("chat completion response has no choices")?;
    if choice.message.role != "assistant" {
        return Err("chat completion response role is not assistant".into());
    }
    Ok(choice.message.content)
}

fn client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .https_only(false)
        .build()
}

fn authorized(builder: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder, BoxError> {
    match std::env::var("RVLLM_API_KEY") {
        Ok(key) if key.is_empty() => Ok(builder),
        Ok(key)
            if key.trim() != key
                || key
                    .bytes()
                    .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control()) =>
        {
            Err("RVLLM_API_KEY contains whitespace or control characters".into())
        }
        Ok(key) => Ok(builder.bearer_auth(key)),
        Err(std::env::VarError::NotPresent) => Ok(builder),
        Err(error) => Err(format!("invalid RVLLM_API_KEY: {error}").into()),
    }
}

async fn read_bounded(response: reqwest::Response, limit: usize) -> Result<Vec<u8>, BoxError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(format!("response body exceeds {limit} bytes").into());
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(format!("response body exceeds {limit} bytes").into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn read_prefix(response: reqwest::Response, limit: usize) -> Result<String, BoxError> {
    let mut body = Vec::new();
    let mut truncated = false;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = limit.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
        if body.len() == limit {
            truncated = true;
            break;
        }
    }
    let mut text = String::from_utf8_lossy(&body).into_owned();
    if truncated {
        text.push('…');
    }
    Ok(text)
}

fn endpoint_url(endpoint: &str) -> Result<reqwest::Url, BoxError> {
    let url = reqwest::Url::parse(endpoint)?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err("endpoint must use http or https".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("endpoint must not contain credentials".into());
    }
    if url.fragment().is_some() {
        return Err("endpoint must not contain a fragment".into());
    }
    let host = url.host_str().ok_or("endpoint has no host")?;
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if url.scheme() == "http" && !loopback {
        return Err("plaintext HTTP is allowed only for loopback endpoints".into());
    }
    Ok(url)
}

fn models_url(
    chat_endpoint: &str,
) -> Result<reqwest::Url, Box<dyn std::error::Error + Send + Sync>> {
    let mut url = endpoint_url(chat_endpoint)?;
    url.set_path("/v1/models");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chat_completion_response() {
        let body = br#"{"choices":[{"message":{"role":"assistant","content":"ready"}}]}"#;
        assert_eq!(parse_chat_completion(body).unwrap(), "ready");
    }

    #[test]
    fn rejects_chat_completion_without_a_choice() {
        assert!(parse_chat_completion(br#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn derives_models_url() {
        assert_eq!(
            models_url("http://127.0.0.1:8080/v1/chat/completions?x=1")
                .unwrap()
                .as_str(),
            "http://127.0.0.1:8080/v1/models"
        );
    }

    #[test]
    fn rejects_unsafe_endpoints() {
        assert!(endpoint_url("http://example.com/v1/chat/completions").is_err());
        assert!(endpoint_url("https://user:secret@example.com/v1/chat/completions").is_err());
        assert!(endpoint_url("file:///tmp/socket").is_err());
        assert!(endpoint_url("http://[::1]:8080/v1/chat/completions").is_ok());
    }
}
