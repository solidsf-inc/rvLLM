use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rvllm_runtime::gemma4_bring_up::{SamplingParams, DEFAULT_TOP_K, MIN_TEMPERATURE};
use serde::Deserialize;
use serde_json::{json, Value};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static SEED_CTR: AtomicU64 = AtomicU64::new(0);

/// Marker bytes wrapped around image slot indices in the rendered prompt.
/// `\x1F` is ASCII Unit Separator; it cannot appear in user text or in any
/// Gemma chat-template token, so it is safe as an injection sentinel that
/// the worker substitutes for the real `<boi>{soft × N}<eoi>` string once
/// per-image `output_length` is known.
pub const IMAGE_SLOT_MARK: char = '\x1F';
const MAX_CHAT_MESSAGES: usize = 128;
const MAX_CONTENT_PARTS: usize = 256;
const MAX_IMAGES_PER_REQUEST: usize = 8;

/// Builds the slot marker text inserted into the rendered prompt for image
/// number `n` (0-indexed). The worker scans for these and replaces each
/// with the per-image `<boi>` + soft-token-times-N + `<eoi>` string.
pub fn image_slot_marker(n: usize) -> String {
    format!("{m}IMAGE_SLOT_{n}{m}", m = IMAGE_SLOT_MARK)
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    /// Sampling temperature. Default 0 = greedy argmax — the bit-exact
    /// production path this server has always run. Values are clamped to
    /// [0, 2] (the OpenAI range); values below 1e-3 behave as greedy.
    pub temperature: Option<f32>,
    /// Nucleus (top-p) cut over the temperature-scaled softmax. Clamped to
    /// [0, 1]; 1 (default) disables, 0 keeps only the top token. Ignored
    /// when temperature selects greedy.
    pub top_p: Option<f32>,
    /// Top-k cut. Omitted sampled requests default to 50; omitted greedy
    /// requests use 0. An explicit 0 requests the full vocabulary and will
    /// fail closed at execution if the vocabulary exceeds device capacity.
    pub top_k: Option<i64>,
    /// Sampling seed. Same seed + params + prompt -> the identical token
    /// stream. Absent -> a per-request entropy seed (non-deterministic,
    /// matching OpenAI). Ignored for greedy requests.
    pub seed: Option<i64>,
    pub stream: Option<bool>,
    pub stop: Option<StopSpec>,
    pub n: Option<usize>,
    pub ignore_eos: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub seed: Option<i64>,
    pub stream: Option<bool>,
    pub stop: Option<StopSpec>,
    pub n: Option<usize>,
    pub add_special_tokens: Option<bool>,
    pub ignore_eos: Option<bool>,
    pub return_token_ids: Option<bool>,
    pub prompt_logprobs: Option<usize>,
}

/// `/v1/completions` prompt forms. Superset of the plain-string OpenAI prompt:
/// also accepts pre-tokenized integer-id prompts (single or batched), required
/// for `prompt_logprobs` and useful for exact token control.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    Null(()),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub text: Option<String>,
    pub image_url: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct PreparedChat {
    /// Rendered chat-template prompt. Each image in a user message has been
    /// replaced by an `image_slot_marker(i)` sentinel; the worker substitutes
    /// the real `<boi>{soft × N}<eoi>` string at request-execution time.
    pub prompt: String,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    /// Clamped per-request sampling params (see the field docs on
    /// `ChatCompletionRequest` for the clamp semantics). Greedy when the
    /// request omitted `temperature` or sent 0.
    pub sampling: SamplingParams,
    /// Image sources in slot order. `images[i]` corresponds to the slot marker
    /// `image_slot_marker(i)` in `prompt`. The public server accepts bounded
    /// `data:` URLs; network and local-file loading are disabled.
    pub images: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PreparedCompletion {
    pub prompt: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub max_tokens: usize,
    pub add_bos: bool,
    pub ignore_eos: bool,
    pub return_token_ids: bool,
    pub prompt_logprobs: bool,
    /// Validated per-request sampling params. Greedy when the request omitted
    /// `temperature` or sent zero.
    pub sampling: SamplingParams,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub error_type: &'static str,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            error_type: "server_error",
        }
    }

    pub fn busy(message: impl Into<String>) -> Self {
        Self {
            status: 429,
            message: message.into(),
            error_type: "server_busy",
        }
    }
}

impl MessageContent {
    /// Render this message's content into the prompt string while collecting
    /// any `image_url` content parts into `images`. The rendered string has
    /// `image_slot_marker(i)` interleaved at the position each image
    /// appeared in the original message; the worker substitutes the real
    /// `<boi>{soft × N}<eoi>` once per-image `output_length` is known.
    fn render_into(
        &self,
        images: &mut Vec<String>,
        content_parts: &mut usize,
    ) -> Result<String, ApiError> {
        match self {
            MessageContent::Text(s) => Ok(s.clone()),
            MessageContent::Parts(parts) => {
                *content_parts = content_parts
                    .checked_add(parts.len())
                    .ok_or_else(|| ApiError::invalid("content part count overflow"))?;
                if *content_parts > MAX_CONTENT_PARTS {
                    return Err(ApiError::invalid(format!(
                        "at most {MAX_CONTENT_PARTS} content parts are supported"
                    )));
                }
                let mut out = String::new();
                for part in parts {
                    match part.kind.as_deref() {
                        Some("text") | None => {
                            if let Some(t) = part.text.as_deref() {
                                out.push_str(t);
                            }
                        }
                        Some("image_url") | Some("image") => {
                            if images.len() >= MAX_IMAGES_PER_REQUEST {
                                return Err(ApiError::invalid(format!(
                                    "at most {MAX_IMAGES_PER_REQUEST} images are supported"
                                )));
                            }
                            let url =
                                extract_image_url(part.image_url.as_ref()).ok_or_else(|| {
                                    ApiError::invalid(
                                        "image_url content part is missing a url string",
                                    )
                                })?;
                            if !is_data_url(&url) {
                                return Err(ApiError::invalid(
                                    "only data: image URLs are supported",
                                ));
                            }
                            let idx = images.len();
                            images.push(url);
                            out.push_str(&image_slot_marker(idx));
                        }
                        Some(other) => {
                            return Err(ApiError::invalid(format!(
                                "unsupported content part type: {other}"
                            )))
                        }
                    }
                }
                Ok(out)
            }
            MessageContent::Null(()) => Ok(String::new()),
        }
    }
}

/// Pull the URL string out of an OpenAI `image_url` value, which may be
/// either a bare string or an object of shape `{ "url": "...", "detail": ... }`.
fn extract_image_url(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(m) => m.get("url").and_then(|u| u.as_str()).map(str::to_string),
        _ => None,
    }
}

fn is_data_url(value: &str) -> bool {
    value
        .as_bytes()
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"data:"))
}

pub fn prepare_chat_request(
    req: ChatCompletionRequest,
    served_model: &str,
    default_max_tokens: usize,
    default_system_prompt: Option<&str>,
) -> Result<PreparedChat, ApiError> {
    if req.model != served_model {
        return Err(ApiError::not_found(format!(
            "model '{}' is not served by this rvllm-server; available model is '{}'",
            req.model, served_model
        )));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("only n=1 is supported"));
    }
    reject_unsupported_generation_options(req.stream, req.stop.as_ref())?;
    let sampling = sampling_from_parts(req.temperature, req.top_p, req.top_k, req.seed)?;

    let max_tokens = req.max_tokens.unwrap_or(default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::invalid("max_tokens must be > 0"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    if req.messages.len() > MAX_CHAT_MESSAGES {
        return Err(ApiError::invalid(format!(
            "at most {MAX_CHAT_MESSAGES} messages are supported"
        )));
    }

    let messages = apply_default_system_prompt(req.messages, default_system_prompt);

    let mut images = Vec::new();
    let prompt = render_gemma_chat(&messages, &mut images)?;
    Ok(PreparedChat {
        prompt,
        max_tokens,
        ignore_eos: req.ignore_eos.unwrap_or(false),
        sampling,
        images,
    })
}

pub fn prepare_completion_request(
    req: CompletionRequest,
    served_model: &str,
    default_max_tokens: usize,
) -> Result<PreparedCompletion, ApiError> {
    if req.model != served_model {
        return Err(ApiError::not_found(format!(
            "model '{}' is not served by this rvllm-server; available model is '{}'",
            req.model, served_model
        )));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("only n=1 is supported"));
    }
    reject_unsupported_generation_options(req.stream, req.stop.as_ref())?;
    let max_tokens = req.max_tokens.unwrap_or(default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::invalid("max_tokens must be > 0"));
    }

    let (prompt, prompt_token_ids) = match req.prompt {
        CompletionPrompt::Text(text) => {
            if text.is_empty() {
                return Err(ApiError::invalid("prompt must not be empty"));
            }
            (text, None)
        }
        CompletionPrompt::Texts(texts) => {
            let mut it = texts.into_iter();
            let Some(text) = it.next() else {
                return Err(ApiError::invalid("prompt must not be empty"));
            };
            if it.next().is_some() {
                return Err(ApiError::invalid("only a single prompt is supported"));
            }
            if text.is_empty() {
                return Err(ApiError::invalid("prompt must not be empty"));
            }
            (text, None)
        }
        CompletionPrompt::Tokens(tokens) => {
            if tokens.is_empty() {
                return Err(ApiError::invalid("prompt token IDs must not be empty"));
            }
            (String::new(), Some(tokens))
        }
        CompletionPrompt::TokenBatches(batches) => {
            let mut it = batches.into_iter();
            let Some(tokens) = it.next() else {
                return Err(ApiError::invalid("prompt token IDs must not be empty"));
            };
            if it.next().is_some() {
                return Err(ApiError::invalid("only a single prompt is supported"));
            }
            if tokens.is_empty() {
                return Err(ApiError::invalid("prompt token IDs must not be empty"));
            }
            (String::new(), Some(tokens))
        }
    };

    let prompt_logprobs = req.prompt_logprobs.unwrap_or(0) > 0;
    if prompt_logprobs && prompt_token_ids.is_none() {
        return Err(ApiError::invalid(
            "prompt_logprobs requires an integer-token prompt",
        ));
    }

    let sampling = sampling_from_parts(req.temperature, req.top_p, req.top_k, req.seed)?;

    Ok(PreparedCompletion {
        prompt,
        prompt_token_ids,
        max_tokens,
        add_bos: req.add_special_tokens.unwrap_or(true),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        return_token_ids: req.return_token_ids.unwrap_or(false),
        prompt_logprobs,
        sampling,
    })
}

fn reject_unsupported_generation_options(
    stream: Option<bool>,
    stop: Option<&StopSpec>,
) -> Result<(), ApiError> {
    if stream.unwrap_or(false) {
        return Err(ApiError::invalid(
            "stream=true is not supported; use stream=false",
        ));
    }
    if stop.is_some() {
        return Err(ApiError::invalid("stop is not supported; omit stop"));
    }
    Ok(())
}

/// Validate request sampling fields before they reach CPU or GPU samplers.
fn sampling_from_parts(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    seed: Option<i64>,
) -> Result<SamplingParams, ApiError> {
    let temperature = temperature.unwrap_or(0.0);
    let top_p = top_p.unwrap_or(1.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid(
            "temperature must be finite and in [0, 2]",
        ));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(ApiError::invalid("top_p must be finite and in [0, 1]"));
    }
    let top_k = top_k.unwrap_or_else(|| {
        if temperature < MIN_TEMPERATURE {
            0
        } else {
            i64::from(DEFAULT_TOP_K)
        }
    });
    if !(0..=1024).contains(&top_k) {
        return Err(ApiError::invalid("top_k must be in [0, 1024]"));
    }
    Ok(SamplingParams {
        temperature,
        top_p,
        top_k: top_k as u32,
        seed: seed.map(|s| s as u64).unwrap_or_else(entropy_seed),
    })
}

/// Per-request seed when the caller did not pin one: wall-clock nanos plus
/// a golden-ratio counter step so two requests in the same nanosecond still
/// diverge. Non-deterministic by design — pass `seed` to reproduce.
fn entropy_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos.wrapping_add(SEED_CTR.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed))
}

fn apply_default_system_prompt(
    mut messages: Vec<ChatMessage>,
    default_system_prompt: Option<&str>,
) -> Vec<ChatMessage> {
    let Some(prompt) = default_system_prompt
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return messages;
    };
    messages.insert(
        0,
        ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(prompt.into()),
        },
    );
    messages
}

pub fn render_gemma_chat(
    messages: &[ChatMessage],
    images: &mut Vec<String>,
) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }

    let mut out = String::new();
    let mut system = String::new();
    let mut saw_turn = false;
    let mut content_parts = 0usize;

    for msg in messages {
        let role = msg.role.as_str();
        let text = msg.content.render_into(images, &mut content_parts)?;
        match role {
            "system" | "developer" => {
                append_system(&mut system, &text);
            }
            "user" => {
                let merged = if system.is_empty() {
                    text
                } else {
                    let mut s = String::new();
                    s.push_str(system.trim_end());
                    s.push_str("\n\n");
                    s.push_str(text.trim_end());
                    system.clear();
                    s
                };
                push_turn(&mut out, "user", &merged);
                saw_turn = true;
            }
            "assistant" => {
                push_turn(&mut out, "model", &text);
                saw_turn = true;
            }
            other => {
                return Err(ApiError::invalid(format!(
                    "unsupported message role for Gemma chat template: {other}"
                )));
            }
        }
    }

    if !system.is_empty() {
        push_turn(&mut out, "user", &system);
        saw_turn = true;
    }
    if !saw_turn {
        return Err(ApiError::invalid(
            "messages must include a user or assistant turn",
        ));
    }

    out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    Ok(out)
}

pub fn created_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn completion_id(created: u64) -> String {
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-rvllm-{created}-{n}")
}

pub fn models_response(model: &str) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "solidSF"
        }]
    })
}

#[allow(clippy::too_many_arguments)]
pub fn completion_response(
    id: &str,
    model: &str,
    created: u64,
    content: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn text_completion_response(
    id: &str,
    model: &str,
    created: u64,
    content: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
    generated_token_ids: Option<&[u32]>,
    prompt_token_ids: Option<&[u32]>,
    prompt_logprobs: Option<&[Option<f64>]>,
) -> Value {
    let mut choice = json!({
        "index": 0,
        "text": content,
        "finish_reason": finish_reason
    });
    if let Some(token_ids) = generated_token_ids {
        choice["token_ids"] = json!(token_ids);
    }
    if let (Some(token_ids), Some(logprobs)) = (prompt_token_ids, prompt_logprobs) {
        choice["prompt_logprobs"] = prompt_logprobs_value(token_ids, logprobs);
    }

    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

fn prompt_logprobs_value(token_ids: &[u32], logprobs: &[Option<f64>]) -> Value {
    let values: Vec<Value> = token_ids
        .iter()
        .enumerate()
        .map(|(i, token_id)| match logprobs.get(i).copied().flatten() {
            Some(logprob) => json!({ token_id.to_string(): logprob }),
            None => Value::Null,
        })
        .collect();
    Value::Array(values)
}

pub fn error_response(err: &ApiError) -> Value {
    json!({
        "error": {
            "message": err.message,
            "type": err.error_type,
            "param": null,
            "code": null
        }
    })
}

fn append_system(dst: &mut String, text: &str) {
    if !dst.is_empty() {
        dst.push_str("\n\n");
    }
    dst.push_str(text.trim_end());
}

fn push_turn(out: &mut String, role: &str, text: &str) {
    out.push_str("<|turn>");
    out.push_str(role);
    out.push('\n');
    out.push_str(text.trim_end());
    out.push_str("<turn|>\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: MessageContent::Text(content.into()),
        }
    }

    #[test]
    fn renders_gemma_turns() {
        let mut images = Vec::new();
        let prompt = render_gemma_chat(&[msg("user", "hello")], &mut images).unwrap();
        assert_eq!(
            prompt,
            "<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        assert!(images.is_empty());
    }

    #[test]
    fn folds_system_into_first_user() {
        let mut images = Vec::new();
        let prompt =
            render_gemma_chat(&[msg("system", "be brief"), msg("user", "hi")], &mut images)
                .unwrap();
        assert!(prompt.contains("be brief\n\nhi"));
        assert!(images.is_empty());
    }

    #[test]
    fn collects_image_url_into_slot_marker() {
        use serde_json::json;
        let user = ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    kind: Some("text".into()),
                    text: Some("describe ".into()),
                    image_url: None,
                },
                ContentPart {
                    kind: Some("image_url".into()),
                    text: None,
                    image_url: Some(json!({ "url": "data:image/png;base64,AA==" })),
                },
            ]),
        };
        let mut images = Vec::new();
        let prompt = render_gemma_chat(&[user], &mut images).unwrap();
        assert_eq!(images, vec!["data:image/png;base64,AA==".to_string()]);
        let want = format!("describe {}", image_slot_marker(0));
        assert!(
            prompt.contains(&want),
            "prompt missing slot marker: {prompt}"
        );
    }

    #[test]
    fn caps_content_parts_and_images_before_worker_execution() {
        let image_part = || ContentPart {
            kind: Some("image_url".into()),
            text: None,
            image_url: Some(json!("data:image/png;base64,AA==")),
        };
        let user = ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(
                (0..=MAX_IMAGES_PER_REQUEST).map(|_| image_part()).collect(),
            ),
        };
        let err = render_gemma_chat(&[user], &mut Vec::new()).expect_err("must cap images");
        assert!(err.message.contains("at most 8 images"));

        let user = ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(
                (0..=MAX_CONTENT_PARTS)
                    .map(|_| ContentPart {
                        kind: Some("text".into()),
                        text: Some("x".into()),
                        image_url: None,
                    })
                    .collect(),
            ),
        };
        let err = render_gemma_chat(&[user], &mut Vec::new()).expect_err("must cap parts");
        assert!(err.message.contains("at most 256 content parts"));
    }

    #[test]
    fn rejects_image_url_without_url() {
        use serde_json::json;
        let user = ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![ContentPart {
                kind: Some("image_url".into()),
                text: None,
                image_url: Some(json!({ "detail": "high" })),
            }]),
        };
        let mut images = Vec::new();
        let err = render_gemma_chat(&[user], &mut images).expect_err("missing url must 400");
        assert_eq!(err.status, 400);
    }

    #[test]
    fn rejects_network_and_local_image_sources() {
        use serde_json::json;
        for source in [
            "https://example.com/a.png",
            "file:///tmp/a.png",
            "/tmp/a.png",
        ] {
            let user = ChatMessage {
                role: "user".into(),
                content: MessageContent::Parts(vec![ContentPart {
                    kind: Some("image_url".into()),
                    text: None,
                    image_url: Some(json!({ "url": source })),
                }]),
            };
            let mut images = Vec::new();
            let err = render_gemma_chat(&[user], &mut images).expect_err("source must fail");
            assert_eq!(err.status, 400);
            assert!(images.is_empty());
        }
    }

    fn req(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "served".into(),
            messages,
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            stop: None,
            n: None,
            ignore_eos: None,
        }
    }

    fn completion_req() -> CompletionRequest {
        CompletionRequest {
            model: "served".into(),
            prompt: CompletionPrompt::Text("hello".into()),
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            stop: None,
            n: None,
            add_special_tokens: None,
            ignore_eos: None,
            return_token_ids: None,
            prompt_logprobs: None,
        }
    }

    #[test]
    fn rejects_streaming_for_both_generation_routes() {
        let mut chat = req(vec![msg("user", "hi")]);
        chat.stream = Some(true);
        let err = prepare_chat_request(chat, "served", 16, None).expect_err("must reject");
        assert_eq!(err.status, 400);
        assert_eq!(
            err.message,
            "stream=true is not supported; use stream=false"
        );

        let mut completion = completion_req();
        completion.stream = Some(true);
        let err = prepare_completion_request(completion, "served", 16).expect_err("must reject");
        assert_eq!(err.status, 400);
        assert_eq!(
            err.message,
            "stream=true is not supported; use stream=false"
        );
    }

    #[test]
    fn rejects_stop_for_both_generation_routes() {
        let mut chat = req(vec![msg("user", "hi")]);
        chat.stop = Some(StopSpec::One("END".into()));
        let err = prepare_chat_request(chat, "served", 16, None).expect_err("must reject");
        assert_eq!(err.status, 400);
        assert_eq!(err.message, "stop is not supported; omit stop");

        let mut completion = completion_req();
        completion.stop = Some(StopSpec::Many(vec!["END".into()]));
        let err = prepare_completion_request(completion, "served", 16).expect_err("must reject");
        assert_eq!(err.status, 400);
        assert_eq!(err.message, "stop is not supported; omit stop");
    }

    #[test]
    fn injects_default_system_prompt() {
        let r = req(vec![msg("system", "request system"), msg("user", "hi")]);
        let prepared = prepare_chat_request(r, "served", 16, Some("server system")).unwrap();
        assert!(prepared
            .prompt
            .contains("server system\n\nrequest system\n\nhi"));
    }

    #[test]
    fn prepares_completion_token_prompt() {
        let req = CompletionRequest {
            model: "served".into(),
            prompt: CompletionPrompt::Tokens(vec![1, 2, 3]),
            max_tokens: Some(4),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            stop: None,
            n: None,
            add_special_tokens: Some(false),
            ignore_eos: Some(true),
            return_token_ids: Some(true),
            prompt_logprobs: None,
        };
        let prepared = prepare_completion_request(req, "served", 16).unwrap();
        assert_eq!(prepared.prompt_token_ids, Some(vec![1, 2, 3]));
        assert_eq!(prepared.max_tokens, 4);
        assert!(!prepared.add_bos);
        assert!(prepared.ignore_eos);
        assert!(prepared.return_token_ids);
    }

    #[test]
    fn rejects_empty_text_prompt_without_special_tokens() {
        let mut req = completion_req();
        req.prompt = CompletionPrompt::Text(String::new());
        req.add_special_tokens = Some(false);
        let err = prepare_completion_request(req, "served", 16).expect_err("must reject");
        assert_eq!(err.status, 400);
        assert_eq!(err.message, "prompt must not be empty");
    }

    #[test]
    fn prompt_logprobs_require_integer_tokens() {
        let req = CompletionRequest {
            model: "served".into(),
            prompt: CompletionPrompt::Text("hello".into()),
            max_tokens: Some(1),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            stop: None,
            n: None,
            add_special_tokens: None,
            ignore_eos: None,
            return_token_ids: None,
            prompt_logprobs: Some(1),
        };
        let err = prepare_completion_request(req, "served", 16).expect_err("text must reject");
        assert_eq!(err.status, 400);

        let req = CompletionRequest {
            model: "served".into(),
            prompt: CompletionPrompt::Tokens(vec![1, 2, 3]),
            max_tokens: Some(1),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            stop: None,
            n: None,
            add_special_tokens: None,
            ignore_eos: None,
            return_token_ids: None,
            prompt_logprobs: Some(1),
        };
        let prepared = prepare_completion_request(req, "served", 16).expect("tokens must parse");
        assert!(prepared.prompt_logprobs);
    }

    #[test]
    fn model_metadata_is_owned_by_solidsf() {
        assert_eq!(models_response("served")["data"][0]["owned_by"], "solidSF");
    }

    #[test]
    fn greedy_by_default_and_nonzero_temperature_accepted() {
        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = None;
        let prepared = prepare_chat_request(r, "served", 16, None).unwrap();
        assert!(prepared.sampling.is_greedy());
        assert_eq!(prepared.sampling.top_k, 0);

        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = Some(0.7);
        r.seed = Some(42);
        let prepared = prepare_chat_request(r, "served", 16, None).unwrap();
        assert!(!prepared.sampling.is_greedy());
        assert_eq!(prepared.sampling.temperature, 0.7);
        assert_eq!(prepared.sampling.top_k, DEFAULT_TOP_K);
        assert_eq!(prepared.sampling.seed, 42);
    }

    #[test]
    fn explicit_zero_top_k_stays_full_vocab_and_fails_closed_when_too_large() {
        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = Some(0.7);
        r.top_k = Some(0);
        let sampling = prepare_chat_request(r, "served", 16, None)
            .unwrap()
            .sampling;
        assert_eq!(sampling.top_k, 0);
        assert!(sampling.kernel_k(262_144).is_err());
    }

    #[test]
    fn rejects_invalid_sampling_fields() {
        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = Some(9.5);
        r.top_p = Some(-0.5);
        r.top_k = Some(-3);
        assert!(prepare_chat_request(r, "served", 16, None).is_err());

        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = Some(1.0);
        r.top_k = Some(1_000_000);
        assert!(prepare_chat_request(r, "served", 16, None).is_err());
    }

    #[test]
    fn missing_seed_gets_entropy() {
        let mut a = req(vec![msg("user", "hi")]);
        a.temperature = Some(1.0);
        let mut b = req(vec![msg("user", "hi")]);
        b.temperature = Some(1.0);
        let sa = prepare_chat_request(a, "served", 16, None)
            .unwrap()
            .sampling;
        let sb = prepare_chat_request(b, "served", 16, None)
            .unwrap()
            .sampling;
        assert_ne!(sa.seed, sb.seed, "entropy seeds must differ per request");
    }

    #[test]
    fn finish_reason_lands_in_responses() {
        let v = completion_response("id", "m", 0, "txt", 3, 5, "length");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["total_tokens"], 8);
    }
}
