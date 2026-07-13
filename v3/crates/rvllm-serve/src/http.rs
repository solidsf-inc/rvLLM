use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::Value;

use crate::openai::{
    completion_id, completion_response, created_unix, error_response, models_response,
    prepare_chat_request, prepare_completion_request, text_completion_response, ApiError,
    ChatCompletionRequest, CompletionRequest,
};
use crate::worker::{GenerateError, GenerateRequest, WorkerHandle};
use crate::ServeConfig;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_HEADERS: usize = 100;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct State {
    config: ServeConfig,
    worker: WorkerHandle,
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

enum ReadRequestError {
    BadRequest(String),
    Unauthorized,
}

struct ConnectionPermit {
    active: Arc<AtomicUsize>,
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn serve(config: ServeConfig, worker: WorkerHandle) -> Result<(), String> {
    let addr = config.addr();
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    tracing::info!("rvllm-server listening on http://{addr}");

    let state = Arc::new(State { config, worker });
    let active_connections = Arc::new(AtomicUsize::new(0));

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let Some(permit) = try_acquire_connection(
                    Arc::clone(&active_connections),
                    state.config.max_inflight_requests,
                ) else {
                    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
                    let _ = write_json(
                        &mut stream,
                        429,
                        &error_response(&ApiError::busy("server is busy")),
                    );
                    continue;
                };
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    let _permit = permit;
                    if let Err(e) = handle_connection(stream, state) {
                        tracing::debug!("connection error: {e}");
                    }
                });
            }
            Err(e) => tracing::warn!("accept failed: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: Arc<State>) -> Result<(), String> {
    stream
        .set_nodelay(true)
        .map_err(|e| format!("set TCP_NODELAY: {e}"))?;
    stream
        .set_read_timeout(Some(SOCKET_TIMEOUT))
        .map_err(|e| format!("set read timeout: {e}"))?;
    stream
        .set_write_timeout(Some(SOCKET_TIMEOUT))
        .map_err(|e| format!("set write timeout: {e}"))?;
    let req = match read_request(&mut stream, state.config.api_key.as_deref()) {
        Ok(req) => req,
        Err(ReadRequestError::Unauthorized) => return write_unauthorized(&mut stream),
        Err(ReadRequestError::BadRequest(message)) => {
            tracing::debug!(error = %message, "invalid HTTP request");
            return write_json(
                &mut stream,
                400,
                &error_response(&ApiError::invalid("invalid HTTP request")),
            );
        }
    };
    let path = req.path.split('?').next().unwrap_or(req.path.as_str());

    match (req.method.as_str(), path) {
        ("OPTIONS", _) => write_empty(&mut stream, 204),
        ("GET", "/health") => write_text(&mut stream, 200, "ok\n"),
        ("GET", "/metrics") => write_text(&mut stream, 200, &metrics_response(&state)),
        ("GET", "/status") => write_json(&mut stream, 200, &status_response(&state)),
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &models_response(&state.config.served_model_name),
        ),
        ("POST", "/v1/completions") => handle_completion(stream, state, req),
        ("POST", "/v1/chat/completions") => handle_chat(stream, state, req),
        _ => write_json(
            &mut stream,
            404,
            &error_response(&ApiError::not_found("not found")),
        ),
    }
}

fn handle_completion(mut stream: TcpStream, state: Arc<State>, req: Request) -> Result<(), String> {
    let content_type = req
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");
    if !content_type.is_empty() && !content_type.contains("application/json") {
        return write_json(
            &mut stream,
            415,
            &error_response(&ApiError::invalid("content-type must be application/json")),
        );
    }

    let completion_req: CompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            return write_json(
                &mut stream,
                400,
                &error_response(&ApiError::invalid(format!("invalid JSON body: {e}"))),
            )
        }
    };

    let prepared =
        match prepare_completion_request(completion_req, &state.config.served_model_name, 256) {
            Ok(p) => p,
            Err(e) => return write_json(&mut stream, e.status, &error_response(&e)),
        };

    let created = created_unix();
    let id = completion_id(created);
    let return_token_ids = prepared.return_token_ids;
    let prompt_token_ids_for_logprobs = prepared.prompt_token_ids.clone();
    match state.worker.generate(GenerateRequest {
        prompt: prepared.prompt,
        prompt_token_ids: prepared.prompt_token_ids,
        max_tokens: prepared.max_tokens,
        sampling: prepared.sampling,
        add_bos: prepared.add_bos,
        ignore_eos: prepared.ignore_eos,
        prompt_logprobs: prepared.prompt_logprobs,
        images: Vec::new(),
    }) {
        Ok(out) => {
            let generated_token_ids = return_token_ids.then_some(out.token_ids.as_slice());
            write_json(
                &mut stream,
                200,
                &text_completion_response(
                    &id,
                    &state.config.served_model_name,
                    created,
                    &out.text,
                    out.prompt_tokens,
                    out.completion_tokens,
                    out.finish_reason.as_str(),
                    generated_token_ids,
                    prompt_token_ids_for_logprobs.as_deref(),
                    out.prompt_logprobs.as_deref(),
                ),
            )
        }
        Err(e) => write_generate_error(&mut stream, e),
    }
}

fn handle_chat(mut stream: TcpStream, state: Arc<State>, req: Request) -> Result<(), String> {
    let content_type = req
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");
    if !content_type.is_empty() && !content_type.contains("application/json") {
        return write_json(
            &mut stream,
            415,
            &error_response(&ApiError::invalid("content-type must be application/json")),
        );
    }

    let chat_req: ChatCompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            return write_json(
                &mut stream,
                400,
                &error_response(&ApiError::invalid(format!("invalid JSON body: {e}"))),
            )
        }
    };

    let prepared = match prepare_chat_request(
        chat_req,
        &state.config.served_model_name,
        256,
        state.config.default_system_prompt.as_deref(),
    ) {
        Ok(p) => p,
        Err(e) => return write_json(&mut stream, e.status, &error_response(&e)),
    };

    let created = created_unix();
    let id = completion_id(created);
    match state.worker.generate(GenerateRequest {
        prompt: prepared.prompt,
        prompt_token_ids: None,
        max_tokens: prepared.max_tokens,
        sampling: prepared.sampling,
        add_bos: true,
        ignore_eos: prepared.ignore_eos,
        prompt_logprobs: false,
        images: prepared.images,
    }) {
        Ok(out) => write_json(
            &mut stream,
            200,
            &completion_response(
                &id,
                &state.config.served_model_name,
                created,
                &out.text,
                out.prompt_tokens,
                out.completion_tokens,
                out.finish_reason.as_str(),
            ),
        ),
        Err(e) => write_generate_error(&mut stream, e),
    }
}

fn write_generate_error(stream: &mut TcpStream, err: GenerateError) -> Result<(), String> {
    let err = api_error_for_generate(err);
    write_json(stream, err.status, &error_response(&err))
}

fn api_error_for_generate(err: GenerateError) -> ApiError {
    match err {
        GenerateError::Busy { .. } => ApiError::busy("server is at capacity; retry later"),
        GenerateError::Invalid(error) => ApiError::invalid(error),
        GenerateError::Engine(error) => {
            tracing::error!(error = %error, "generation failed");
            ApiError::internal("generation failed")
        }
    }
}

/// Plaintext serving counters (prometheus exposition style, no crate).
fn metrics_response(state: &State) -> String {
    use std::sync::atomic::Ordering;

    use crate::worker::ServeMetrics;

    let m = state.worker.metrics();
    let stats = state.worker.stats();
    format!(
        "rvllm_requests_ok {}\n\
         rvllm_requests_err {}\n\
         rvllm_prompt_tokens_total {}\n\
         rvllm_completion_tokens_total {}\n\
         rvllm_decode_tok_s_ema {:.2}\n\
         rvllm_ttft_ms_ema {:.1}\n\
         rvllm_active_seats {}\n\
         rvllm_graph_captures_total {}\n",
        m.requests_ok.load(Ordering::Relaxed),
        m.requests_err.load(Ordering::Relaxed),
        m.prompt_tokens_total.load(Ordering::Relaxed),
        m.completion_tokens_total.load(Ordering::Relaxed),
        ServeMetrics::load_f64(&m.decode_tok_s_ema),
        ServeMetrics::load_f64(&m.ttft_ms_ema),
        stats.in_flight,
        m.graph_captures_total.load(Ordering::Relaxed),
    )
}

fn status_response(state: &State) -> Value {
    let stats = state.worker.stats();
    serde_json::json!({
        "object": "rvllm.status",
        "model": state.config.served_model_name,
        "backend": state.config.backend.as_str(),
        "max_model_len": state.config.max_model_len,
        "max_num_seqs": state.config.max_num_seqs,
        "max_inflight_requests": stats.max_inflight,
        "in_flight_requests": stats.in_flight
    })
}

fn try_acquire_connection(active: Arc<AtomicUsize>, limit: usize) -> Option<ConnectionPermit> {
    let acquired = active
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            (current < limit).then(|| current + 1)
        })
        .is_ok();
    acquired.then(|| ConnectionPermit { active })
}

fn read_request(
    stream: &mut TcpStream,
    api_key: Option<&str>,
) -> Result<Request, ReadRequestError> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(idx) = find_header_end(&buf) {
            if idx > MAX_HEADER_BYTES {
                return Err(ReadRequestError::BadRequest(
                    "request headers too large".into(),
                ));
            }
            break idx;
        }
        if buf.len() >= MAX_HEADER_BYTES + 4 {
            return Err(ReadRequestError::BadRequest(
                "request headers too large".into(),
            ));
        }
        let n = stream
            .read(&mut tmp)
            .map_err(|e| ReadRequestError::BadRequest(format!("read: {e}")))?;
        if n == 0 {
            return Err(ReadRequestError::BadRequest(
                "connection closed before headers".into(),
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = std::str::from_utf8(&buf[..header_end])
        .map_err(|e| ReadRequestError::BadRequest(format!("request headers are not UTF-8: {e}")))?;
    let mut lines = head.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| ReadRequestError::BadRequest("empty request".into()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ReadRequestError::BadRequest("missing method".into()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| ReadRequestError::BadRequest("missing path".into()))?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| ReadRequestError::BadRequest("missing HTTP version".into()))?;
    if parts.next().is_some() || method.len() > 16 || path.len() > 8 * 1024 {
        return Err(ReadRequestError::BadRequest("invalid request line".into()));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(ReadRequestError::BadRequest(format!(
            "unsupported HTTP version: {version}"
        )));
    }

    let mut headers = HashMap::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        if index >= MAX_HEADERS {
            return Err(ReadRequestError::BadRequest("too many headers".into()));
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| ReadRequestError::BadRequest("malformed header".into()))?;
        let key = key.trim().to_ascii_lowercase();
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(ReadRequestError::BadRequest("invalid header name".into()));
        }
        if headers.insert(key, value.trim().to_string()).is_some() {
            return Err(ReadRequestError::BadRequest(
                "duplicate request header".into(),
            ));
        }
    }
    if headers.contains_key("transfer-encoding") {
        return Err(ReadRequestError::BadRequest(
            "transfer encoding is unsupported".into(),
        ));
    }

    let route = path.split('?').next().unwrap_or(path.as_str());
    if !(method == "GET" && route == "/health"
        || bearer_authorized(api_key, headers.get("authorization").map(String::as_str)))
    {
        return Err(ReadRequestError::Unauthorized);
    }

    let content_len = match headers.get("content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| ReadRequestError::BadRequest("invalid content-length".into()))?,
        None => 0,
    };
    if content_len > MAX_BODY_BYTES {
        return Err(ReadRequestError::BadRequest(
            "request body too large".into(),
        ));
    }

    let body_start = header_end + 4;
    while buf.len().saturating_sub(body_start) < content_len {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| ReadRequestError::BadRequest(format!("read body: {e}")))?;
        if n == 0 {
            return Err(ReadRequestError::BadRequest(
                "connection closed before body".into(),
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[body_start..body_start + content_len].to_vec();

    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn bearer_authorized(api_key: Option<&str>, authorization: Option<&str>) -> bool {
    let Some(expected) = api_key else {
        return true;
    };
    let Some(value) = authorization else {
        return false;
    };
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer")
        || token.is_empty()
        || token.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return false;
    }
    constant_time_eq(expected.as_bytes(), token.as_bytes())
}

fn constant_time_eq(expected: &[u8], provided: &[u8]) -> bool {
    let mut difference = expected.len() ^ provided.len();
    let max_len = expected.len().max(provided.len());
    for index in 0..max_len {
        let left = expected.get(index).copied().unwrap_or(0);
        let right = provided.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_empty(stream: &mut TcpStream, status: u16) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        status_text(status),
    );
    stream
        .write_all(head.as_bytes())
        .map_err(|e| format!("write response: {e}"))
}

fn write_text(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), String> {
    write_response(stream, status, "text/plain; charset=utf-8", body.as_bytes())
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|e| format!("json serialize: {e}"))?;
    write_response(stream, status, "application/json", &body)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(status),
        content_type,
        body.len(),
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|e| format!("write response: {e}"))
}

fn write_unauthorized(stream: &mut TcpStream) -> Result<(), String> {
    let value = error_response(&ApiError {
        status: 401,
        message: "invalid or missing bearer token".into(),
        error_type: "authentication_error",
    });
    let body = serde_json::to_vec(&value).map_err(|e| format!("json serialize: {e}"))?;
    let head = format!(
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nWWW-Authenticate: Bearer\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len(),
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|e| format!("write response: {e}"))
}

fn status_text(status: u16) -> String {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        429 => "Too Many Requests",
        415 => "Unsupported Media Type",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    format!("{status} {reason}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        (client, server)
    }

    #[test]
    fn bearer_auth_is_strict() {
        assert!(bearer_authorized(None, None));
        assert!(bearer_authorized(Some("secret"), Some("Bearer secret")));
        assert!(bearer_authorized(Some("secret"), Some("bearer secret")));
        assert!(!bearer_authorized(Some("secret"), None));
        assert!(!bearer_authorized(Some("secret"), Some("Basic secret")));
        assert!(!bearer_authorized(Some("secret"), Some("Bearer secret ")));
        assert!(!bearer_authorized(Some("secret"), Some("Bearer secreu")));
        assert!(!bearer_authorized(Some("secret"), Some("Bearer secret0")));
    }

    #[test]
    fn rejects_unauthorized_request_before_body_read() {
        let (mut client, mut server) = tcp_pair();
        server
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        client
            .write_all(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 1000\r\n\r\n")
            .unwrap();
        assert!(matches!(
            read_request(&mut server, Some("secret")),
            Err(ReadRequestError::Unauthorized)
        ));
    }

    #[test]
    fn health_is_the_only_unauthenticated_route() {
        let (mut health_client, mut health_server) = tcp_pair();
        health_client
            .write_all(b"GET /health HTTP/1.1\r\n\r\n")
            .unwrap();
        assert!(read_request(&mut health_server, Some("secret")).is_ok());

        let (mut status_client, mut status_server) = tcp_pair();
        status_client
            .write_all(b"GET /status HTTP/1.1\r\n\r\n")
            .unwrap();
        assert!(matches!(
            read_request(&mut status_server, Some("secret")),
            Err(ReadRequestError::Unauthorized)
        ));
    }

    #[test]
    fn connection_gate_is_bounded() {
        let active = Arc::new(AtomicUsize::new(0));
        let first = try_acquire_connection(Arc::clone(&active), 1).unwrap();
        assert!(try_acquire_connection(Arc::clone(&active), 1).is_none());
        drop(first);
        assert!(try_acquire_connection(active, 1).is_some());
    }

    #[test]
    fn internal_engine_details_are_not_returned_to_clients() {
        let error = api_error_for_generate(GenerateError::Engine(
            "/private/model/path: CUDA_ERROR_ILLEGAL_ADDRESS".into(),
        ));
        assert_eq!(error.status, 500);
        assert_eq!(error.message, "generation failed");
        assert!(!error.message.contains("/private"));
        assert!(!error.message.contains("CUDA"));
    }

    #[test]
    fn invalid_pretokenized_prompt_is_an_http_400() {
        let error = api_error_for_generate(GenerateError::Invalid(
            "prompt token ID is outside the loaded model vocabulary".into(),
        ));
        assert_eq!(error.status, 400);
        assert_eq!(
            error.message,
            "prompt token ID is outside the loaded model vocabulary"
        );
    }
}
