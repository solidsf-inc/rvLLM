"""Live checks for the OpenAI-compatible endpoints rvLLM implements."""

import json
import os
import unittest
import urllib.error
import urllib.parse
import urllib.request


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = urllib.request.build_opener(RejectRedirects())


def validate_base_url(raw):
    if not raw or raw.strip() != raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ValueError("RVLLM_URL must not contain whitespace or control characters")
    try:
        endpoint = urllib.parse.urlsplit(raw)
        port = endpoint.port
    except ValueError as error:
        raise ValueError(f"invalid RVLLM_URL: {error}") from error
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("RVLLM_URL must be an http(s) origin")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("RVLLM_URL must not contain credentials, a query, or a fragment")
    if endpoint.path not in ("", "/"):
        raise ValueError("RVLLM_URL must be an origin without an API path")
    if endpoint.scheme == "http" and endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("plaintext HTTP is allowed only for loopback endpoints")
    netloc = endpoint.hostname
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit((endpoint.scheme, netloc, "", "", ""))


_RAW_BASE_URL = os.environ.get("RVLLM_URL")
BASE_URL = validate_base_url(_RAW_BASE_URL) if _RAW_BASE_URL else None
MODEL = os.environ.get("RVLLM_MODEL")
TIMEOUT = float(os.environ.get("RVLLM_TEST_TIMEOUT", "60"))


def request(path, method="GET", payload=None):
    headers = {}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    api_key = os.environ.get("RVLLM_API_KEY")
    if api_key:
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in api_key):
            raise ValueError("RVLLM_API_KEY contains whitespace or control characters")
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{BASE_URL.rstrip('/')}{path}", body, headers, method=method)
    return NO_REDIRECT_OPENER.open(req, timeout=TIMEOUT)


class EndpointPolicy(unittest.TestCase):
    def test_rejects_unsafe_origins(self):
        self.assertEqual(validate_base_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")
        self.assertEqual(validate_base_url("https://example.com"), "https://example.com")
        for url in (
            "http://example.com",
            "https://user:secret@example.com",
            "https://example.com/api",
            "https://example.com?token=x",
        ):
            with self.assertRaises(ValueError):
                validate_base_url(url)


@unittest.skipUnless(BASE_URL and MODEL, "set RVLLM_URL and RVLLM_MODEL")
class ApiCompatibility(unittest.TestCase):
    def chat_payload(self, stream=False):
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": stream,
        }

    def test_health(self):
        with request("/health") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"ok\n")

    def test_models(self):
        with request("/v1/models") as response:
            payload = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertIn(MODEL, [model["id"] for model in payload["data"]])

    def test_chat_completion(self):
        with request("/v1/chat/completions", "POST", self.chat_payload()) as response:
            payload = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")

    def test_rejects_chat_stream(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            request("/v1/chat/completions", "POST", self.chat_payload(True))
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(
            json.load(raised.exception)["error"]["message"],
            "stream=true is not supported; use stream=false",
        )

    def test_rejects_stop(self):
        payload = self.chat_payload()
        payload["stop"] = "END"
        with self.assertRaises(urllib.error.HTTPError) as raised:
            request("/v1/chat/completions", "POST", payload)
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(
            json.load(raised.exception)["error"]["message"],
            "stop is not supported; omit stop",
        )

    def test_text_completion(self):
        payload = {
            "model": MODEL,
            "prompt": "Continue: alpha beta",
            "max_tokens": 8,
            "temperature": 0,
        }
        with request("/v1/completions", "POST", payload) as response:
            body = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertEqual(body["object"], "text_completion")
        self.assertIn("text", body["choices"][0])

    def test_rejects_multiple_choices(self):
        payload = self.chat_payload()
        payload["n"] = 2
        with self.assertRaises(urllib.error.HTTPError) as raised:
            request("/v1/chat/completions", "POST", payload)
        self.assertEqual(raised.exception.code, 400)

    def test_unknown_model(self):
        payload = self.chat_payload()
        payload["model"] = "not-served"
        with self.assertRaises(urllib.error.HTTPError) as raised:
            request("/v1/chat/completions", "POST", payload)
        self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
