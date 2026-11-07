"""Local Ollama structured-planning adapter, stdlib-only.

Explicitly constrained to loopback URLs. No third-party API or outbound search.
The existing production LlamaIndex RAG configuration is not changed.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from urllib.error import URLError


@dataclass(frozen=True)
class ModelReply:
    content: str
    total_tokens: int
    model: str


class OllamaAdapter:
    def __init__(
        self, model: str = "qwen3.5:27b", base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 180, num_ctx: int = 8192, max_predict: int = 768,
        transport=None,
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Research Agent Ollama URL must use local HTTP loopback")
        if parsed.username or parsed.password or parsed.path.strip("/") or parsed.query:
            raise ValueError("Only base Ollama origin is accepted")
        if not model or timeout_seconds <= 0 or num_ctx < 1024 or max_predict <= 0:
            raise ValueError("Invalid local model or resource budget")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds
        self.num_ctx = num_ctx
        self.max_predict = max_predict
        self._transport = transport
        self.last_tokens = 0
        self.total_tokens = 0

    def _send(self, payload: dict) -> dict:
        if self._transport:
            result = self._transport(payload)
        else:
            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as stream:
                    result = json.loads(stream.read(16 * 1024 * 1024))
            except (OSError, URLError, ValueError) as exc:
                raise RuntimeError(f"Ollama request failed: {type(exc).__name__}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
            raise RuntimeError("Unexpected Ollama response shape")
        self.last_tokens = max(0, int(result.get("prompt_eval_count") or 0)) + max(
            0, int(result.get("eval_count") or 0)
        )
        self.total_tokens += self.last_tokens
        return result

    def chat(self, system: str, question: str, schema: dict | None = None) -> ModelReply:
        payload = {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question}],
            "options": {"temperature": 0.1, "num_ctx": self.num_ctx,
                        "num_predict": self.max_predict},
        }
        if schema is not None:
            payload["format"] = schema
        raw = self._send(payload)
        content = str(raw["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return ModelReply(content, self.last_tokens, self.model)

    def structured(self, system: str, question: str, schema: dict) -> dict:
        reply = self.chat(system, question + "\nReturn ONLY valid JSON.", schema)
        try:
            parsed = json.loads(reply.content)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid structured Ollama output") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from Ollama")
        return parsed
