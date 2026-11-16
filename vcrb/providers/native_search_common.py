"""One-request native search transports using the existing attempt accounting.

Each ``create`` below performs exactly one HTTP POST. Retries, admission and
latency remain in OpenAICompatProvider; there are no SDK retries or polling.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import require_api_key
from ..models import ProviderSpec
from .base import GenerationParams
from .http_audit import response_identity
from .openai_compat import OpenAICompatProvider, _safe_json


def https_endpoint(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Native search requires an HTTPS provider base URL without credentials or query")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + suffix, "", ""))


def validate_image_data_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Native image search requires the supplied Base64 image data URL")
    header, separator, encoded = value.partition(",")
    if (not separator or not header.startswith("data:image/") or not header.endswith(";base64")
            or header[5:-7] not in {"image/jpeg", "image/png", "image/gif", "image/webp"}):
        raise ValueError("Unsupported image data URL")
    try:
        if not base64.b64decode(encoded, validate=True):
            raise ValueError("Empty image")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid image Base64") from error
    return value


class NativeSearchHTTPError(Exception):
    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        self.response = response
        try:
            self.body = _safe_json(response.json())
        except ValueError:
            self.body = {"message": "Provider returned a non-JSON HTTP error"}
            # DashScope can deliver an SSE error body even with HTTP400.
            # Keep the business code so an account-wide denial stops new calls.
            for line in response.text.splitlines():
                if line.startswith('data:'):
                    try:
                        candidate = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if isinstance(candidate, dict) and candidate.get('code'):
                        self.body = _safe_json(candidate)
                        break
        super().__init__(f"HTTP {self.status_code}: {self.body}")


class APIConnectionError(ConnectionError):
    """Classified like the compatible SDK, without leaking request secrets."""


class NativeSearchProvider(OpenAICompatProvider):
    """Bridge one native HTTP exchange to the existing provider attempt loop."""

    evidence_vendor = ""

    def __init__(self, provider_spec: ProviderSpec, *,
                 http_client_factory: Callable[[], httpx.Client] | None = None, **kwargs):
        super().__init__(provider_spec, **kwargs)
        self._http_client_factory = http_client_factory
        self._http_client: httpx.Client | None = None

    def build_request_kwargs(self, model, messages: list[dict], gen: GenerationParams) -> dict:
        kwargs = super().build_request_kwargs(model, messages, gen)
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)
        # Validate the actual native payload before budget admission or HTTP.
        self.native_payload(kwargs)
        return kwargs

    def native_payload(self, chat_kwargs: dict) -> dict:
        raise NotImplementedError

    def completion_from_native(self, data: dict):
        raise NotImplementedError

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                api_key = require_api_key(self.spec.api_key_env)
                self._http_client = (self._http_client_factory() if self._http_client_factory else
                                     httpx.Client(timeout=self.timeout, follow_redirects=False))

                def create(**kwargs):
                    payload = self.native_payload(kwargs)
                    try:
                        response = self._http_client.post(self.endpoint, json=payload, headers={
                            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                        })
                    except httpx.TimeoutException as error:
                        raise TimeoutError("Native search request timed out") from error
                    except httpx.RequestError as error:
                        raise APIConnectionError("Native search connection failed") from error
                    if response.status_code >= 400:
                        raise NativeSearchHTTPError(response)
                    data = response.json()
                    completion = self.completion_from_native(data)
                    completion._http_audit_identity = response_identity(data, headers=response.headers)
                    return completion

                self._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return self._client

    def _parse_response(self, response, attempts: int, latency_ms: int):
        from .search_evidence import normalize_search_evidence

        result = super()._parse_response(response, attempts, latency_ms)
        native = getattr(response, "_native_response", {})
        result.search_metadata = normalize_search_evidence(self.evidence_vendor, native)
        error = getattr(response, "_native_error", None)
        if error:
            from .errors import is_balance_error
            # A completed HTTP exchange can fail after paid inference/search.
            # Keep its returned usage and traces; do not issue a hidden retry.
            result.ok = False
            result.error_type = 'insufficient_balance' if is_balance_error(error) else error.get("error_type", "server_error")
            result.error_message = json.dumps(_safe_json(error), ensure_ascii=False)[:400]
        return result

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
