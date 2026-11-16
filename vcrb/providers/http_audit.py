"""Optional, private per-HTTP metering audit; amounts are never inferred bills."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from uuid import uuid4

from .errors import classify_exception


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _identifier(value):
    if (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/=+-]{0,255}", value)
            and not value.lower().startswith(("sk-", "bearer"))):
        return value
    return None


def response_identity(response=None, *, headers=None):
    """Keep request IDs distinct from response IDs, with explicit provenance.

    A response ``id`` is deliberately never promoted to a request ID. Only
    selected ID headers and documented response fields are inspected.
    """
    inherited = _get(response, "_http_audit_identity")
    if isinstance(inherited, dict):
        return dict(inherited)
    result = {"provider_request_id": None, "provider_request_id_source": None,
              "response_id": None, "response_id_source": None}
    if headers is None:
        raw_response = _get(response, "response")
        headers = _get(raw_response, "headers")
    selected_headers = {str(k).lower(): v for k, v in headers.items()} if hasattr(headers, "items") else {}
    for name in ("x-request-id", "request-id", "x-dashscope-request-id", "x-ms-request-id"):
        value = _identifier(selected_headers.get(name))
        if value:
            result.update(provider_request_id=value, provider_request_id_source="response_header." + name)
            break
    sdk_id = _identifier(_get(response, "_request_id"))
    if sdk_id and result["provider_request_id"] is None:
        result.update(provider_request_id=sdk_id, provider_request_id_source="sdk._request_id")
    native = _get(response, "_native_response") or _get(response, "_native_data") or response
    body_id = _identifier(_get(native, "request_id"))
    if body_id and result["provider_request_id"] is None:
        result.update(provider_request_id=body_id, provider_request_id_source="response_body.request_id")
    response_id = _identifier(_get(native, "id"))
    if response_id:
        result.update(response_id=response_id, response_id_source="response_body.id")
    events = _get(native, "events", [])
    ids = list(dict.fromkeys(value for item in events if (value := _identifier(_get(item, "request_id"))))) if isinstance(events, list) else []
    if ids:
        result["sse_request_ids"] = ids
        if len(ids) == 1 and result["provider_request_id"] is None:
            result.update(provider_request_id=ids[0], provider_request_id_source="sse_event.request_id")
    return result


def _numeric_tree(value):
    """Metering only: no prompts, content, credentials or arbitrary strings."""
    if value is None or isinstance(value, bool) or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _numeric_tree(item) for key, item in value.items()
                if re.fullmatch(r"[A-Za-z0-9_]{1,100}", str(key))
                and not any(word in str(key).lower() for word in ("key", "secret", "authorization", "header"))}
    if isinstance(value, (list, tuple)):
        return [_numeric_tree(item) for item in value]
    return None


@dataclass
class _Receipt:
    common: dict[str, Any]
    delegate_receipt: Any = None
    monotonic: float = field(default_factory=time.monotonic)
    search_identity: dict[str, Any] = field(default_factory=dict)


class AttemptAuditObserver:
    """Compose an existing budget observer with fsynced start/end JSONL rows.

    A durable start is appended after budget admission and before transport.
    Missing end rows identify interrupted requests; they must not be assumed
    unbilled. SDK retries remain disabled by the existing provider.
    """

    def __init__(self, path: Path | str, run_id: str, delegate=None):
        self.path, self.run_id, self.delegate = Path(path), run_id, delegate
        self._lock = threading.Lock()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    def _append(self, record):
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "ab", closefd=False) as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)

    def _before(self, kind, model, image, prompt, attempt):
        method = "before_search" if kind == "search" else "before_attempt"
        underlying = getattr(self.delegate, method)(model, image, prompt, attempt) if self.delegate else None
        receipt = _Receipt({
            "schema_version": 1, "audit_attempt_id": uuid4().hex, "run_id": self.run_id,
            "model_id": model.id, "provider": model.provider, "kind": kind,
            "attempt_no": attempt, "started_at": self._now(),
            "image_encoded_sha256": hashlib.sha256(base64.b64decode(image.split(",", 1)[1], validate=True)).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "budget_reservation_id": _identifier(underlying),
        }, underlying)
        self._append({**receipt.common, "event": "http_attempt_start"})
        return receipt

    def before_attempt(self, model, image_data_url, prompt, attempt_no):
        return self._before("inference", model, image_data_url, prompt, attempt_no)

    def before_search(self, model, image_data_url, prompt, attempt_no=1):
        return self._before("search", model, image_data_url, prompt, attempt_no)

    def observe_search_response(self, receipt, model, *, response=None, headers=None):
        # Optional hook keeps older observer.after_search signatures compatible.
        receipt.search_identity = response_identity(response, headers=headers)

    def _after(self, receipt, model, *, result=None, error=None, units=None):
        kind = receipt.common["kind"]
        try:
            if self.delegate:
                if kind == "search":
                    self.delegate.after_search(receipt.delegate_receipt, model, units=units, error=error)
                else:
                    self.delegate.after_attempt(receipt.delegate_receipt, model, result=result, error=error)
        finally:
            identity = receipt.search_identity if kind == "search" else getattr(result, "http_metadata", {}) or {}
            if not identity and result is not None:
                metadata = getattr(result, "search_metadata", {}) or {}
                identity = response_identity(metadata.get("native"))
            if error is not None:
                body = _get(error, "body")
                identity = response_identity(body, headers=_get(_get(error, "response"), "headers"))
            error_type = classify_exception(error)[0] if isinstance(error, Exception) else getattr(result, "error_type", None)
            if error is not None and not isinstance(error, Exception):
                error_type = "interrupted"
            metadata = getattr(result, "search_metadata", {}) or {}
            self._append({**receipt.common, "event": "http_attempt_end", "ended_at": self._now(),
                          "duration_ms": max(0, round((time.monotonic() - receipt.monotonic) * 1000)),
                          "ok": error is None and (bool(result.ok) if kind == "inference" and result is not None else kind == "search"),
                          "error_type": error_type,
                          "http_status": _get(error, "status_code") if error is not None else getattr(result, "http_status", None) if kind == "inference" else 200,
                          "usage": _numeric_tree(getattr(result, "usage", {}) or {}),
                          "search_usage": {"units": {"search": units}, "known_units": error is None} if kind == "search" else _numeric_tree(metadata.get("billing", {})),
                          **identity})

    def after_attempt(self, receipt, model, *, result=None, error=None):
        self._after(receipt, model, result=result, error=error)

    def after_search(self, receipt, model, *, units=1, error=None):
        self._after(receipt, model, units=units, error=error)
