"""异常 → 错误类型分类（duck-typed，兼容 openai SDK 异常与测试桩）。"""

from __future__ import annotations

import re


_ACCOUNT_ID_RE = re.compile(r"\b(?:org|proj)-[a-z0-9]+\b", re.IGNORECASE)
_KEY_ID_RE = re.compile(r"<\s*ak-[^<>\r\n]+>", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(?<![a-z0-9_])sk-[a-z0-9_-]{8,}", re.IGNORECASE)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def classify_exception(exc: BaseException) -> tuple[str, bool]:
    """返回 (error_type, retryable)。

    error_type 词汇表：
      network / timeout / rate_limit / server_error   → 可重试
      auth_error / model_not_found / bad_request / internal_error → 不可重试
    """
    name = type(exc).__name__
    status = _status_of(exc)

    if name == "APITimeoutError" or isinstance(exc, TimeoutError):
        return "timeout", True
    if name == "APIConnectionError":
        return "network", True
    if status is not None:
        if status == 429:
            return "rate_limit", True
        if status >= 500:
            return "server_error", True
        if status in (401, 403):
            return "auth_error", False
        if status == 404:
            return "model_not_found", False
        return "bad_request", False
    return "internal_error", False


def error_summary(exc: BaseException, limit: int = 400) -> str:
    """去标识化后截断，避免厂商异常正文中的账户/Key 标识进入日志。

    必须先处理完整异常再截断，否则横跨截断边界的 Key 会泄露前缀。
    保留异常类型、状态码和限流描述，供错误分类及故障排查使用。
    """
    text = f"{type(exc).__name__}: {exc}"
    text = _ACCOUNT_ID_RE.sub("[account-id]", text)
    text = _KEY_ID_RE.sub("[api-key-id]", text)
    text = _API_KEY_RE.sub("[api-key]", text)
    return text[:limit]
