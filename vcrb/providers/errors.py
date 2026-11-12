"""异常 → 错误类型分类（duck-typed，兼容 openai SDK 异常与测试桩）。"""

from __future__ import annotations

from typing import Any


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
    """截断的异常描述（不得包含鉴权头信息；openai 异常 message 默认安全）。"""
    text = f"{type(exc).__name__}: {exc}"
    return text[:limit]
