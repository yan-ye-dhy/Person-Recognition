"""Mock Provider —— 全链路离线测试（不发起任何网络请求、零费用）。

脚本化响应（script[model_id] = [entry, ...]，按调用序循环）：
  {"text": "..."}                    → ok，返回该文本
  {"fail_once": "rate_limit"}        → 第一次尝试抛可重试异常，重试后 ok
  {"fail": "server_error"}           → 每次尝试都失败 → api_failure
未脚本化的模型返回确定性 JSON 文本（按 model_id 哈希生成角色名），便于幂等测试。
"""

from __future__ import annotations

from typing import Any

from ..models import ModelSpec
from .base import GenerationParams, ProviderResult, VisionProvider


class _MockHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "mock http error") -> None:
        super().__init__(message)
        self.status_code = status_code


_STATUS_BY_TYPE = {
    "network": None,  # 连接错误无状态码，靠类型名分类 → 用专用子类
    "timeout": None,
    "rate_limit": 429,
    "server_error": 500,
    "auth_error": 401,
    "bad_request": 400,
}


class _MockConnectionError(Exception):
    pass


class _MockTimeoutError(TimeoutError):
    pass


def _make_exception(error_type: str) -> Exception:
    if error_type == "network":
        return _MockConnectionError("mock connection error")
    if error_type == "timeout":
        return _MockTimeoutError("mock timeout")
    status = _STATUS_BY_TYPE.get(error_type, 500)
    return _MockHTTPError(status, f"mock {error_type}")


class _FakeCompletion:
    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content
            self.reasoning_content = None

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _FakeCompletion._Msg(content)
            self.finish_reason = "stop"

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 20
        total_tokens = 120

    def __init__(self, content: str, model_id: str) -> None:
        self.choices = [_FakeCompletion._Choice(content)]
        self.model = f"{model_id}#mock-served"
        self.usage = _FakeCompletion._Usage()


class MockProvider(VisionProvider):
    name = "mock"

    def __init__(
        self,
        script: dict[str, list[dict[str, Any]]] | None = None,
        sleeper: Any = None,
        max_attempts: int = 4,
    ) -> None:
        self.script = script or {}
        self._call_counts: dict[str, int] = {}
        self._sleeper = sleeper if sleeper is not None else (lambda _s: None)
        self.max_attempts = max_attempts

    def recognize(
        self,
        model: ModelSpec,
        image_data_url: str,
        system_prompt: str,
        user_prompt: str,
        gen: GenerationParams,
    ) -> ProviderResult:
        entries = self.script.get(model.id, [])
        index = self._call_counts.get(model.id, 0)
        self._call_counts[model.id] = index + 1
        entry = entries[index % len(entries)] if entries else None

        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            has_fail_spec = entry is not None and (
                "fail" in entry or "fail_once" in entry
            )
            if has_fail_spec:
                error_type = str(
                    entry.get("fail_once") or entry.get("fail") or "server_error"
                )
                persistent = bool(entry.get("fail"))
                if not persistent and attempts > 1:
                    break  # fail_once：首次尝试已失败，本次重试成功
                if persistent and attempts >= self.max_attempts:
                    return ProviderResult(
                        ok=False,
                        attempts=attempts,
                        error_type=error_type,
                        error_message=f"mock persistent {error_type}",
                        http_status=_STATUS_BY_TYPE.get(error_type),
                    )
                self._sleeper(0)
                continue  # 模拟一次失败的传输尝试（异常在适配器内部被消化）
            break

        text = (
            entry["text"]
            if entry is not None and "text" in entry
            else self._default_text(model)
        )
        resp = _FakeCompletion(str(text), model.id)
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider._parse_response(resp, attempts, 0)

    @staticmethod
    def _default_text(model: ModelSpec) -> str:
        return (
            '{"character_name": "mock角色", "is_uncertain": false, '
            '"reason": "deterministic response for ' + model.id + '"}'
        )
