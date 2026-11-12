"""Provider 层测试：Mock、错误分类、OpenAI 兼容适配器（假客户端注入）。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import make_image_bytes
from vcrb.config import load_settings
from vcrb.models import load_registry
from vcrb.providers import (
    GenerationParams,
    MockProvider,
    OpenAICompatProvider,
    ProviderResult,
    classify_exception,
)
from vcrb.providers.errors import error_summary

DATA_URL = "data:image/jpeg;base64," + "AAAA"
GEN = GenerationParams(temperature=0.1, max_tokens=512)
SYS = "system prompt"
USER = "user prompt"


# ---------------------------------------------------------------- 错误分类

class _StatusErr(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc,expected_type,retryable",
    [
        (TimeoutError("t"), "timeout", True),
        (_StatusErr(429), "rate_limit", True),
        (_StatusErr(500), "server_error", True),
        (_StatusErr(503), "server_error", True),
        (_StatusErr(401), "auth_error", False),
        (_StatusErr(403), "auth_error", False),
        (_StatusErr(404), "model_not_found", False),
        (_StatusErr(400), "bad_request", False),
        (ValueError("boom"), "internal_error", False),
    ],
)
def test_classify_exception(exc, expected_type, retryable):
    assert classify_exception(exc) == (expected_type, retryable)


def test_error_summary_truncates():
    assert len(error_summary(ValueError("x" * 1000), limit=100)) <= 120


# ---------------------------------------------------------------- MockProvider

def test_mock_default_deterministic():
    registry = load_registry()
    provider = MockProvider()
    model = registry.get("deepseek-flash")
    r1 = provider.recognize(model, DATA_URL, SYS, USER, GEN)
    r2 = provider.recognize(model, DATA_URL, SYS, USER, GEN)
    assert r1.ok and r2.ok
    assert r1.raw_text == r2.raw_text  # 幂等测试的基础
    assert "mock-served" in (r1.served_model or "")


def test_mock_fail_once_retries_then_ok():
    registry = load_registry()
    provider = MockProvider(
        script={"deepseek-flash": [{"fail_once": "rate_limit", "text": '{"character_name": "x"}'}]}
    )
    r = provider.recognize(registry.get("deepseek-flash"), DATA_URL, SYS, USER, GEN)
    assert r.ok
    assert r.attempts == 2
    assert r.raw_text == '{"character_name": "x"}'


def test_mock_persistent_failure_is_api_failure():
    registry = load_registry()
    provider = MockProvider(
        script={"deepseek-flash": [{"fail": "server_error"}]}, max_attempts=3
    )
    r = provider.recognize(registry.get("deepseek-flash"), DATA_URL, SYS, USER, GEN)
    assert not r.ok
    assert r.error_type == "server_error"
    assert r.attempts == 3
    assert r.http_status == 500


def test_mock_script_cycles():
    registry = load_registry()
    provider = MockProvider(
        script={"glm-4.6v": [{"text": "A"}, {"text": "B"}]}
    )
    results = [
        provider.recognize(registry.get("glm-4.6v"), DATA_URL, SYS, USER, GEN).raw_text
        for _ in range(4)
    ]
    assert results == ["A", "B", "A", "B"]


# ---------------------------------------------------------------- OpenAI 兼容适配器


class FakeCompletions:
    def __init__(self, behaviors: list[Any]) -> None:
        self.behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class FakeClient:
    def __init__(self, behaviors: list[Any]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(behaviors)


def _resp(content: str, model: str = "served-version-1"):
    class Msg:
        pass

    class Choice:
        pass

    class Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class R:
        pass

    m = Msg()
    m.content = content
    m.reasoning_content = None
    c = Choice()
    c.message = m
    c.finish_reason = "stop"
    r = R()
    r.choices = [c]
    r.model = model
    r.usage = Usage()
    return r


def _provider(client: FakeClient) -> OpenAICompatProvider:
    registry = load_registry()
    spec = registry.providers["dashscope"]
    return OpenAICompatProvider(
        spec,
        settings=load_settings(),
        client_factory=lambda: client,
        sleeper=lambda _s: None,
    )


def test_openai_compat_ok_response():
    client = FakeClient([_resp('{"character_name": "测试"}')])
    provider = _provider(client)
    registry = load_registry()
    result = provider.recognize(
        registry.get("qwen3.8-max"), DATA_URL, SYS, USER, GEN
    )
    assert result.ok
    assert result.raw_text == '{"character_name": "测试"}'
    assert result.served_model == "served-version-1"
    assert result.usage["total_tokens"] == 15


def test_openai_compat_request_shape_and_fairness():
    client = FakeClient([_resp("ok")])
    provider = _provider(client)
    registry = load_registry()
    provider.recognize(registry.get("qwen3.8-max"), DATA_URL, SYS, USER, GEN)
    (call,) = client.chat.completions.calls
    # 消息结构：system + user(image_url + text)
    assert call["model"] == "qwen3.8-max"
    messages = call["messages"]
    assert messages[0] == {"role": "system", "content": SYS}
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "image_url", "image_url": {"url": DATA_URL}}
    assert user_content[1] == {"type": "text", "text": USER}
    # 不传任何工具/搜索参数
    assert "tools" not in call
    assert "tool_choice" not in call
    # Qwen 私有参数注入
    assert call["extra_body"] == {"enable_thinking": False}


def test_openai_compat_kimi_omits_temperature_and_extra_body():
    client = FakeClient([_resp("ok")])
    provider = _provider(client)
    registry = load_registry()
    provider.recognize(registry.get("kimi-k2.6"), DATA_URL, SYS, USER, GEN)
    (call,) = client.chat.completions.calls
    assert "temperature" not in call
    assert "extra_body" not in call
    # 对照：deepseek 允许 temperature
    provider.recognize(registry.get("deepseek-flash"), DATA_URL, SYS, USER, GEN)
    call2 = client.chat.completions.calls[-1]
    assert call2["temperature"] == 0.1


def test_openai_compat_retries_then_succeeds():
    client = FakeClient([_StatusErr(503), _resp("recovered")])
    provider = _provider(client)
    registry = load_registry()
    result = provider.recognize(registry.get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert result.ok
    assert result.attempts == 2
    assert result.raw_text == "recovered"


def test_openai_compat_auth_error_not_retried():
    client = FakeClient([_StatusErr(401)] + [_resp("never")] * 5)
    provider = _provider(client)
    registry = load_registry()
    result = provider.recognize(registry.get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert not result.ok
    assert result.error_type == "auth_error"
    assert result.attempts == 1  # 不可重试
    assert len(client.chat.completions.calls) == 1


def test_openai_compat_persistent_5xx_exhausts_retries():
    failures = [_StatusErr(500)] * 6
    client = FakeClient(failures)
    provider = _provider(client)
    registry = load_registry()
    result = provider.recognize(registry.get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert not result.ok
    assert result.error_type == "server_error"
    max_attempts = load_settings()["retry"]["max_attempts"]
    assert result.attempts == max_attempts
    assert len(client.chat.completions.calls) == max_attempts


def test_provider_result_defaults():
    r = ProviderResult(ok=False)
    assert r.raw_text == ""
    assert r.usage == {}
