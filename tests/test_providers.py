"""Provider 层测试：Mock、错误分类、OpenAI 兼容适配器（假客户端注入）。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
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


def test_error_summary_redacts_rate_limit_account_identifiers():
    org_id = "org-example012345"
    project_id = "proj-example987654"
    key_id = "<ak-example-key-id>"
    api_key = "sk-test-only-0123456789abcdef"
    exc = _StatusErr(429)
    exc.args = (
        f"429 rate_limit_reached_error: Your account {org_id} / {project_id} "
        f"{key_id} reached RPM rate limit; current:4, limit:3; "
        f"error_retry_after: 20; Authorization: Bearer {api_key}",
    )
    summary = error_summary(exc)
    for identifier in (org_id, project_id, key_id, api_key):
        assert identifier not in summary
    assert summary.startswith("_StatusErr: 429 rate_limit_reached_error:")
    assert "RPM rate limit; current:4, limit:3" in summary
    assert "error_retry_after: 20" in summary
    assert "[account-id]" in summary
    assert "[api-key-id]" in summary
    assert "[api-key]" in summary
    assert classify_exception(exc) == ("rate_limit", True)


@pytest.mark.parametrize(
    "identifier",
    ["org-ABC012", "<ak-test_012-key>", "sk-proj-test_0123456789ABCDEF"],
)
def test_error_summary_redacts_identifier_variants(identifier):
    summary = error_summary(ValueError(f"request failed: {identifier}"))
    assert identifier not in summary
    assert summary.startswith("ValueError: request failed:")


def test_error_summary_redacts_before_truncation():
    secret = "sk-test-only-" + "a" * 100
    summary = error_summary(ValueError(f"Bearer {secret}; more context"), limit=35)
    assert "sk-" not in summary
    assert secret[:12] not in summary
    assert "[api-key]" in summary
    assert len(summary) <= 35


@pytest.mark.parametrize("message", ["", "bad_request: unsupported temperature"])
def test_error_summary_without_identifiers_is_unchanged(message):
    assert error_summary(ValueError(message)) == f"ValueError: {message}"


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


def _resp(
    content: str | None,
    model: str = "served-version-1",
    *,
    reasoning: str | None = None,
    finish_reason: str = "stop",
):
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
    m.reasoning_content = reasoning
    c = Choice()
    c.message = m
    c.finish_reason = finish_reason
    r = R()
    r.choices = [c]
    r.model = model
    r.usage = Usage()
    return r


def _provider(client: FakeClient, *, settings=None, sleeper=None) -> OpenAICompatProvider:
    registry = load_registry()
    spec = registry.providers["dashscope"]
    return OpenAICompatProvider(
        spec,
        settings=load_settings() if settings is None else settings,
        client_factory=lambda: client,
        sleeper=(lambda _s: None) if sleeper is None else sleeper,
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
    assert result.reasoning_text == ""
    assert result.output_issue is None


def test_openai_compat_plain_response_without_reasoning_field():
    response = _resp('{"character_name": "安欣"}')
    del response.choices[0].message.reasoning_content
    client = FakeClient([response])
    result = _provider(client).recognize(
        load_registry().get("qwen3.8-max"), DATA_URL, SYS, USER, GEN
    )
    assert result.ok
    assert result.raw_text == '{"character_name": "安欣"}'
    assert result.reasoning_text == ""
    assert result.output_issue is None


def test_openai_compat_preserves_final_answer_separately_from_reasoning():
    final = '{"character_name": "安欣", "is_uncertain": false}'
    reasoning = '备选而非答案：{"character_name": "高启强"}'
    client = FakeClient([_resp(final, reasoning=reasoning)])
    result = _provider(client).recognize(
        load_registry().get("qwen3.8-max"), DATA_URL, SYS, USER, GEN
    )
    assert result.raw_text == final
    assert result.reasoning_text == reasoning
    assert result.output_issue is None


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_openai_compat_reasoning_json_cannot_become_final_answer(content):
    from vcrb.evaluation.parser import parse_response

    reasoning = '{"character_name": "安欣", "is_uncertain": false}'
    client = FakeClient([_resp(content, reasoning=reasoning)])
    result = _provider(client).recognize(
        load_registry().get("qwen3.8-max"), DATA_URL, SYS, USER, GEN
    )
    assert result.ok and result.http_status == 200  # 不将模型输出问题改写成网络失败
    assert result.error_type is None
    assert result.raw_text == (content or "")
    assert not parse_response(result.raw_text).parse_ok
    assert result.reasoning_text == reasoning
    assert result.output_issue == "empty_final_content"
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize("content", [None, '{"character_name":', '{"character_name": "安欣"}'])
def test_openai_compat_truncation_is_audited_without_paid_retry(content):
    client = FakeClient([
        _resp(content, reasoning="尚未完成的思考", finish_reason="length"),
        _resp("unexpected retry"),
    ])
    result = _provider(client).recognize(
        load_registry().get("qwen3.8-max"), DATA_URL, SYS, USER, GEN
    )
    assert result.ok and result.http_status == 200
    assert result.error_type is None
    assert result.raw_text == (content or "")
    assert result.finish_reason == "length"
    assert result.output_issue == "truncated_output"
    assert result.reasoning_text == "尚未完成的思考"
    assert result.attempts == 1
    assert len(client.chat.completions.calls) == 1


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
    client = FakeClient([_resp("ok"), _resp("ok")])
    provider = _provider(client)
    registry = load_registry()
    # 参数注入契约与当前在线模型配置分开测试。
    kimi = replace(registry.get("kimi-k2.6"), params={"pass_temperature": False})
    provider.recognize(kimi, DATA_URL, SYS, USER, GEN)
    (call,) = client.chat.completions.calls
    assert "temperature" not in call
    assert "extra_body" not in call
    # 对照：deepseek 允许 temperature
    provider.recognize(registry.get("deepseek-flash"), DATA_URL, SYS, USER, GEN)
    call2 = client.chat.completions.calls[-1]
    assert call2["temperature"] == 0.1


def test_openai_compat_token_budget_defaults_to_max_tokens():
    client = FakeClient([_resp("ok")])
    model = replace(load_registry().get("deepseek-flash"), params={})
    _provider(client).recognize(model, DATA_URL, SYS, USER, GEN)
    (request,) = client.chat.completions.calls
    assert request["max_tokens"] == GEN.max_tokens
    assert "max_completion_tokens" not in request


def test_openai_compat_k3_can_use_max_completion_tokens():
    client = FakeClient([_resp("ok")])
    model = replace(
        load_registry().get("deepseek-flash"),
        id="kimi-k3",
        provider="moonshot",
        vendor="Kimi",
        params={"pass_temperature": False, "token_limit_parameter": "max_completion_tokens"},
    )
    _provider(client).recognize(model, DATA_URL, SYS, USER, GEN)
    (request,) = client.chat.completions.calls
    assert request["model"] == "kimi-k3"
    assert request["max_completion_tokens"] == GEN.max_tokens
    assert "max_tokens" not in request


@pytest.mark.parametrize("parameter", ["max_output_tokens", "", None, 512, []])
def test_openai_compat_rejects_invalid_token_budget_parameter(parameter):
    client = FakeClient([])
    model = replace(
        load_registry().get("deepseek-flash"),
        params={"token_limit_parameter": parameter},
    )
    with pytest.raises(ValueError, match="token_limit_parameter"):
        _provider(client).recognize(model, DATA_URL, SYS, USER, GEN)
    assert client.chat.completions.calls == []


@pytest.mark.parametrize("extra_parameter", ["max_tokens", "max_completion_tokens"])
def test_openai_compat_extra_body_cannot_override_or_duplicate_token_budget(extra_parameter):
    client = FakeClient([])
    model = replace(
        load_registry().get("deepseek-flash"),
        params={"extra_body": {extra_parameter: 999}},
    )
    with pytest.raises(ValueError, match="extra_body"):
        _provider(client).recognize(model, DATA_URL, SYS, USER, GEN)
    assert client.chat.completions.calls == []


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


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def wall_time(self):
        return 1700000000.0 + self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _clocked_provider(monkeypatch, client, *, settings=None):
    clock = _FakeClock()
    monkeypatch.setattr("vcrb.providers.openai_compat.time.monotonic", clock.monotonic)
    monkeypatch.setattr("vcrb.providers.openai_compat.time.time", clock.wall_time)
    provider = _provider(client, settings=settings, sleeper=clock.sleep)
    return provider, clock


def test_openai_compat_min_interval_between_actual_attempts(monkeypatch):
    client = FakeClient([_resp("ok"), _resp("ok"), _resp("ok")])
    model = load_registry().get("glm-4.6v")
    provider, clock = _clocked_provider(monkeypatch, client, settings={
        "runner": {"per_model_min_interval_seconds": {model.id: 6.1}},
    })
    # 其他模型默认0，不会继承限频，也不会消耗该模型的时间配额。
    assert provider.recognize(model, DATA_URL, SYS, USER, GEN).ok
    assert provider.recognize(load_registry().get("deepseek-flash"), DATA_URL, SYS, USER, GEN).ok
    assert clock.sleeps == []
    assert provider.recognize(model, DATA_URL, SYS, USER, GEN).ok
    assert clock.sleeps == pytest.approx([6.1])


def test_openai_compat_retry_also_obeys_min_interval(monkeypatch):
    client = FakeClient([_StatusErr(429), _resp("ok")])
    model = load_registry().get("glm-4.6v")
    provider, clock = _clocked_provider(monkeypatch, client, settings={
        "retry": {"backoff_base_seconds": 2},
        "runner": {"per_model_min_interval_seconds": {model.id: 6.1}},
    })
    result = provider.recognize(model, DATA_URL, SYS, USER, GEN)
    assert result.ok and result.attempts == 2
    assert clock.sleeps == pytest.approx([2.0, 4.1])
    assert clock.now == pytest.approx(6.1)


def test_openai_compat_min_interval_cannot_be_bypassed_by_threads():
    client = FakeClient([_resp("ok") for _ in range(4)])
    model = load_registry().get("glm-4.6v")
    provider = _provider(client, settings={
        "runner": {"per_model_min_interval_seconds": {model.id: 0.02}},
    }, sleeper=time.sleep)
    starts = []
    original_create = client.chat.completions.create

    def record_start(**kwargs):
        starts.append(time.monotonic())
        return original_create(**kwargs)

    client.chat.completions.create = record_start
    barrier = threading.Barrier(4)

    def recognize():
        barrier.wait(timeout=3)
        return provider.recognize(model, DATA_URL, SYS, USER, GEN)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: recognize(), range(4)))
    assert all(result.ok for result in results)
    assert len(starts) == 4
    assert all(right - left >= 0.018 for left, right in zip(starts, starts[1:]))


@pytest.mark.parametrize("interval", [-1, float("inf"), float("nan")])
def test_openai_compat_rejects_invalid_min_interval(interval):
    with pytest.raises(ValueError, match="per_model_min_interval_seconds"):
        _provider(FakeClient([]), settings={
            "runner": {"per_model_min_interval_seconds": {"test-model": interval}},
        })


@pytest.mark.parametrize("header,expected_wait", [("5", 5.0), ("0.5", 2.0), ("invalid", 2.0), ("120", 120.0)])
def test_openai_compat_retry_after_seconds_respects_backoff(monkeypatch, header, expected_wait):
    error = _StatusErr(429)
    error.response = SimpleNamespace(headers={"Retry-After": header})
    client = FakeClient([error, _resp("ok")])
    provider, clock = _clocked_provider(monkeypatch, client, settings={
        "retry": {"backoff_base_seconds": 2},
    })
    result = provider.recognize(load_registry().get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert result.ok and result.attempts == 2
    assert clock.sleeps == pytest.approx([expected_wait])


def test_openai_compat_retry_after_http_date(monkeypatch):
    error = _StatusErr(429)
    client = FakeClient([error, _resp("ok")])
    provider, clock = _clocked_provider(monkeypatch, client)
    retry_at = datetime.fromtimestamp(clock.wall_time() + 12, tz=timezone.utc)
    error.response = SimpleNamespace(headers={"Retry-After": format_datetime(retry_at, usegmt=True)})
    result = provider.recognize(load_registry().get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert result.ok
    assert clock.sleeps == pytest.approx([12.0])


def test_openai_compat_retry_after_header_is_case_insensitive(monkeypatch):
    error = _StatusErr(503)
    error.headers = {"RETRY-AFTER": "4"}
    provider, clock = _clocked_provider(monkeypatch, FakeClient([error, _resp("ok")]))
    assert provider.recognize(load_registry().get("glm-4.6v"), DATA_URL, SYS, USER, GEN).ok
    assert clock.sleeps == pytest.approx([4.0])


@pytest.mark.parametrize("use_date", [False, True])
def test_openai_compat_long_retry_after_defers_instead_of_retrying_early(monkeypatch, use_date):
    error = _StatusErr(429)
    client = FakeClient([error, _resp("must not retry early")])
    provider, clock = _clocked_provider(monkeypatch, client)
    value = "1200"
    if use_date:
        retry_at = datetime.fromtimestamp(clock.wall_time() + 1200, tz=timezone.utc)
        value = format_datetime(retry_at, usegmt=True)
    error.response = SimpleNamespace(headers={"Retry-After": value})
    result = provider.recognize(load_registry().get("glm-4.6v"), DATA_URL, SYS, USER, GEN)
    assert not result.ok and result.error_type == "rate_limit"
    assert result.http_status == 429 and result.attempts == 1
    assert "retry deferred" in result.error_message
    assert "1200" in result.error_message
    assert clock.sleeps == []
    assert len(client.chat.completions.calls) == 1


def test_provider_result_defaults():
    r = ProviderResult(ok=False)
    assert r.raw_text == ""
    assert r.usage == {}
    assert r.reasoning_text == ""
    assert r.output_issue is None
