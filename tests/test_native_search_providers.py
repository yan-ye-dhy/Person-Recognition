"""Offline original-provider search protocols and one-HTTP accounting checks."""
import base64
from dataclasses import replace
import json

import httpx
import pytest

from tests.conftest import make_image_bytes
from vcrb.models import ProviderSpec, load_registry
from vcrb.providers.base import GenerationParams
from vcrb.providers.deepseek_search import DeepSeekSearchProvider, build_deepseek_payload
from vcrb.providers.moonshot_responses import MoonshotResponsesProvider, build_moonshot_payload


PROVIDERS = ["deepseek", "moonshot"]


def model_for(vendor):
    if vendor == "deepseek":
        return replace(load_registry().get("deepseek-flash"), params={
            "api_mode": "deepseek_native_search", "extra_body": {"thinking": {"type": "disabled"}}})
    return replace(load_registry().get("kimi-k3.1-pro"), params={
        "api_mode": "moonshot_responses", "pass_temperature": False,
        "token_limit_parameter": "max_completion_tokens", "extra_body": {"reasoning_effort": "low"}})


def response_for(vendor, *, sources=True):
    text = '{"persons":[{"character":"角色甲","bbox":[0.1,0.2,0.8,0.9]}]}'
    url = "https://example.test/characters"
    if vendor == "deepseek":
        calls = [{"index": 0, "calls": [{"id": "search-1", "type": "deepseek_web_search",
                 "kind": "search", "status": "completed", "action": {"type": "search", "queries": ["作品 角色资料"]},
                 "sources": [{"url": url, "title": "作品资料"}] if sources else []}]}]
        return {"model": "deepseek-served", "choices": [{"finish_reason": "stop", "message": {
            "content": text, "server_tool_calls": calls, "reasoning_content": "独立推理审计"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37,
                      "server_tool_use": {"tool_search": 2, "tool_image_search": 0, "tool_fetch": 0,
                                          "web_search_requests": 1}}}
    return {"id": "response-1", "model": "kimi-served", "status": "completed", "output": [
        {"type": "web_search_call", "id": "search-1", "status": "completed", "action": {
            "type": "search", "queries": ["作品 角色资料"],
            "sources": [{"url": url, "title": "作品资料"}] if sources else []}},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "独立推理审计"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    ], "usage": {"input_tokens": 30, "output_tokens": 7, "total_tokens": 37,
                 "input_tokens_details": {"cached_tokens": 5}}}


def make_provider(vendor, monkeypatch, handler, *, attempts=1):
    monkeypatch.setattr("vcrb.providers.native_search_common.require_api_key", lambda _: "sk-unit-secret0123456789")
    cls, base = ((DeepSeekSearchProvider, "https://api.deepseek.com") if vendor == "deepseek" else
                 (MoonshotResponsesProvider, "https://api.moonshot.cn/v1"))
    return cls(ProviderSpec(vendor, base, "TEST_KEY"),
               settings={"retry": {"max_attempts": attempts, "timeout_seconds": 5}}, sleeper=lambda _: None,
               http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def call(provider, vendor, *, observer=None):
    image = make_image_bytes((64, 40), (50, 70, 90))
    encoded = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    result = provider.recognize(model_for(vendor), encoded, "原始系统提示：只输出JSON。", "作品：示例作品；识别主要人物。",
                                GenerationParams(.1, 4096), attempt_observer=observer)
    return result, image, encoded


@pytest.mark.parametrize("vendor", PROVIDERS)
def test_native_search_preserves_exact_image_and_prompts_and_returns_actual_sources(vendor, monkeypatch):
    received = []
    def handler(request):
        received.append((request, json.loads(request.content)))
        return httpx.Response(200, json=response_for(vendor))
    provider = make_provider(vendor, monkeypatch, handler)
    result, image, encoded = call(provider, vendor)
    request, body = received[0]
    assert result.ok and result.raw_text.startswith('{"persons":')
    assert result.reasoning_text == "独立推理审计" and "独立推理" not in result.raw_text
    assert result.usage["prompt_tokens"] == 30 and result.usage["completion_tokens"] == 7
    assert result.search_metadata["retrieved"] is True and result.search_metadata["attempted"] is True
    assert result.search_metadata["sources"] == [{"url": "https://example.test/characters", "title": "作品资料"}]
    assert result.search_metadata["queries"] == ["作品 角色资料"]
    assert body["stream"] is False and len(received) == 1
    assert request.headers["Authorization"] == "Bearer sk-unit-secret0123456789"
    if vendor == "deepseek":
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert body["messages"][0] == {"role": "system", "content": "原始系统提示：只输出JSON。"}
        blocks = body["messages"][1]["content"]
        assert blocks[0]["image_url"]["url"] == encoded
        assert blocks[1]["text"] == "作品：示例作品；识别主要人物。"
        assert body["thinking"] == {"type": "disabled"}
        assert body["tools"] == [{"type": "deepseek_web_search"}]
        assert "tool_choice" not in body and body["max_tokens"] == 4096 and body["temperature"] == .1
        assert result.search_metadata["billing"]["units"]["search"] == 2  # queries, not result blocks
    else:
        assert str(request.url) == "https://api.moonshot.cn/v1/responses"
        assert body["instructions"] == "原始系统提示：只输出JSON。"
        blocks = body["input"][0]["content"]
        assert blocks[0] == {"type": "input_image", "image_url": encoded}
        assert base64.b64decode(blocks[0]["image_url"].split(",")[1]) == image
        assert blocks[1] == {"type": "input_text", "text": "作品：示例作品；识别主要人物。"}
        assert body["reasoning"] == {"effort": "low"} and "temperature" not in body
        assert body["tools"] == [{"type": "web_search", "search_content_types": ["text"]}]
        assert body["include"] == ["web_search_call.action.sources"]
        assert body["tool_choice"] == "auto" and body["max_output_tokens"] == 4096
        assert result.usage["input_tokens_details"]["cached_tokens"] == 5
        assert result.search_metadata["billing"]["units"]["search"] == 1
    assert "sk-unit" not in json.dumps(result.search_metadata)
    provider.close()


@pytest.mark.parametrize("vendor", PROVIDERS)
def test_search_without_returned_sources_is_not_proof_even_with_billable_counts(vendor, monkeypatch):
    provider = make_provider(vendor, monkeypatch, lambda _: httpx.Response(200, json=response_for(vendor, sources=False)))
    result, _, _ = call(provider, vendor)
    assert result.ok and result.search_metadata["retrieved"] is False
    assert result.search_metadata["attempted"] is True and result.search_metadata["sources"] == []
    assert result.search_metadata["billing"]["known_units"] is True


@pytest.mark.parametrize("vendor", PROVIDERS)
@pytest.mark.parametrize("failure,kind", [(httpx.ReadTimeout("timeout"), "timeout"), (httpx.ConnectError("network"), "network")])
def test_retry_has_one_observed_boundary_per_actual_http(vendor, failure, kind, monkeypatch):
    received, events = [], []
    def handler(request):
        received.append(request)
        if len(received) == 1:
            raise failure
        return httpx.Response(200, json=response_for(vendor))
    class Observer:
        def before_attempt(self, model, image, prompt, number):
            events.append(("before", number)); return number
        def after_attempt(self, receipt, model, *, result, error):
            events.append(("after", receipt))
            if receipt == 1:
                from vcrb.providers.errors import classify_exception
                assert classify_exception(error) == (kind, True)
            else:
                assert result.ok and error is None
    provider = make_provider(vendor, monkeypatch, handler, attempts=2)
    result, _, _ = call(provider, vendor, observer=Observer())
    assert result.ok and result.attempts == len(received) == 2
    assert events == [("before", 1), ("after", 1), ("before", 2), ("after", 2)]


@pytest.mark.parametrize("vendor", PROVIDERS)
def test_http_200_generation_failure_keeps_paid_usage_and_does_not_hide_retry(vendor, monkeypatch):
    response = response_for(vendor)
    response["error"] = {"code": "server_error", "message": "failure sk-unit-secret0123456789"}
    if vendor == "deepseek":
        response["choices"] = []
    else:
        response["status"] = "failed"
    received, observed = [], []
    def handler(request):
        received.append(request); return httpx.Response(200, json=response)
    class Observer:
        def before_attempt(self, *args): return 1
        def after_attempt(self, receipt, model, *, result, error):
            observed.append(result)
            assert error is None and not result.ok and result.usage["prompt_tokens"] == 30
    provider = make_provider(vendor, monkeypatch, handler, attempts=3)
    result, _, _ = call(provider, vendor, observer=Observer())
    assert not result.ok and result.error_type == "server_error" and result.http_status == 200
    assert result.usage["completion_tokens"] == 7 and result.attempts == len(received) == len(observed) == 1
    assert "sk-unit-secret" not in result.error_message
    assert "sk-unit-secret" not in json.dumps(result.search_metadata)


@pytest.mark.parametrize("vendor", PROVIDERS)
@pytest.mark.parametrize("status,kind", [(400, "bad_request"), (401, "auth_error"), (404, "model_not_found")])
def test_http_errors_classify_without_retrying_or_exposing_keys(vendor, status, kind, monkeypatch):
    received = []
    def handler(request):
        received.append(request)
        return httpx.Response(status, json={"error": {"message": "bad sk-unit-secret0123456789"}})
    provider = make_provider(vendor, monkeypatch, handler, attempts=3)
    result, _, _ = call(provider, vendor)
    assert not result.ok and result.error_type == kind and result.http_status == status
    assert len(received) == result.attempts == 1 and "sk-unit-secret" not in result.error_message


def test_moonshot_truncation_remains_final_partial_output_without_continuation(monkeypatch):
    response = response_for("moonshot")
    response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    received = []
    def handler(request):
        received.append(request); return httpx.Response(200, json=response)
    provider = make_provider("moonshot", monkeypatch, handler, attempts=3)
    result, _, _ = call(provider, "moonshot")
    assert result.ok and result.output_issue == "truncated_output" and result.finish_reason == "length"
    assert len(received) == 1 and result.search_metadata["retrieved"] is True


@pytest.mark.parametrize("status", ["in_progress", "queued", "cancelled", None])
def test_moonshot_never_polls_or_continues_nonterminal_responses(status, monkeypatch):
    response = response_for("moonshot"); response["status"] = status
    received = []
    def handler(request):
        received.append(request); return httpx.Response(200, json=response)
    provider = make_provider("moonshot", monkeypatch, handler, attempts=3)
    result, _, _ = call(provider, "moonshot")
    assert not result.ok and len(received) == 1 and result.usage["prompt_tokens"] == 30


@pytest.mark.parametrize("vendor", PROVIDERS)
def test_invalid_or_public_image_rejected_before_http_and_thinking_is_not_silently_enabled(vendor, monkeypatch):
    received = []
    provider = make_provider(vendor, monkeypatch, lambda req: received.append(req))
    with pytest.raises(ValueError, match="image data URL"):
        provider.recognize(model_for(vendor), "https://example.test/image.jpg", "sys", "user", GenerationParams())
    model = model_for(vendor)
    params = dict(model.params)
    params["extra_body"] = ({"thinking": {"type": "enabled"}} if vendor == "deepseek" else {"reasoning_effort": "high"})
    with pytest.raises(ValueError, match="thinking disabled|baseline low"):
        provider.recognize(replace(model, params=params), "data:image/jpeg;base64,AAAA", "sys", "user", GenerationParams())
    assert received == []


@pytest.mark.parametrize("cls", [DeepSeekSearchProvider, MoonshotResponsesProvider])
@pytest.mark.parametrize("base", ["http://example.test", "https://name:secret@example.test", "https://example.test?key=secret"])
def test_native_endpoint_cannot_leak_credentials_or_downgrade_https(cls, base):
    with pytest.raises(ValueError):
        cls(ProviderSpec("test", base, "KEY"))


def test_moonshot_usage_mismatch_is_retained_for_budget_validator(monkeypatch):
    response = response_for("moonshot"); response["usage"]["total_tokens"] = 999
    provider = make_provider("moonshot", monkeypatch, lambda _: httpx.Response(200, json=response))
    result, _, _ = call(provider, "moonshot")
    assert result.usage["total_tokens"] == 999


def test_deepseek_failed_tool_sources_are_not_claimed_as_retrieved(monkeypatch):
    response = response_for("deepseek")
    response["choices"][0]["message"]["server_tool_calls"][0]["calls"][0]["status"] = "failed"
    provider = make_provider("deepseek", monkeypatch, lambda _: httpx.Response(200, json=response))
    result, _, _ = call(provider, "deepseek")
    assert result.ok and result.search_metadata["retrieved"] is False
