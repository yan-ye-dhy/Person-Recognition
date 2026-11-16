"""Official account schema checks using mock HTTP only; no paid calls."""
import json

import httpx
import pytest

from vcrb.cost_probe_balance import public_balance_status, snapshot_balance
from vcrb.models import ModelRegistry, ModelSpec, ProviderSpec


STAMP = "2026-11-16T12:00:00+08:00"
SECRET = "private-test-credential"


def registry(provider="deepseek", base_url=None):
    origins = {"deepseek": "https://api.deepseek.com", "moonshot": "https://api.moonshot.cn/v1",
               "zhipu": "https://open.bigmodel.cn/api/paas/v4"}
    spec = ProviderSpec(provider, base_url or origins[provider], "TEST_EXISTING_KEY")
    model = ModelSpec("test-model", provider, provider, "test", "test", "flash", "doc", True)
    return ModelRegistry({provider: spec}, [model])


def invoke(monkeypatch, body, *, provider="deepseek", status=200, headers=None, base_url=None):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text=body, headers=headers)
    monkeypatch.setattr("vcrb.cost_probe_balance.get_api_key", lambda _: SECRET)
    result = snapshot_balance("test-model", registry=registry(provider, base_url), captured_at=STAMP,
                              http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return result, calls


def deepseek_payload(**overrides):
    payload = {"is_available": True, "balance_infos": [
        {"currency": "CNY", "total_balance": "110.001234567890123456789",
         "granted_balance": "10.000000000000000000001", "topped_up_balance": "100.001234567890123456788"},
        {"currency": "USD", "total_balance": "2.10", "granted_balance": "0.10", "topped_up_balance": "2.00"}]}
    payload.update(overrides)
    return json.dumps(payload)


def kimi_payload():
    return ('{"code":0,"status":true,"data":{"available_balance":46.588930000000001,'
            '"voucher_balance":46.588930000000001,"cash_balance":-3.00001000000000001},"scode":"0x0"}')


def test_deepseek_uses_original_official_get_preserves_currency_and_precision(monkeypatch, capsys):
    result, calls = invoke(monkeypatch, deepseek_payload())
    assert result["status"] == "collected" and result["kind"] == "account_balance_snapshot_not_bill"
    assert len(calls) == 1 and calls[0].method == "GET"
    assert str(calls[0].url) == "https://api.deepseek.com/user/balance"
    assert calls[0].headers["Authorization"] == f"Bearer {SECRET}"
    assert result["balances"][0]["total_balance"] == "110.001234567890123456789"
    assert result["balances"][1]["currency"] == "USD"
    assert result["captured_at"] == "2026-11-16T04:00:00+00:00"
    assert result["private_raw_balance"]["balance_infos"][1]["total_balance"] == "2.10"
    assert "fee" not in result and "cost" not in result
    assert SECRET not in json.dumps(result)
    assert capsys.readouterr().out == ""


def test_kimi_success_envelope_and_negative_cash_do_not_recompute_available(monkeypatch):
    result, calls = invoke(monkeypatch, kimi_payload(), provider="moonshot")
    assert str(calls[0].url) == "https://api.moonshot.cn/v1/users/me/balance"
    assert result["balances"] == [{"currency": "CNY", "available_balance": "46.588930000000001",
                                   "voucher_balance": "46.588930000000001", "cash_balance": "-3.00001000000000001"}]
    assert result["private_raw_balance"]["data"]["cash_balance"] == "-3.00001000000000001"
    json.dumps(result)  # Decimal values must be serializable without float conversion.


@pytest.mark.parametrize("code,status", [(123, True), (True, True), (0, False), ("0", True), (0, "true")])
def test_kimi_business_envelope_failure_is_not_collected(monkeypatch, code, status):
    payload = json.loads(kimi_payload()); payload.update(code=code, status=status)
    result, calls = invoke(monkeypatch, json.dumps(payload), provider="moonshot")
    assert len(calls) == 1 and result["status"] == "unavailable"
    assert result["reason"] == "invalid_balance_schema" and "balances" not in result


@pytest.mark.parametrize("payload", [
    {}, {"is_available": True}, {"is_available": True, "balance_infos": []},
    {"is_available": "true", "balance_infos": [{}]},
    {"is_available": True, "balance_infos": [{"currency": "EUR", "total_balance": "1"}]},
    {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": True,
                                              "granted_balance": "0", "topped_up_balance": "1"}]},
    {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "NaN",
                                              "granted_balance": "0", "topped_up_balance": "1"}]},
])
def test_missing_or_invalid_financial_schema_never_becomes_zero_balance(monkeypatch, payload):
    result, _ = invoke(monkeypatch, json.dumps(payload))
    assert result["status"] == "unavailable" and "balances" not in result


@pytest.mark.parametrize("payload", ['not JSON', '{"is_available":true,"is_available":false}', '{"x":NaN}'])
def test_malformed_duplicate_or_nonfinite_json_is_unavailable(monkeypatch, payload):
    result, _ = invoke(monkeypatch, payload)
    assert result["reason"] == "invalid_json"


@pytest.mark.parametrize("status", [301, 302, 401, 403, 429, 500])
def test_http_failure_does_not_retry_follow_redirects_or_expose_response(monkeypatch, status):
    result, calls = invoke(monkeypatch, SECRET, status=status, headers={"Location": "https://example.org/leak"})
    assert len(calls) == 1 and result["http_status"] == status and result["status"] == "unavailable"
    assert "private_raw_balance" not in result and SECRET not in json.dumps(result)


@pytest.mark.parametrize("error,reason", [(httpx.ReadTimeout("private-test-credential"), "transport_timeout"),
                                         (httpx.ConnectError("private-test-credential"), "transport_error")])
def test_transport_failure_is_single_attempt_and_exception_text_is_never_forwarded(monkeypatch, error, reason):
    calls = []
    def handler(request):
        calls.append(request); raise error
    monkeypatch.setattr("vcrb.cost_probe_balance.get_api_key", lambda _: SECRET)
    result = snapshot_balance("test-model", registry=registry(),
                              http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(calls) == 1 and result["reason"] == reason
    assert SECRET not in json.dumps(result)


@pytest.mark.parametrize("provider,origin,reason", [
    ("zhipu", None, "unsupported_provider"),
    ("deepseek", "https://example.com", "unsupported_provider_origin"),
    ("deepseek", "http://api.deepseek.com", "unsupported_provider_origin"),
    ("deepseek", "https://api.deepseek.com:444", "unsupported_provider_origin"),
    ("deepseek", "https://user:password@api.deepseek.com", "unsupported_provider_origin"),
    ("moonshot", "https://api.moonshot.ai/v1", "unsupported_provider_origin"),
])
def test_unsupported_vendor_or_host_is_rejected_before_accessing_key_or_http(monkeypatch, provider, origin, reason):
    monkeypatch.setattr("vcrb.cost_probe_balance.get_api_key", lambda _: pytest.fail("must not read key"))
    result = snapshot_balance("test-model", registry=registry(provider, origin),
                              http_client_factory=lambda: pytest.fail("must not construct HTTP client"))
    assert result["reason"] == reason and "balances" not in result


def test_missing_credentials_does_not_attempt_http(monkeypatch):
    monkeypatch.setattr("vcrb.cost_probe_balance.get_api_key", lambda _: None)
    result = snapshot_balance("test-model", registry=registry(),
                              http_client_factory=lambda: pytest.fail("must not perform HTTP"))
    assert result["reason"] == "credentials_unavailable"


def test_unregistered_model_is_unavailable():
    assert snapshot_balance("missing-model", registry=registry())["reason"] == "configuration_unavailable"


def test_public_projection_has_no_balance_raw_response_docs_or_arbitrary_error(monkeypatch):
    private, _ = invoke(monkeypatch, deepseek_payload())
    public = public_balance_status(private)
    assert public == {"model_id": "test-model", "provider": "deepseek", "captured_at": "2026-11-16T04:00:00+00:00",
                      "status": "collected", "kind": "balance_collection_status_not_bill"}
    assert public_balance_status({"status": "unavailable", "reason": {"secret": SECRET}})["reason"] == "unavailable"


def test_invalid_naive_timestamp_fails_before_network():
    with pytest.raises(ValueError, match="timezone"):
        snapshot_balance("test-model", captured_at="2026-11-16T12:00:00", registry=registry(),
                         http_client_factory=lambda: pytest.fail("must not perform HTTP"))
