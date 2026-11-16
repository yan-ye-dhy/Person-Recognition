"""Read-only account balance snapshots for private cost-probe reconciliation.

These are account snapshots, not bills or per-request charges. Callers must save
the returned snapshot privately: it contains financial balances and the raw
successful response. Use ``public_balance_status`` for shareable metadata.
There are no model calls, automatic retries, FX conversions, or settlement-delay
assumptions here. A missing snapshot must never be interpreted as zero cost.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from .config import ConfigError, get_api_key
from .models import ModelRegistry, load_registry


BALANCE_TIMEOUT_SECONDS = 15.0
_ENDPOINTS = {
    "deepseek": ("api.deepseek.com", "https://api.deepseek.com/user/balance"),
    "moonshot": ("api.moonshot.cn", "https://api.moonshot.cn/v1/users/me/balance"),
}
_DOCS = {
    "deepseek": "https://api-docs.deepseek.com/api/get-user-balance/",
    "moonshot": "https://platform.kimi.com/docs/api/balance",
}


def _timestamp(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("captured_at must be an ISO timestamp with a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _money(value: Any) -> str:
    # JSON floating point numbers are decoded directly as Decimal, avoiding a
    # binary-float intermediate. Do not round to cents or convert currencies.
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("invalid balance number")
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError("invalid balance number") from None
    if not amount.is_finite():
        raise ValueError("nonfinite balance number")
    return format(amount, "f")


def _private_json(value: Any) -> Any:
    """Keep the raw structure JSON serializable without losing decimal digits."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _private_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_private_json(item) for item in value]
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate balance response key")
        result[key] = value
    return result


def _decode_balance(provider: str, payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("balance response must be an object")
    if provider == "deepseek":
        if not isinstance(payload.get("is_available"), bool):
            raise ValueError("missing DeepSeek availability status")
        items = payload.get("balance_infos")
        if not isinstance(items, list) or not items:
            raise ValueError("missing DeepSeek balances")
        balances = []
        for item in items:
            if not isinstance(item, dict) or item.get("currency") not in {"CNY", "USD"}:
                raise ValueError("invalid DeepSeek currency")
            balances.append({"currency": item["currency"], **{
                field: _money(item.get(field))
                for field in ("total_balance", "granted_balance", "topped_up_balance")
            }})
        if len({item["currency"] for item in balances}) != len(balances):
            raise ValueError("duplicate DeepSeek currency")
        return balances
    if provider == "moonshot":
        # The documented success envelope is code=0 AND status=true. In
        # particular, True must not pass as the integer zero/one status code.
        if type(payload.get("code")) is not int or payload["code"] != 0 or payload.get("status") is not True:
            raise ValueError("unsuccessful Kimi business status")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("missing Kimi balance data")
        result = {"currency": "CNY", **{
            field: _money(data.get(field))
            for field in ("available_balance", "voucher_balance", "cash_balance")
        }}
        if Decimal(result["voucher_balance"]) < 0:
            raise ValueError("negative Kimi voucher balance")
        # cash_balance may be negative; then available_balance need not equal
        # cash + voucher. The official available balance is retained verbatim.
        return [result]
    raise ValueError("unsupported balance provider")


def snapshot_balance(
    model_id: str,
    *,
    captured_at: str | datetime | None = None,
    registry: ModelRegistry | None = None,
    http_client_factory: Callable[[], httpx.Client] | None = None,
) -> dict[str, Any]:
    """Return a PRIVATE, JSON-safe account snapshot using existing credentials.

    Supported: original DeepSeek and mainland Kimi accounts. Unsupported
    vendors, missing credentials, HTTP/transport failures, and malformed schemas
    yield ``status='unavailable'`` with a bounded, non-sensitive reason code.
    Exactly one read-only GET is attempted; redirects are never followed.
    No files are written, credentials printed, or bill amounts inferred.
    ``captured_at`` may be supplied as a timezone-aware ISO string or datetime.
    """
    # Validate user-supplied timestamps before any network operation.
    supplied_time = _timestamp(captured_at) if captured_at is not None else None
    result: dict[str, Any] = {
        "schema_version": "1.0", "kind": "account_balance_snapshot_not_bill",
        "model_id": model_id, "provider": None, "status": "unavailable",
    }

    def finish(reason: str | None = None) -> dict[str, Any]:
        result["captured_at"] = supplied_time or _timestamp(None)
        if reason is not None:
            result["reason"] = reason
        return result

    try:
        registry = registry if registry is not None else load_registry()
        model = registry.get(model_id)
        spec = registry.provider_spec(model)
    except (ConfigError, KeyError):
        return finish("configuration_unavailable")
    provider = model.provider
    result["provider"] = provider
    if provider not in _ENDPOINTS:
        return finish("unsupported_provider")
    host, endpoint = _ENDPOINTS[provider]
    try:
        parsed = urlsplit(spec.base_url)
        valid_origin = (parsed.scheme == "https" and parsed.hostname == host
                        and parsed.port in (None, 443) and parsed.username is None
                        and parsed.password is None and not parsed.query and not parsed.fragment)
    except ValueError:
        valid_origin = False
    if not valid_origin:
        # Never send a configured provider's credential to an unverified host.
        return finish("unsupported_provider_origin")
    key = get_api_key(spec.api_key_env)
    if not key:
        return finish("credentials_unavailable")
    result["official_documentation"] = _DOCS[provider]
    try:
        with (http_client_factory() if http_client_factory else
              httpx.Client(timeout=BALANCE_TIMEOUT_SECONDS, follow_redirects=False)) as client:
            response = client.get(endpoint, headers={"Authorization": f"Bearer {key}"},
                                  timeout=BALANCE_TIMEOUT_SECONDS, follow_redirects=False)
    except httpx.TimeoutException:
        return finish("transport_timeout")
    except httpx.RequestError:
        return finish("transport_error")
    result["http_status"] = response.status_code
    if response.status_code != 200:
        return finish("http_redirect" if 300 <= response.status_code < 400 else "http_error")
    try:
        payload = json.loads(response.text, parse_float=Decimal, object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, UnicodeError):
        return finish("invalid_json")
    try:
        balances = _decode_balance(provider, payload)
    except (ValueError, TypeError, InvalidOperation):
        return finish("invalid_balance_schema")
    result.update(status="collected", balances=balances,
                  private_raw_balance=_private_json(payload))
    return finish()


def public_balance_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Share only collection status, never account amounts or raw responses."""
    status = "collected" if snapshot.get("status") == "collected" else "unavailable"
    public = {key: snapshot.get(key) for key in ("model_id", "provider", "captured_at")}
    public.update(status=status, kind="balance_collection_status_not_bill")
    if status == "unavailable":
        # Only reason codes this module itself emits are publishable. Never
        # forward a caller-supplied exception/body disguised as a reason.
        reason = snapshot.get("reason")
        allowed = {"configuration_unavailable", "unsupported_provider", "unsupported_provider_origin",
                   "credentials_unavailable", "transport_timeout", "transport_error", "http_redirect",
                   "http_error", "invalid_json", "invalid_balance_schema"}
        public["reason"] = reason if isinstance(reason, str) and reason in allowed else "unavailable"
    return public
