"""DeepSeek Flash's original-provider native web/image search, thinking off.

Official wire contract: https://api-docs.deepseek.com/guides/web_tools/
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from .native_search_common import NativeSearchProvider, https_endpoint, validate_image_data_url
from .openai_compat import _safe_json


def build_deepseek_payload(chat_kwargs: dict) -> dict:
    messages = deepcopy(chat_kwargs["messages"])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    validate_image_data_url(block.get("image_url", {}).get("url"))
    extra = chat_kwargs.get("extra_body") or {}
    if extra.get("thinking", {"type": "disabled"}) != {"type": "disabled"}:
        raise ValueError("DeepSeek search profile must keep thinking disabled")
    limit = chat_kwargs.get("max_tokens", chat_kwargs.get("max_completion_tokens"))
    if type(limit) is not int or limit < 1:
        raise ValueError("A positive output-token limit is required")
    payload = {
        "model": chat_kwargs["model"], "messages": messages, "max_tokens": limit,
        "thinking": {"type": "disabled"}, "stream": False,
        "tools": [{"type": "deepseek_web_search"}],
    }
    # DeepSeek explicitly forbids forcing this tool; required is merely auto.
    # Omit tool_choice so the service can decide whether to retrieve.
    if "temperature" in chat_kwargs:
        payload["temperature"] = chat_kwargs["temperature"]
    return payload


def completion_from_deepseek(data: dict):
    if not isinstance(data, dict):
        raise ValueError("DeepSeek response must be an object")
    failure = data.get("error")
    choices = data.get("choices") or []
    if not choices and not failure:
        raise ValueError("DeepSeek response lacks choices")
    choice = choices[0] if choices else {}
    if not isinstance(choice, dict):
        raise ValueError("DeepSeek response choice must be an object")
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("DeepSeek response message must be an object")
    finish = choice.get("finish_reason")
    error = None
    if failure:
        error = {"error_type": "server_error", "native_error": _safe_json(failure)}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=message.get("content"), reasoning_content=message.get("reasoning_content")),
            finish_reason=finish)],
        model=data.get("model"), usage=_safe_json(data.get("usage") or {}),
        _native_response=_safe_json(data), _native_error=error,
    )


def normalize_deepseek_search(data: dict) -> dict:
    from .search_evidence import normalize_search_evidence
    return normalize_search_evidence("deepseek", data)


class DeepSeekSearchProvider(NativeSearchProvider):
    name = "deepseek_native_search"
    evidence_vendor = "deepseek"

    def __init__(self, provider_spec, **kwargs):
        super().__init__(provider_spec, **kwargs)
        self.endpoint = https_endpoint(provider_spec.base_url, "/chat/completions")

    @property
    def transport_profile(self) -> dict:
        return {"api_mode": self.name, "endpoint": self.endpoint, "stream": False,
                "tools": [{"type": "deepseek_web_search"}], "thinking": {"type": "disabled"},
                "tool_choice": "auto", "search_forcing_supported": False}

    native_payload = staticmethod(build_deepseek_payload)
    completion_from_native = staticmethod(completion_from_deepseek)
