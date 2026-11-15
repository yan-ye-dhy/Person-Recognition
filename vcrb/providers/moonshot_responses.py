"""Original Moonshot Responses API with server-side web search and sources.

Official wire contract: https://platform.kimi.com/docs/api/responses.md
Kimi K3.1 cannot disable reasoning; the existing benchmark's low effort is kept.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from .native_search_common import NativeSearchProvider, https_endpoint, validate_image_data_url
from .openai_compat import _safe_json


def build_moonshot_payload(chat_kwargs: dict) -> dict:
    instructions = None
    inputs = []
    for message in chat_kwargs["messages"]:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if instructions is not None or not isinstance(content, str):
                raise ValueError("Expected one unchanged text system prompt")
            instructions = content
            continue
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported Responses message role")
        if isinstance(content, str):
            blocks = [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "image_url":
                    image = block.get("image_url") or {}
                    converted = {"type": "input_image", "image_url": validate_image_data_url(image.get("url"))}
                    if "detail" in image:
                        converted["detail"] = image["detail"]
                    blocks.append(converted)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "input_text", "text": block["text"]})
                else:
                    raise ValueError("Unsupported Responses content block")
        else:
            raise ValueError("Unsupported Responses message content")
        inputs.append({"role": role, "content": blocks})
    extra = chat_kwargs.get("extra_body") or {}
    if extra.get("reasoning_effort", "low") != "low":
        raise ValueError("Kimi search profile must retain baseline low reasoning effort")
    limit = chat_kwargs.get("max_completion_tokens", chat_kwargs.get("max_tokens"))
    if type(limit) is not int or limit < 1:
        raise ValueError("A positive output-token limit is required")
    payload = {"model": chat_kwargs["model"], "input": inputs,
               "max_output_tokens": limit, "reasoning": {"effort": "low"},
               "tools": [{"type": "web_search", "search_content_types": ["text"]}],
               "include": ["web_search_call.action.sources"], "tool_choice": "auto", "stream": False}
    if instructions is not None:
        payload["instructions"] = instructions
    # The K3.1 baseline intentionally omits unsupported sampling overrides.
    return payload


def _converted_usage(data: dict) -> dict:
    usage = deepcopy(data.get("usage") or {})
    if not isinstance(usage, dict):
        return {}
    for native, common in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = usage.get(native)
        if type(value) is int and value >= 0:
            usage[common] = value
    if "total_tokens" not in usage and all(type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens")):
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return _safe_json(usage)


def completion_from_moonshot(data: dict):
    if not isinstance(data, dict):
        raise ValueError("Moonshot response must be an object")
    output = data.get("output")
    if not isinstance(output, list) and not data.get("error"):
        raise ValueError("Moonshot response lacks output items")
    final, reasoning = [], []
    unexpected = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    final.append(block["text"])
        elif kind == "reasoning":
            for block in (item.get("content") or []) + (item.get("summary") or []):
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    reasoning.append(block["text"])
        elif kind in {"function_call", "custom_tool_call"}:
            unexpected.append(kind)
    status = data.get("status")
    reason = (data.get("incomplete_details") or {}).get("reason")
    finish = "length" if reason == "max_output_tokens" else ("stop" if status == "completed" else status)
    error = None
    if data.get("error") or status not in {"completed", "incomplete"}:
        error = {"error_type": "server_error", "status": status, "native_error": _safe_json(data.get("error"))}
    if unexpected:
        error = {"error_type": "internal_error", "message": "Unexpected client tool calls; no hidden continuation performed", "types": unexpected}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="\n".join(final), reasoning_content="\n".join(reasoning)),
                                 finish_reason=finish)],
        model=data.get("model"), usage=_converted_usage(data),
        _native_response=_safe_json(data), _native_error=error,
    )


def normalize_moonshot_search(data: dict) -> dict:
    from .search_evidence import normalize_search_evidence
    return normalize_search_evidence("moonshot", data)


class MoonshotResponsesProvider(NativeSearchProvider):
    name = "moonshot_responses"
    evidence_vendor = "moonshot"

    def __init__(self, provider_spec, **kwargs):
        super().__init__(provider_spec, **kwargs)
        self.endpoint = https_endpoint(provider_spec.base_url, "/responses")

    @property
    def transport_profile(self) -> dict:
        return {"api_mode": self.name, "endpoint": self.endpoint, "stream": False,
                "tools": [{"type": "web_search", "search_content_types": ["text"]}],
                "include": ["web_search_call.action.sources"], "tool_choice": "auto",
                "reasoning": {"effort": "low"}, "search_forcing_supported": False}

    native_payload = staticmethod(build_moonshot_payload)
    completion_from_native = staticmethod(completion_from_moonshot)
