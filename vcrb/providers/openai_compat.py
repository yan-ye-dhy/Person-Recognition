"""OpenAI 兼容适配器 —— DeepSeek / Kimi(Moonshot) / GLM(智谱) / Qwen(百炼) 共用。

四家均提供 OpenAI Chat Completions 兼容接口（2026-11 官方文档核实），
差异通过 models.yaml 的 params 字段注入：
  pass_temperature: false → 不传 temperature（Kimi k2.5/k2.6 指定采样参数会报错）
  extra_body: {...}       → 厂商私有参数（如 Qwen enable_thinking=false）

公平性约束：
- 不传 tools / 不启用联网搜索（Qwen compatible-mode 的搜索默认关闭，如显式
  支持会再传 enable_search=false；本项目不传任何工具参数）；
- 记录 served_model（response.model）供 run 元数据核对实际服务版本。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..config import require_api_key
from ..models import ModelSpec, ProviderSpec
from .base import GenerationParams, ProviderResult, VisionProvider
from .errors import classify_exception, error_summary


class OpenAICompatProvider(VisionProvider):
    name = "openai_compat"

    def __init__(
        self,
        provider_spec: ProviderSpec,
        *,
        settings: dict[str, Any] | None = None,
        client_factory: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        """client_factory 供测试注入假客户端；sleeper 供测试免除退避等待。"""
        self.spec = provider_spec
        retry = (settings or {}).get("retry", {})
        self.max_attempts = int(retry.get("max_attempts", 4))
        self.backoff_base = float(retry.get("backoff_base_seconds", 2.0))
        self.timeout = float(retry.get("timeout_seconds", 120))
        self._client: Any = None
        self._client_factory = client_factory
        self._sleeper = sleeper if sleeper is not None else time.sleep

    # ---- client ----
    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI  # 延迟导入，测试无需真实 SDK 网络

                api_key = require_api_key(self.spec.api_key_env)
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self.spec.base_url,
                    timeout=self.timeout,
                    max_retries=0,  # 重试策略由本类统一控制
                )
        return self._client

    # ---- 请求构造（独立成函数便于单测断言公平性） ----
    @staticmethod
    def build_messages(
        image_data_url: str, system_prompt: str, user_prompt: str
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def build_request_kwargs(
        self,
        model: ModelSpec,
        messages: list[dict[str, Any]],
        gen: GenerationParams,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "max_tokens": gen.max_tokens,
        }
        if model.passes_temperature:
            kwargs["temperature"] = gen.temperature
        extra_body = model.extra_body
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    # ---- 主流程 ----
    def recognize(
        self,
        model: ModelSpec,
        image_data_url: str,
        system_prompt: str,
        user_prompt: str,
        gen: GenerationParams,
    ) -> ProviderResult:
        messages = self.build_messages(image_data_url, system_prompt, user_prompt)
        kwargs = self.build_request_kwargs(model, messages, gen)
        client = self._get_client()

        attempts = 0
        total_latency = 0
        while attempts < self.max_attempts:
            attempts += 1
            start = time.monotonic()
            try:
                resp = client.chat.completions.create(**kwargs)
                total_latency += int((time.monotonic() - start) * 1000)
                return self._parse_response(resp, attempts, total_latency)
            except Exception as exc:  # noqa: BLE001 - 统一分类
                total_latency += int((time.monotonic() - start) * 1000)
                error_type, retryable = classify_exception(exc)
                if retryable and attempts < self.max_attempts:
                    self._sleeper(self.backoff_base * (2 ** (attempts - 1)))
                    continue
                status = getattr(exc, "status_code", None)
                return ProviderResult(
                    ok=False,
                    latency_ms=total_latency,
                    attempts=attempts,
                    error_type=error_type,
                    error_message=error_summary(exc),
                    http_status=status if isinstance(status, int) else None,
                )
        # 理论不可达（循环内必 return），防御性兜底
        return ProviderResult(
            ok=False,
            latency_ms=total_latency,
            attempts=attempts,
            error_type="internal_error",
            error_message="retry loop exhausted",
        )

    @staticmethod
    def _parse_response(resp: Any, attempts: int, latency_ms: int) -> ProviderResult:
        choice = resp.choices[0]
        message = choice.message
        content = message.content if message is not None else None
        # 个别思考型模型把正文放在 reasoning_content 之后；content 为 None 时尝试拼接
        if content is None and message is not None:
            reasoning = getattr(message, "reasoning_content", None)
            content = reasoning if isinstance(reasoning, str) else ""
        usage_obj = getattr(resp, "usage", None)
        usage: dict[str, Any] = {}
        if usage_obj is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage_obj, key, None)
                if value is not None:
                    usage[key] = value
        return ProviderResult(
            ok=True,
            raw_text=content or "",
            finish_reason=getattr(choice, "finish_reason", None),
            served_model=getattr(resp, "model", None),
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
            http_status=200,
        )
