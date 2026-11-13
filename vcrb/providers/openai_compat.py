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

import math
import threading
import time
from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator

from ..config import require_api_key
from ..models import ModelSpec, ProviderSpec
from .base import GenerationParams, ProviderResult, VisionProvider
from .errors import classify_exception, error_summary


class OpenAICompatProvider(VisionProvider):
    name = "openai_compat"
    supports_attempt_observer = True
    MAX_AUTOMATIC_RETRY_WAIT_SECONDS = 120.0

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
        runner = (settings or {}).get("runner", {}) or {}
        intervals = runner.get("per_model_min_interval_seconds", {}) or {}
        self._min_intervals: dict[str, float] = {}
        for model_id, seconds in intervals.items():
            interval = float(seconds)
            if not math.isfinite(interval) or interval < 0:
                raise ValueError("per_model_min_interval_seconds 必须为有限非负秒数")
            self._min_intervals[str(model_id)] = interval
        self._request_start_lock = threading.Lock()
        self._last_request_started: dict[str, float] = {}

    @contextmanager
    def _http_attempt_slot(self, model_id: str) -> Iterator[None]:
        """指定模型按实际 HTTP 尝试启动时间限频，重试与并发线程均不能绕过。

        只有设置了正间隔的模型使用锁；锁覆盖 HTTP 尝试，防止线程拿到许可后
        被挂起、恢复时与下一线程同时发请求。排队等待不计入 HTTP 响应延迟。
        """
        interval = self._min_intervals.get(model_id, 0.0)
        if interval <= 0:
            yield
            return
        with self._request_start_lock:
            previous = self._last_request_started.get(model_id)
            if previous is not None:
                remaining = interval - (time.monotonic() - previous)
                if remaining > 0:
                    self._sleeper(remaining)
            yield

    @staticmethod
    def _retry_after_seconds(exc: BaseException) -> float | None:
        """解析 HTTP Retry-After 秒数/日期；正文中的非标准字段不作为协议。"""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        if headers is None or not hasattr(headers, "items"):
            return None
        raw = next((value for key, value in headers.items() if str(key).lower() == "retry-after"), None)
        if raw is None:
            return None
        text = str(raw).strip()
        try:
            delay = float(text)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(text)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return None
        return max(0.0, delay) if math.isfinite(delay) else None

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
        token_limit_parameter = model.params.get("token_limit_parameter", "max_tokens")
        if token_limit_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError(
                "token_limit_parameter 只允许 max_tokens 或 max_completion_tokens"
            )
        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            token_limit_parameter: gen.max_tokens,
        }
        if model.passes_temperature:
            kwargs["temperature"] = gen.temperature
        extra_body = model.extra_body
        # SDK 会把 extra_body 合并到请求 JSON，不能让它覆盖预算或添加第二种预算字段。
        if any(key in extra_body for key in ("max_tokens", "max_completion_tokens")):
            raise ValueError(
                "extra_body 不得包含输出预算；请使用 token_limit_parameter 与 generation.max_tokens"
            )
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
        *,
        attempt_observer: Any = None,
    ) -> ProviderResult:
        """可选观察器在实际 HTTP 前预留、结束后结算；默认 v1 不启用。

        观察器失败直接向上抛出，不能被分类成可重试的 API 错误。SDK 自身
        的重试已关闭，所以每对 before/after 恰好对应一次客户端 HTTP 调用。
        """
        messages = self.build_messages(image_data_url, system_prompt, user_prompt)
        kwargs = self.build_request_kwargs(model, messages, gen)
        client = self._get_client()

        attempts = 0
        total_latency = 0
        while attempts < self.max_attempts:
            attempts += 1
            error: BaseException | None = None
            result = None
            with self._http_attempt_slot(model.id):
                receipt = (attempt_observer.before_attempt(model, image_data_url, user_prompt, attempts)
                           if attempt_observer is not None else None)
                start = time.monotonic()
                if self._min_intervals.get(model.id, 0.0) > 0:
                    # Budget admission can take time; rate limiting measures
                    # the actual HTTP start, after admission, under the slot lock.
                    self._last_request_started[model.id] = start
                try:
                    resp = client.chat.completions.create(**kwargs)
                except BaseException as exc:  # interrupt still retains the admitted attempt
                    error = exc
                total_latency += int((time.monotonic() - start) * 1000)
            if error is None:
                try:
                    result = self._parse_response(resp, attempts, total_latency)
                except BaseException as exc:
                    error = exc
            if attempt_observer is not None:
                attempt_observer.after_attempt(receipt, model, result=result, error=error)
            if error is None:
                return result
            if not isinstance(error, Exception):
                raise error
            else:
                error_type, retryable = classify_exception(error)
                deferred_note = ""
                if retryable and attempts < self.max_attempts:
                    retry_after = self._retry_after_seconds(error)
                    delay = max(self.backoff_base * (2 ** (attempts - 1)), retry_after or 0.0)
                    if delay <= self.MAX_AUTOMATIC_RETRY_WAIT_SECONDS:
                        self._sleeper(delay)
                        continue
                    # 不把长服务端等待截短后提前撞限流；交给后续断点重试。
                    deferred_note = (
                        f"; retry deferred: required wait {delay:.3f}s exceeds automatic "
                        f"retry wait limit {self.MAX_AUTOMATIC_RETRY_WAIT_SECONDS:.0f}s"
                    )
                status = getattr(error, "status_code", None)
                return ProviderResult(
                    ok=False,
                    latency_ms=total_latency,
                    attempts=attempts,
                    error_type=error_type,
                    error_message=error_summary(error) + deferred_note,
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
        # 推理中可能出现备选姓名/示例 JSON，不能代替模型最终提交的答案。
        reasoning = getattr(message, "reasoning_content", None) if message is not None else None
        reasoning_text = reasoning if isinstance(reasoning, str) else ""
        final_text = content if isinstance(content, str) else ""
        finish_reason = getattr(choice, "finish_reason", None)
        output_issue = None
        if finish_reason == "length":
            output_issue = "truncated_output"
        elif not final_text.strip():
            output_issue = "empty_final_content"
        usage_obj = getattr(resp, "usage", None)
        usage: dict[str, Any] = {}
        if usage_obj is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage_obj, key, None)
                if value is not None:
                    usage[key] = value
        return ProviderResult(
            ok=True,
            raw_text=final_text,
            finish_reason=finish_reason,
            served_model=getattr(resp, "model", None),
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
            http_status=200,
            reasoning_text=reasoning_text,
            output_issue=output_issue,
        )
