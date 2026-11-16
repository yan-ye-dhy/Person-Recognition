"""Provider 统一抽象。

职责边界（重要）：
- Provider 只负责「调用与传输」：把图片+提示词发给模型，取回原始文本。
- 「响应语义」判定（拒答/无法确定/名字对错）一律交给 evaluation 层，
  Provider 不做任何内容猜测。
- outcome 只有两类：ok（拿到模型回复）/ api_failure（网络/服务端故障重试后仍失败）。
- ok 仅表示接口调用成功；空正文和截断通过 output_issue 单独记录，不改写成网络失败。
- reasoning_text 仅供审计，不是最终答案，绝不交给角色评分解析器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..models import ModelSpec

# 可重试的错误类型（传输层故障）
RETRYABLE_ERROR_TYPES = {"network", "timeout", "rate_limit", "server_error"}


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class ProviderResult:
    ok: bool
    raw_text: str = ""
    finish_reason: str | None = None
    served_model: str | None = None   # 响应中模型自报的实际版本（response.model）
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    attempts: int = 1
    error_type: str | None = None     # network/timeout/rate_limit/server_error/auth_error/bad_request/...
    error_message: str | None = None
    http_status: int | None = None
    reasoning_text: str = ""          # 可见推理字段单独留存，绝不回填 raw_text
    output_issue: str | None = None    # empty_final_content / truncated_output；不触发付费重试
    search_metadata: dict[str, Any] = field(default_factory=dict)
    http_metadata: dict[str, Any] = field(default_factory=dict)
    # Request/response IDs and their provenance for optional per-HTTP audits.
    # Only provider-returned evidence. Configuration or an intent count alone
    # must not be treated as proof that retrieval actually completed.


class VisionProvider(ABC):
    """所有厂商适配器的统一接口。"""

    name: str = "base"

    @abstractmethod
    def recognize(
        self,
        model: ModelSpec,
        image_data_url: str,
        system_prompt: str,
        user_prompt: str,
        gen: GenerationParams,
    ) -> ProviderResult:
        """识别 image_data_url（data:image/...;base64,...）中红框标注的人物。

        实现方契约：
        - 不得在请求中携带模型 ID 以外的任何样本身份信息；
        - 传输层错误按策略在内部重试（指数退避），重试耗尽返回 ok=False；
        - 绝不把 API Key 写入异常信息或结果对象。
        """
