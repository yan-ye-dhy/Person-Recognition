"""模型输出解析：JSON 容错提取 + 拒答检测。

与拒答的边界（用户要求严格区分）：
- refused   模型明确拒绝执行任务（内容命中拒答关键词且未给出角色名）
- uncertain 模型尝试识别但表示无法确定（is_uncertain=true / 名字为空、无拒答词）
- parse_error 响应无法解析为 JSON 且不构成拒答
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParsedResponse:
    raw_text: str
    parse_ok: bool = False
    character_name: str | None = None
    is_uncertain: bool | None = None
    reason: str = ""
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def extract_json_text(raw: str) -> str:
    """剥掉代码块围栏/前后噪声，截取首个 { 到最后一个 } 的片段。"""
    text = (raw or "").strip()
    if not text:
        return ""
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def parse_response(raw_text: str | None) -> ParsedResponse:
    raw = raw_text or ""
    parsed = ParsedResponse(raw_text=raw)
    snippet = extract_json_text(raw)
    if not snippet:
        parsed.error = "响应为空"
        return parsed
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        parsed.error = f"JSON 解析失败: {exc.msg}（片段: {snippet[:80]}…）"
        return parsed
    if not isinstance(data, dict):
        parsed.error = f"JSON 顶层不是对象: {type(data).__name__}"
        return parsed
    parsed.parse_ok = True
    name = None
    for key in ("character_name", "characterName", "name"):
        if key in data:
            name = data[key]
            break
    if name is not None and not isinstance(name, str):
        name = str(name)
    if name is not None and name.strip() == "":
        name = None
    if name is not None:
        name = name.strip()
    uncertain = None
    for key in ("is_uncertain", "isUncertain", "uncertain"):
        if key in data:
            uncertain = data[key]
            break
    if isinstance(uncertain, str):
        uncertain = uncertain.strip().lower() in {"true", "yes", "是", "无法确定"}
    parsed.character_name = name
    parsed.is_uncertain = bool(uncertain) if uncertain is not None else None
    parsed.reason = str(data.get("reason", "") or "")
    return parsed


# 默认拒答关键词（settings.yaml 可扩展）
DEFAULT_REFUSAL_KEYWORDS = [
    "无法识别身份",
    "不能识别人脸",
    "无法识别人脸",
    "不进行人脸识别",
    "不提供人脸识别",
    "涉及隐私",
    "无法确认人物身份",
    "涉及真实人物",
    "出于隐私",
    "我不能识别",
    "我无法识别",
    "抱歉，我不能",
]


def looks_like_refusal(text: str | None, keywords: list[str] | None = None) -> bool:
    """保守检测：内容命中关键词才判拒答（避免把『无法确定』误判为拒答）。"""
    if not text:
        return False
    body = text or ""
    for kw in keywords or DEFAULT_REFUSAL_KEYWORDS:
        if kw and kw in body:
            return True
    return False
