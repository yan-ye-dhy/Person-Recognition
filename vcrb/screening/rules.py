"""归属判定规则：来源证据优先，视觉信号只做否决/降级，绝不单独放行。

判定矩阵（用户要求 2/5）：
- verified（自动）= 可信域命中 **且** 页面标题/页面 URL/图片 URL 含作品标识
- rejected（自动）= 拉黑域（二创平台）｜动画作品中的黑白图（漫画截图嫌疑）
- needs_review（自动）= 仅有作品标识（域不可信）｜仅可信域无标识｜
  竖版海报嫌疑｜异常画幅｜插画留白嫌疑｜真人作品的黑白图｜
  疑似拼贴/组图（网格结构启发式存在误报——水墨云雾、贯穿字幕条也会触发，
  经实测校准后不作为自动排除依据，交人工批量审核判定）
- pending = 未筛查（如图片不可读时保留待人工）
视觉标记单独出现时永远不产生 verified。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass
class RuleVerdict:
    status: str                 # verified / rejected / needs_review / pending
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def evidence_tokens(work: Any) -> list[str]:
    """作品标识词（中文名、slug、官方英文名/别名）。"""
    tokens = [work.name, work.slug, *getattr(work, "evidence_aliases", ())]
    return [t for t in tokens if t]


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower()


def _domain_matches(host: str, domain: str) -> bool:
    """host 等于 domain 或以其为后缀（子域也算），防 'notpixiv.net' 误伤。"""
    if not host or not domain:
        return False
    return host == domain or host.endswith("." + domain)


def classify_sample(
    work: Any,
    source_url: str,
    page_title: str = "",
    page_url: str = "",
    visual_flags: list[str] | None = None,
    blocklist_domains: list[str] | None = None,
) -> RuleVerdict:
    """对单条样本做归属判定。visual_flags 来自 visual.analyze_image。"""
    flags = list(visual_flags or [])
    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "image_host": _domain_of(source_url),
        "page_host": _domain_of(page_url),
        "title": (page_title or "").strip()[:200],
    }

    if "unreadable" in flags:
        return RuleVerdict("pending", ["图片不可读，保留待人工"], evidence)

    tokens = evidence_tokens(work)
    haystack = unquote(f"{page_title} {page_url} {source_url}").lower()
    matched_token = next(
        (t for t in tokens if t.lower() in haystack), ""
    )
    evidence["matched_token"] = matched_token

    trusted = list(getattr(work, "trusted_domains", ()) or [])
    trusted_match = ""
    for d in trusted:
        if _domain_matches(evidence["image_host"], d):
            trusted_match = f"{evidence['image_host']}(图片域)"
            break
        if _domain_matches(evidence["page_host"], d):
            trusted_match = f"{evidence['page_host']}(页面域)"
            break
    is_trusted = bool(trusted_match)
    evidence["trusted_domain"] = is_trusted
    evidence["trusted_match"] = trusted_match
    blocked = any(
        _domain_matches(evidence["image_host"], d)
        or _domain_matches(evidence["page_host"], d)
        for d in (blocklist_domains or [])
    )
    evidence["blocked_domain"] = blocked

    is_animation = work.category == "animation"

    # ---- 拉黑域：二创/同人平台 → 明确排除 ----
    if blocked:
        return RuleVerdict(
            "rejected",
            ["来源属于二创/同人平台（拉黑域）"],
            evidence,
        )

    # ---- 动画作品：黑白图 → 漫画截图嫌疑，明确排除 ----
    if is_animation and "grayscale_image" in flags:
        reasons.append("低饱和度黑白画面：漫画/黑白素材嫌疑（动画作品要求彩色正片）")
        return RuleVerdict("rejected", reasons, evidence)

    # ---- 拼贴网格 → 降级人工复核（启发式有实测误报，不自动排除，
    #      也不允许自动 verified：多格组图不适合做单帧测试素材） ----
    if "collage_grid" in flags:
        if is_trusted and matched_token:
            reasons.append(
                f"可信域+作品标识「{matched_token}」，但检出等距贯穿分割结构："
                "宣传拼贴/组图嫌疑，需人工确认"
            )
        else:
            reasons.append("检出等距贯穿分割结构：宣传拼贴/组图嫌疑（启发式有误报，需人工确认）")
        _append_visual_notes(reasons, [f for f in flags if f != "collage_grid"], is_animation)
        return RuleVerdict("needs_review", reasons, evidence)

    # ---- 可信域 + 作品标识 → verified（唯一自动放行路径） ----

    # ---- 可信域 + 作品标识 → verified（唯一自动放行路径） ----
    if is_trusted and matched_token:
        reasons.append(
            f"可信来源域 {trusted_match} 且出现作品标识「{matched_token}」"
        )
        _append_visual_notes(reasons, flags, is_animation)
        return RuleVerdict("verified", reasons, evidence)

    # ---- 以下全部降级 ----
    if matched_token and not is_trusted:
        reasons.append(
            f"出现作品标识「{matched_token}」但来源域不可信（{evidence['image_host']}）"
        )
    elif is_trusted and not matched_token:
        reasons.append(
            f"可信域（{evidence['image_host'] or evidence['page_host']}）但未见作品标识"
        )
    else:
        reasons.append(
            "无作品归属证据：标题/URL 均未出现作品标识，来源域不可信"
        )
    _append_visual_notes(reasons, flags, is_animation)
    return RuleVerdict("needs_review", reasons, evidence)


def _append_visual_notes(reasons: list[str], flags: list[str], is_animation: bool) -> None:
    if "poster_layout" in flags:
        reasons.append("竖版画幅：海报嫌疑（要求正片截图/官方剧照，需人工确认）")
    if "odd_aspect" in flags:
        reasons.append("画幅明显偏离常见正片比例，需人工确认")
    if "white_border" in flags and is_animation:
        reasons.append("边缘大面积留白：插画/杂志页嫌疑（动画类需人工确认）")
    if "grayscale_image" in flags and not is_animation:
        reasons.append("低饱和度黑白画面：需人工确认（真人作品偶有黑白戏份）")
