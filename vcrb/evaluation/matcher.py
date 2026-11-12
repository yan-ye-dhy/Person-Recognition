"""角色名匹配：别名命中、演员名识别、相似字符串转人工复核。

原则（用户要求）：
- 命中人工确认的 canonical 名或别名 → 正确；
- 回答演员真名 → actor_name（视为未正确完成任务）；
- 字符串高度相似但未精确/别名/子串命中 → needs_review，绝不自动判对；
- 不因两个姓名「看起来像」就判定正确。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..annotation.rosters import Roster

# 去掉的标点/符号（含中英文括号、引号、书名号、间隔号等）
_PUNCT_TABLE = dict.fromkeys(
    map(
        ord,
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
        " \t\r\n!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        "·、。．，：；！？「」『』（）《》〈〉【】〔〕…―--～〜"
        "“”‘’\"'′″",
    ),
    None,
)

_ROLE_MARK_RE = re.compile(r"[（(][^（）()]*?(?:饰|扮演|出演)[^（）()]*?[）)]")
_NAME_SPLIT_RE = re.compile(r"(?:饰演|扮演|出演|饰)")


def normalize_name(name: str | None) -> str:
    """姓名归一化：NFKC → 去『（XX饰）』片段 → 去空白与标点。"""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", str(name))
    text = _ROLE_MARK_RE.sub("", text)  # “高启强（张颂文饰）” → “高启强”
    text = text.translate(_PUNCT_TABLE)
    text = text.replace(" ", "").replace("　", "")
    return text.strip()


def split_role_marker(name: str) -> str:
    """『张颂文饰演的高启强』/『张颂文饰高启强』→ 取角色一侧（最后一段）。"""
    text = unicodedata.normalize("NFKC", str(name))
    parts = [p for p in _NAME_SPLIT_RE.split(text) if p.strip()]
    if len(parts) >= 2:
        last = parts[-1].strip()
        if last.startswith("的"):
            last = last[1:]
        return normalize_name(last)
    return normalize_name(name)


@dataclass
class MatchResult:
    kind: str  # exact / alias / substring / actor / actor_of_other / other_character / fuzzy / none / empty
    matched: str = ""
    similarity: float = 0.0

    @property
    def is_correct(self) -> bool:
        return self.kind in {"exact", "alias", "substring"}

    @property
    def needs_review(self) -> bool:
        return self.kind == "fuzzy"


class CharacterMatcher:
    def __init__(self, rosters: dict[str, Roster], fuzzy_threshold: float = 0.9) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        # 归一化名 → 角色canonical；归一化演员 → 角色
        self._names: dict[str, dict[str, str]] = {}
        self._actors: dict[str, dict[str, str]] = {}
        for slug, roster in rosters.items():
            name_map: dict[str, str] = {}
            actor_map: dict[str, str] = {}
            for char in roster.characters:
                name_map[normalize_name(char.name)] = char.name
                # needs_review 条目的别名未经人工确认，不得用于评分别名（防误判正确）；
                # canonical 名保留（GT 由标注者从名单选取，相当于人工确认过名字本身）
                if not char.needs_review:
                    for alias in char.aliases:
                        name_map[normalize_name(alias)] = char.name
                for actor in char.actors:
                    actor_map[normalize_name(actor)] = char.name
            self._names[slug] = {k: v for k, v in name_map.items() if k}
            self._actors[slug] = {k: v for k, v in actor_map.items() if k}

    def match(
        self,
        predicted: str | None,
        work_slug: str,
        gt_character: str,
        gt_aliases: list[str] | None = None,
    ) -> MatchResult:
        gt_names = {normalize_name(gt_character)}
        gt_names |= {normalize_name(a) for a in (gt_aliases or [])}
        gt_names.discard("")

        candidates = [normalize_name(predicted)]
        derived = split_role_marker(predicted or "")
        if derived and derived not in candidates:
            candidates.append(derived)

        for p in candidates:
            if not p:
                continue
            if p in gt_names:
                return MatchResult("exact", matched=p)
            # 子串包含（“电视剧狂飙里的高启强” / 只答姓氏别名的场景，要求长度≥2 防误伤）
            for g in gt_names:
                if len(p) >= 2 and len(g) >= 2 and (p in g or g in p):
                    return MatchResult("substring", matched=p)
            names = self._names.get(work_slug, {})
            if p in names:
                if names[p] == gt_character:
                    return MatchResult("alias", matched=p)
                return MatchResult("other_character", matched=names[p])
            actors = self._actors.get(work_slug, {})
            if p in actors:
                if actors[p] == gt_character:
                    return MatchResult("actor", matched=p)
                return MatchResult("actor_of_other", matched=actors[p])
        # 模糊相似 → 转人工
        best = 0.0
        best_name = ""
        for p in candidates:
            if not p:
                continue
            for g in gt_names:
                ratio = SequenceMatcher(None, p, g).ratio()
                if ratio > best:
                    best, best_name = ratio, g
        if best >= self.fuzzy_threshold:
            return MatchResult("fuzzy", matched=best_name, similarity=best)
        if not any(candidates):
            return MatchResult("empty")
        return MatchResult("none", similarity=best)
