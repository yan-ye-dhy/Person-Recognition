"""图片搜索来源。三通道设计（任一失败不阻塞整体采集）：

1. BingImageSource —— Bing 图片异步接口 HTML 解析（无需 API Key，可能受反爬限制）
2. UrlListSource  —— data/url_lists/{slug}.txt 静态 URL 清单（人工/外部工具准备）
3. 手动投放目录 data/manual_import/{slug}/（由 CLI `collect import` 扫描）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx


@dataclass
class ImageHit:
    url: str
    source_type: str = "web_image"
    license_note: str = ""
    referer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "source_type": self.source_type,
            "license_note": self.license_note,
            "referer": self.referer,
        }


# 查询词 → 素材类型推断（溯源分类用；不伪造，不确定记 unknown/web_image）
def source_type_from_query(query: str) -> str:
    if "剧照" in query:
        return "official_still"
    if "截图" in query or "截屏" in query:
        return "episode_screenshot"
    return "web_image"


class BingImageSource:
    """Bing 图片搜索（async 接口，解析 murl 原图链接）。"""

    SEARCH_URL = "https://www.bing.com/images/async"

    def __init__(self, user_agent: str, timeout: float = 15.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    def search(self, query: str, limit: int = 35) -> list[ImageHit]:
        params = f"q={quote_plus(query)}&first=1&count={max(limit, 35)}&mmasync=1"
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        try:
            resp = httpx.get(
                f"{self.SEARCH_URL}?{params}",
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return []  # 网络受阻：如实返回空，由上层记录
        return self.parse_html(resp.text, referer=f"https://www.bing.com/search?q={quote_plus(query)}")

    _MURL_RE = re.compile(r'"murl"\s*:\s*"(https?://[^"]+)"')

    def parse_html(self, html: str, referer: str = "") -> list[ImageHit]:
        # cn.bing.com 会把属性做 HTML 实体编码（murl&quot;:...），先反转义再匹配
        import html as _html

        text = _html.unescape(html)
        urls = self._MURL_RE.findall(text)
        seen: set[str] = set()
        hits: list[ImageHit] = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            hits.append(
                ImageHit(
                    url=url,
                    source_type="web_image",
                    license_note="Bing 图片搜索结果，来源以 source_url 为准",
                    referer=referer,
                )
            )
        return hits


class UrlListSource:
    """静态 URL 清单文件（每行一个 URL；支持 # 注释与 JSON 行扩展字段）。"""

    def __init__(self, list_path: Path) -> None:
        self.list_path = list_path

    def hits(self) -> list[ImageHit]:
        if not self.list_path.exists():
            return []
        out: list[ImageHit] = []
        for line in self.list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(
                    ImageHit(
                        url=str(data.get("url", "")),
                        source_type=str(data.get("source_type", "web_image")),
                        license_note=str(data.get("license_note", "")),
                    )
                )
            else:
                out.append(ImageHit(url=line))
        return [h for h in out if h.url]
