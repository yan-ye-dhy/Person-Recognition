"""网络图片采集：搜索 → 下载 → 落盘 + 溯源清单。

原则：
- 不为凑数伪造素材：下载失败/来源不明都如实记录在报告中；
- 每张图写 manifest 条目（来源 URL、素材类型、获取时间、许可备注）；
- 重复内容（md5）自动跳过。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..config import Work
from .sources import BingImageSource, ImageHit, UrlListSource, source_type_from_query


@dataclass
class CollectReport:
    work_slug: str
    hits: int = 0
    downloaded: int = 0
    skipped_small: int = 0
    skipped_duplicate: int = 0
    skipped_no_evidence: int = 0   # 来源证据预检未通过（标题/页面无作品标识）
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "hits": self.hits,
            "downloaded": self.downloaded,
            "skipped_small": self.skipped_small,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_no_evidence": self.skipped_no_evidence,
            "failed": self.failed,
        }


def _is_image_response(resp: httpx.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    return ctype.startswith("image/") or ctype == "application/octet-stream"


def has_work_evidence(hit: ImageHit, work: Work) -> bool:
    """来源证据预检：页面标题 / 页面 URL / 图片 URL 中出现作品标识。

    仅凭关键词搜索命中的图片默认不可信（用户要求 8），无作品标识的
    通用搜索结果不进入候选库。
    """
    from urllib.parse import unquote

    tokens = [work.name, work.slug, *work.evidence_aliases]
    haystack = unquote(
        f"{hit.page_title} {hit.page_url} {hit.url}"
    ).lower()
    return any(t.lower() in haystack for t in tokens if t)


def download_hits(
    hits: list[ImageHit],
    raw_dir: Path,
    user_agent: str,
    request_timeout: float,
    min_image_bytes: int,
    max_total: int,
    report: CollectReport,
    client: httpx.Client | None = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.jsonl"
    existing_md5: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_md5.add(json.loads(line).get("md5", ""))
            except json.JSONDecodeError:
                continue

    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": user_agent},
        timeout=request_timeout,
        follow_redirects=True,
    )
    try:
        for hit in hits:
            if report.downloaded >= max_total:
                break
            report.hits += 1
            try:
                headers = {"Referer": hit.referer} if hit.referer else {}
                resp = client.get(hit.url, headers=headers)
                resp.raise_for_status()
                if not _is_image_response(resp):
                    report.failed += 1
                    continue
                data = resp.content
            except httpx.HTTPError as exc:
                report.failed += 1
                report.errors.append(f"{hit.url[:120]}: {type(exc).__name__}")
                continue
            if len(data) < min_image_bytes:
                report.skipped_small += 1
                continue
            digest = hashlib.md5(data).hexdigest()
            if digest in existing_md5:
                report.skipped_duplicate += 1
                continue
            suffix = ".jpg"
            ctype = resp.headers.get("content-type", "").lower()
            if "png" in ctype:
                suffix = ".png"
            elif "webp" in ctype:
                suffix = ".webp"
            filename = f"web_{digest[:16]}{suffix}"
            (raw_dir / filename).write_bytes(data)
            entry = {
                "file": filename,
                "source_type": hit.source_type,
                "source_url": hit.url,
                "page_url": hit.page_url,
                "page_title": hit.page_title,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license_note": hit.license_note or "来源许可以 source_url 所示页面为准",
                "md5": digest,
            }
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing_md5.add(digest)
            report.downloaded += 1
    finally:
        if own_client:
            client.close()


def collect_web(
    work: Work,
    settings: dict[str, Any],
    raw_dir: Path,
    url_list_dir: Path | None = None,
    limit: int | None = None,
    client: httpx.Client | None = None,
    require_evidence: bool = True,
) -> CollectReport:
    """按 works.yaml 的查询词采集图片到 raw_dir。

    require_evidence=True（默认）：仅下载来源标题/页面带作品标识的命中，
    无证据的通用搜索结果直接跳过并计数（skipped_no_evidence）。
    UrlListSource 手动清单视为人工已核，不参与预检。
    """
    cfg = settings.get("collect", {})
    report = CollectReport(work_slug=work.slug)
    max_total = int(limit or cfg.get("max_images_per_work", 150))
    bing = BingImageSource(
        user_agent=str(cfg.get("user_agent", "vcrb")),
        timeout=float(cfg.get("request_timeout_seconds", 15)),
    )

    all_hits: list[ImageHit] = []
    for query in work.search_queries:
        stype = source_type_from_query(query)
        for hit in bing.search(query):
            hit.source_type = stype
            all_hits.append(hit)
    # 静态 URL 清单通道（存在即并入；人工清单视为已核）
    list_dir = url_list_dir
    if list_dir is None:
        from ..paths import DATA_DIR

        list_dir = DATA_DIR / "url_lists"
    url_source = UrlListSource(list_dir / f"{work.slug}.txt")
    manual_hits = url_source.hits()

    # 去重 URL
    seen: set[str] = set()
    deduped: list[ImageHit] = []
    for hit in [*all_hits, *manual_hits]:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        if require_evidence and not has_work_evidence(hit, work):
            report.skipped_no_evidence += 1
            continue
        deduped.append(hit)

    download_hits(
        deduped,
        raw_dir=raw_dir,
        user_agent=str(cfg.get("user_agent", "vcrb")),
        request_timeout=float(cfg.get("request_timeout_seconds", 15)),
        min_image_bytes=int(cfg.get("min_image_bytes", 10240)),
        max_total=max_total,
        report=report,
        client=client,
    )
    return report
