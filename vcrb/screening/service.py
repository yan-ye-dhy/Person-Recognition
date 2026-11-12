"""筛查编排：run_screening（全量重筛） / backfill_evidence（回补页面标题证据）
/ report_screening（六项数字报告）。

- 自动筛查不覆盖人工决定：某样本最近一次归属决定的 actor 以 "user" 开头
  时，自动规则跳过该样本（可经 --force 覆盖）。
- 视觉指标写入 attribution_evidence，人工审核页可见判定依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Work, load_settings
from ..dataset.store import AnnotationStore
from ..paths import CANDIDATES_DIR
from .rules import classify_sample
from .visual import analyze_image


@dataclass
class ScreenReport:
    work_slug: str
    total: int = 0
    decided: int = 0
    kept_user: int = 0
    skipped_missing_file: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "total": self.total,
            "decided": self.decided,
            "kept_user": self.kept_user,
            "skipped_missing_file": self.skipped_missing_file,
            "counts": self.counts,
        }


def run_screening(
    store: AnnotationStore,
    work: Work,
    settings: dict[str, Any],
    candidates_dir: Path | None = None,
    force: bool = False,
) -> ScreenReport:
    """对一部作品的全量候选执行 证据规则 + 视觉启发式 筛查。"""
    cfg = settings.get("screening", {})
    blocklist = list(cfg.get("blocklist_domains", []))
    cdir = candidates_dir or (CANDIDATES_DIR / work.slug)
    report = ScreenReport(work_slug=work.slug)

    for row in store.samples(work.slug):
        report.total += 1
        if not force:
            actor = store.last_attribution_actor(row["pk"])
            if actor and actor.startswith("user"):
                report.kept_user += 1
                continue
        img = cdir / row["filename"]
        if not img.exists():
            report.skipped_missing_file += 1
            continue
        visual = analyze_image(img, cfg, is_animation=work.category == "animation")
        verdict = classify_sample(
            work=work,
            source_url=row.get("source_url") or "",
            page_title=row.get("page_title") or "",
            page_url=row.get("page_url") or "",
            visual_flags=visual.flags,
            blocklist_domains=blocklist,
        )
        evidence = dict(verdict.evidence)
        evidence["visual_metrics"] = visual.metrics
        evidence["visual_flags"] = visual.flags
        store.set_attribution(
            [row["pk"]],
            verdict.status,
            verdict.reasons,
            evidence=evidence,
            actor="auto:rules",
        )
        report.decided += 1
        report.counts[verdict.status] = report.counts.get(verdict.status, 0) + 1
    return report


@dataclass
class BackfillReport:
    work_slug: str
    queries: int = 0
    hits: int = 0
    matched: int = 0
    updated: int = 0
    rescreened: int = 0
    errors: list[str] = field(default_factory=list)


def backfill_evidence(
    store: AnnotationStore,
    work: Work,
    settings: dict[str, Any],
    candidates_dir: Path | None = None,
    force: bool = False,
) -> BackfillReport:
    """重放搜索词拿页面标题/URL（免费，非模型 API），按图片 URL 回填证据。

    仅更新 page_title/page_url 为空的样本；回填后对未被人工决定的样本
    自动重跑规则。
    """
    from ..collectors.sources import BingImageSource

    cfg = settings.get("collect", {})
    scfg = settings.get("screening", {})
    blocklist = list(scfg.get("blocklist_domains", []))
    bing = BingImageSource(
        user_agent=str(cfg.get("user_agent", "vcrb")),
        timeout=float(cfg.get("request_timeout_seconds", 15)),
    )
    report = BackfillReport(work_slug=work.slug)

    url_map: dict[str, tuple[str, str]] = {}
    for query in work.search_queries:
        report.queries += 1
        try:
            hits = bing.search(query)
        except Exception as exc:  # 网络失败如实记录，不中断
            report.errors.append(f"{query[:30]}: {type(exc).__name__}")
            continue
        report.hits += len(hits)
        for hit in hits:
            if hit.url not in url_map and (hit.page_title or hit.page_url):
                url_map[hit.url] = (hit.page_url, hit.page_title)

    cdir = candidates_dir or (CANDIDATES_DIR / work.slug)
    for row in store.samples(work.slug):
        pair = url_map.get(row.get("source_url") or "")
        if pair is None:
            continue
        report.matched += 1
        if not store.set_page_evidence(row["pk"], pair[0], pair[1]):
            continue  # 原有证据保留
        report.updated += 1
        actor = store.last_attribution_actor(row["pk"])
        if not force and actor and actor.startswith("user"):
            continue
        img = cdir / row["filename"]
        flags: list[str] = []
        if img.exists():
            flags = analyze_image(
                img, scfg, is_animation=work.category == "animation"
            ).flags
        verdict = classify_sample(
            work=work,
            source_url=row.get("source_url") or "",
            page_title=pair[1],
            page_url=pair[0],
            visual_flags=flags,
            blocklist_domains=blocklist,
        )
        store.set_attribution(
            [row["pk"]], verdict.status, verdict.reasons,
            evidence={
                "image_host": verdict.evidence.get("image_host", ""),
                "page_host": verdict.evidence.get("page_host", ""),
                "title": verdict.evidence.get("title", ""),
                "matched_token": verdict.evidence.get("matched_token", ""),
                "trusted_domain": verdict.evidence.get("trusted_domain", False),
                "backfilled": True,
            },
            actor="auto:rules",
        )
        report.rescreened += 1
    return report


def report_screening(
    store: AnnotationStore, works: list[Work], target_per_work: int = 100
) -> list[dict[str, Any]]:
    """用户要求 9 的六项数字（每作品一行）。"""
    attr = store.attribution_stats()
    rows: list[dict[str, Any]] = []
    for work in works:
        bucket = attr.get(work.slug, {})
        samples = store.samples(work.slug)
        total = len(samples)
        # 自动排除 = rejected 且最近一次决定是自动规则
        auto_rejected = 0
        eligible = 0
        for r in samples:
            actor = store.last_attribution_actor(r["pk"]) or ""
            status = r.get("attribution_status") or "pending"
            if status == "rejected" and actor.startswith("auto"):
                auto_rejected += 1
            if status == "verified" and r.get("status") == "verified":
                eligible += 1
        verified_attr = bucket.get("verified", 0)
        needs = bucket.get("needs_review", 0) + bucket.get("pending", 0)
        rows.append(
            {
                "work": work.name,
                "slug": work.slug,
                "candidates": total,
                "auto_rejected": auto_rejected,
                "attribution_verified": verified_attr,
                "needs_human": needs,
                "benchmark_ready": eligible,
                "gap_to_100": max(0, target_per_work - eligible),
            }
        )
    return rows


__all__ = ["BackfillReport", "ScreenReport", "backfill_evidence", "report_screening", "run_screening"]
