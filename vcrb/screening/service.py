"""筛查编排：run_screening（全量重筛） / backfill_evidence（回补页面标题证据）
/ report_screening（六项数字报告）。

- 自动筛查不覆盖用户或用户授权的逐图代标决定（可经 --force 覆盖）。
- 视觉指标写入 attribution_evidence，人工审核页可见判定依据。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DEFAULT_TARGET_PER_WORK, Work, load_settings
from ..dataset.store import AnnotationStore
from ..paths import CANDIDATES_DIR
from .rules import classify_sample
from .visual import analyze_image


def _reviewed_actor(actor: str | None) -> bool:
    """仅保护用户决定和明确登记的授权代标，不保护一般自动模型输出。"""
    return bool(actor and (actor.startswith("user") or actor == "assistant:delegated_review"))


@dataclass
class ScreenReport:
    work_slug: str
    total: int = 0
    decided: int = 0
    kept_user: int = 0
    kept_assistant: int = 0
    skipped_missing_file: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "total": self.total,
            "decided": self.decided,
            "kept_user": self.kept_user,
            "kept_assistant": self.kept_assistant,
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
            if _reviewed_actor(actor):
                if actor == "assistant:delegated_review":
                    report.kept_assistant += 1
                else:
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
        if not force and _reviewed_actor(actor):
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
    store: AnnotationStore, works: list[Work], target_per_work: int = DEFAULT_TARGET_PER_WORK
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
                "target_per_work": target_per_work,
                "gap_to_target": max(0, target_per_work - eligible),
            }
        )
    return rows


@dataclass
class PurgeReport:
    work_slug: str
    mode: str
    planned: int = 0
    deleted_candidates: int = 0
    deleted_raws: int = 0
    missing_files: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "mode": self.mode,
            "planned": self.planned,
            "deleted_candidates": self.deleted_candidates,
            "deleted_raws": self.deleted_raws,
            "missing_files": self.missing_files,
        }


def purge_candidates(
    store: AnnotationStore,
    work: Work,
    mode: str,                       # rejected=仅已排除 | all=整池清空（重采前用）
    candidates_dir: Path | None = None,
    raw_dir: Path | None = None,
    apply: bool = False,
    reason: str = "",
    force: bool = False,
) -> PurgeReport:
    """删除不可用素材：候选文件 + 候选清单行 + 原始文件 + 原始清单行 + 库行。

    - 默认 dry-run（apply=False 只统计不动手）；
    - 删除明细写入 purge_log 表（文件名/md5/来源/原因），可追溯；
    - 原始 raw 文件一并删除，防止 `vcrb clean` 从 raw 复活被删素材；
    - all 模式默认保护归属 verified 样本；force=True 解除保护
      （仅在用户明确要求整池清空时使用）。
    """
    from ..paths import RAW_DIR

    if mode not in ("rejected", "all"):
        raise ValueError(f"非法 purge 模式：{mode}")
    cdir = candidates_dir or (CANDIDATES_DIR / work.slug)
    rdir = raw_dir or (RAW_DIR / work.slug)
    report = PurgeReport(work_slug=work.slug, mode=mode)

    # 候选 manifest：cand 文件名 → original_file
    cand_manifest: dict[str, str] = {}
    cand_manifest_path = cdir / "manifest.jsonl"
    if cand_manifest_path.exists():
        for line in cand_manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("file"):
                cand_manifest[str(entry["file"])] = str(entry.get("original_file", ""))

    targets: list[dict[str, Any]] = []
    for row in store.samples(work.slug):
        status = row.get("attribution_status") or "pending"
        if mode == "rejected" and status != "rejected":
            continue
        if mode == "all" and status == "verified" and not force:
            # 整池清空时保护已核实素材（防止误操作全清）；--force 显式解除
            report.reasons.append(
                f"跳过 {row['filename']}（归属已核实，--status all 不删除 verified；如需清空加 --force）"
            )
            continue
        targets.append(
            {
                "pk": row["pk"],
                "work_slug": work.slug,
                "filename": row["filename"],
                "md5": row.get("md5"),
                "source_url": row.get("source_url", ""),
                "page_title": row.get("page_title", ""),
                "original_file": cand_manifest.get(row["filename"], ""),
                "reason": reason or (f"purge:{mode}"),
                "status": status,
            }
        )
    report.planned = len(targets)
    if not apply:
        return report

    # 删文件 + 重写两份 manifest
    keep_cand_manifest: set[str] = {t["filename"] for t in targets}
    keep_raw_files: set[str] = {t["original_file"] for t in targets if t["original_file"]}
    for t in targets:
        cf = cdir / t["filename"]
        if cf.exists():
            cf.unlink()
            report.deleted_candidates += 1
        else:
            report.missing_files += 1
        if t["original_file"]:
            rf = rdir / t["original_file"]
            if rf.exists():
                rf.unlink()
                report.deleted_raws += 1

    for manifest_path, key_field, excluded in (
        (cand_manifest_path, "file", keep_cand_manifest),
        (rdir / "manifest.jsonl", "file", keep_raw_files),
    ):
        if not manifest_path.exists():
            continue
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)  # 坏行原样保留
                continue
            if isinstance(entry, dict) and str(entry.get(key_field, "")) in excluded:
                continue
            kept.append(line)
        manifest_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    store.purge_samples(targets)
    return report


__all__ = [
    "BackfillReport",
    "PurgeReport",
    "ScreenReport",
    "backfill_evidence",
    "purge_candidates",
    "report_screening",
    "run_screening",
]
