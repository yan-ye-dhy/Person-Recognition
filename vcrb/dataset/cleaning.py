"""清洗流水线：有效性检测 → 去重（精确 + 感知哈希）→ 质量筛选 → 候选集。

产物：data/candidates/{slug}/cand_0001.jpg... + manifest.jsonl（溯源随图流转）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from ..imageutils import check_image, is_solid_color


@dataclass
class CleanReport:
    work_slug: str
    total: int = 0
    kept: int = 0
    invalid: int = 0
    quality_rejected: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "total": self.total,
            "kept": self.kept,
            "invalid": self.invalid,
            "quality_rejected": self.quality_rejected,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
        }


def _md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _phash(path: Path) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """两个十六进制感知哈希字符串的汉明距离（按位异或计数）。"""
    if len(hash_a) != len(hash_b):
        return 64  # 视为完全不同
    n = int(hash_a, 16) ^ int(hash_b, 16)
    return bin(n).count("1")


def is_near_duplicate(hash_a: str, hash_b: str, threshold: int) -> bool:
    return hamming_distance(hash_a, hash_b) <= threshold


def load_manifest(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """读取 raw 目录的 manifest.jsonl → {原文件名: 条目}。"""
    manifest_path = raw_dir / "manifest.jsonl"
    mapping: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("file"):
                mapping[str(entry["file"])] = entry
    return mapping


def clean_work(
    work_slug: str,
    raw_dir: Path,
    candidates_dir: Path,
    settings: dict[str, Any],
) -> CleanReport:
    report = CleanReport(work_slug=work_slug)
    cfg = settings.get("clean", {})
    min_width = int(cfg.get("min_width", 300))
    min_height = int(cfg.get("min_height", 200))
    threshold = int(cfg.get("phash_hamming_threshold", 8))

    if not raw_dir.exists():
        report.reasons.append(f"raw 目录不存在：{raw_dir}")
        return report

    manifest = load_manifest(raw_dir)
    files = sorted(
        p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    report.total = len(files)
    if not files:
        return report

    seen_md5: dict[str, str] = {}
    seen_phash: list[str] = []
    kept: list[tuple[Path, dict[str, Any]]] = []

    for path in files:
        check = check_image(path, min_width=min_width, min_height=min_height)
        if not check.valid:
            report.invalid += 1
            report.reasons.append(f"{path.name}: {check.reason}")
            continue
        if is_solid_color(path):
            report.quality_rejected += 1
            report.reasons.append(f"{path.name}: 纯色/空画面")
            continue
        digest = _md5(path)
        if digest in seen_md5:
            report.exact_duplicates += 1
            continue
        ph = _phash(path)
        if any(is_near_duplicate(ph, prev, threshold) for prev in seen_phash):
            report.near_duplicates += 1
            continue
        seen_md5[digest] = path.name
        seen_phash.append(ph)
        kept.append((path, manifest.get(path.name, {})))

    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = candidates_dir / "manifest.jsonl"
    with out_manifest.open("w", encoding="utf-8") as fh:
        for idx, (path, meta) in enumerate(kept, start=1):
            new_name = f"cand_{idx:04d}{path.suffix.lower()}"
            shutil.copyfile(path, candidates_dir / new_name)
            entry = {
                "file": new_name,
                "original_file": path.name,
                "source_type": meta.get("source_type", "unknown"),
                "source_url": meta.get("source_url", ""),
                "retrieved_at": meta.get("retrieved_at", ""),
                "license_note": meta.get("license_note", ""),
                "cleaned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            # 清洗改名后仍须能回到原播放页与原视频时间点。
            for key in (
                "page_url", "page_title", "video_timestamp_seconds", "source_channel",
                "segment_index", "segment_start_seconds", "timestamp_basis",
                "decode_warnings", "requires_visual_review",
            ):
                if key in meta:
                    entry[key] = meta[key]
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    report.kept = len(kept)
    return report
