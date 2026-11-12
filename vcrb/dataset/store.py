"""标注存储（SQLite data/annotations.db）与正式集导出。

- candidates 落库后默认 pending，人工在 Streamlit 标注页改成
  verified / invalid / uncertain；
- 仅 verified 样本可导出；导出时图片复制为与角色无关的序号文件名
  并重编码清除 EXIF（防答案泄露）。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..imageutils import check_image, strip_and_normalize
from ..paths import ANNOTATIONS_DB, ensure_dirs
from .schema import Sample

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    work_slug TEXT NOT NULL,
    image_path TEXT NOT NULL UNIQUE,   -- 相对 candidates 目录的文件名
    filename TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    target_character TEXT,
    character_aliases TEXT DEFAULT '[]',
    character_importance TEXT,
    visual_difficulty TEXT,
    is_multi_person INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    target_bbox TEXT,
    source_type TEXT DEFAULT 'unknown',
    source_url TEXT DEFAULT '',
    note TEXT DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_work_status ON samples(work_slug, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CandidateItem:
    work_slug: str
    filename: str
    width: int
    height: int
    source_type: str = "unknown"
    source_url: str = ""


class AnnotationStore:
    def __init__(self, db_path: Path = ANNOTATIONS_DB) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- 候选入库 ----
    def register_candidates(
        self, work_slug: str, items: list[CandidateItem], candidates_dir: Path
    ) -> int:
        """登记候选样本（已存在路径则跳过），返回新增数量。"""
        added = 0
        for item in items:
            try:
                self.conn.execute(
                    """
                    INSERT INTO samples
                    (work_slug, image_path, filename, width, height,
                     status, source_type, source_url, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        work_slug,
                        f"{work_slug}/{item.filename}",
                        item.filename,
                        item.width,
                        item.height,
                        item.source_type,
                        item.source_url,
                        _now(),
                    ),
                )
                added += 1
            except sqlite3.IntegrityError:
                continue  # 重复登记幂等
        self.conn.commit()
        return added

    # ---- 标注更新 ----
    _EDITABLE = {
        "status": str,
        "target_character": str,
        "character_importance": str,
        "visual_difficulty": str,
        "is_multi_person": int,
        "source_type": str,
        "source_url": str,
        "note": str,
    }

    def update_annotation(self, pk: int, fields: dict[str, Any]) -> None:
        """更新标注字段。tags/bbox/aliases 以 JSON 序列化存储。"""
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in {"character_aliases", "tags"}:
                sets.append(f"{key} = ?")
                values.append(json.dumps(value or [], ensure_ascii=False))
            elif key == "target_bbox":
                sets.append("target_bbox = ?")
                values.append(json.dumps(value) if value is not None else None)
            elif key in self._EDITABLE:
                sets.append(f"{key} = ?")
                values.append(self._EDITABLE[key](value))
            else:
                raise ValueError(f"不可编辑字段：{key}")
        sets.append("updated_at = ?")
        values.append(_now())
        values.append(pk)
        self.conn.execute(
            f"UPDATE samples SET {', '.join(sets)} WHERE pk = ?", values
        )
        self.conn.commit()

    # ---- 查询 ----
    def samples(
        self, work_slug: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM samples"
        conds, params = [], []
        if work_slug:
            conds.append("work_slug = ?")
            params.append(work_slug)
        if status:
            conds.append("status = ?")
            params.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY pk"
        rows = self.conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("character_aliases", "tags"):
                try:
                    item[key] = json.loads(item.get(key) or "[]")
                except (json.JSONDecodeError, TypeError):
                    item[key] = []
            if item.get("target_bbox"):
                try:
                    item["target_bbox"] = json.loads(item["target_bbox"])
                except (json.JSONDecodeError, TypeError):
                    item["target_bbox"] = None
            out.append(item)
        return out

    def get(self, pk: int) -> dict[str, Any] | None:
        rows = self.conn.execute("SELECT * FROM samples WHERE pk = ?", (pk,)).fetchall()
        return dict(rows[0]) if rows else None

    def stats(self) -> dict[str, dict[str, int]]:
        """每作品 × 状态计数。"""
        rows = self.conn.execute(
            "SELECT work_slug, status, COUNT(*) AS n FROM samples GROUP BY work_slug, status"
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["work_slug"], {})[row["status"]] = row["n"]
        return out

    def verified(self) -> list[dict[str, Any]]:
        return self.samples(status="verified")


# ---------------------------------------------------------------- 导出


def export_dataset(
    store: AnnotationStore,
    works: list[Any],  # list[Work]
    out_dir: Path | None = None,
    candidates_dir: Path | None = None,
    target_per_group: int = 20,
) -> Path:
    """把 verified 样本导出为匿名化正式集（dataset.json + 重编码图片）。

    - 文件名统一 {slug}_{0001}.jpg，与角色身份无关；
    - 重存 JPEG 清除 EXIF；bbox 保持归一化（坐标与图片内容一一对应）；
    - 配额达成情况写入 dataset.json 的 quota 字段，缺口如实展示。
    """
    ensure_dirs()
    out_dir = out_dir or FINAL_DIR
    candidates_dir = candidates_dir or out_dir.parent / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_by_slug = {w.slug: w for w in works}

    samples_out: list[dict[str, Any]] = []
    quota: dict[str, dict[str, int]] = {}
    for slug, work in work_by_slug.items():
        rows = [r for r in store.verified() if r["work_slug"] == slug]
        rows.sort(key=lambda r: r["pk"])  # 稳定顺序保证 sample_id 可复现
        groups: dict[str, int] = {}
        for idx, row in enumerate(rows, start=1):
            src = candidates_dir / slug / row["filename"]
            if not src.is_file():
                raise FileNotFoundError(f"候选图片缺失：{src}")
            sample_id = f"{slug}_{idx:04d}"
            rel_path = f"{slug}/{sample_id}.jpg"
            strip_and_normalize(src, out_dir / rel_path)
            check = check_image(out_dir / rel_path)
            # 校验重编码后尺寸与标注坐标仍对应（内容未变，仅元数据清除）
            sample = Sample(
                sample_id=sample_id,
                work_slug=slug,
                work_name=work.name,
                work_category=work.category,
                image_path=rel_path,
                target_character=row["target_character"] or "",
                character_importance=row["character_importance"] or "supporting",
                visual_difficulty=row["visual_difficulty"] or "normal",
                is_multi_person=bool(row["is_multi_person"]),
                character_aliases=list(row.get("character_aliases") or []),
                tags=list(row.get("tags") or []),
                target_bbox=row.get("target_bbox"),
                image_width=check.width,
                image_height=check.height,
                annotation_status="verified",
                source_type=row.get("source_type") or "unknown",
                source_url=row.get("source_url") or "",
                note=row.get("note") or "",
            )
            errors = sample.validate()
            if errors:
                raise ValueError(f"{sample_id} 导出校验失败：{errors}")
            # 防泄露断言：文件名/路径不得包含角色名
            assert sample.target_character not in rel_path, "文件名泄露角色名"
            samples_out.append(sample.to_dict())
            groups[sample.sampling_group] = groups.get(sample.sampling_group, 0) + 1
        quota[slug] = groups

    payload = {
        "exported_at": _now(),
        "total": len(samples_out),
        "quota_per_group": quota,
        "target_per_group": target_per_group,
        "samples": samples_out,
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir / "dataset.json"
