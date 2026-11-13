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

from ..config import DEFAULT_TARGET_PER_GROUP, DEFAULT_TARGET_PER_WORK
from ..imageutils import check_image, strip_and_normalize
from ..paths import ANNOTATIONS_DB, FINAL_DIR, ensure_dirs
from .schema import SAMPLING_GROUPS, Sample, sampling_group

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
    md5 TEXT,                          -- 内容指纹：重跑 clean 重新编号时保持标注-图片对应
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_work_status ON samples(work_slug, status);
"""

_ATTRIBUTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS attribution_log (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_pk INTEGER NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    actor TEXT NOT NULL,          -- auto:rules / auto:visual / user
    reasons TEXT DEFAULT '[]',    -- JSON 数组，逐条筛判依据
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attr_log_sample ON attribution_log(sample_pk);
CREATE TABLE IF NOT EXISTS purge_log (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    work_slug TEXT NOT NULL,
    filename TEXT NOT NULL,
    original_file TEXT DEFAULT '',
    md5 TEXT,
    source_url TEXT DEFAULT '',
    page_title TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    deleted_at TEXT NOT NULL
);
"""

ATTRIBUTION_STATUSES = ("pending", "verified", "rejected", "needs_review")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_md5(path: Path) -> str | None:
    import hashlib

    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _image_size(path: Path) -> tuple[int | None, int | None]:
    from PIL import Image

    try:
        with Image.open(path) as img:
            return img.size
    except OSError:
        return None, None


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
        self.conn.executescript(_ATTRIBUTION_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """老库补列（无 md5 / 归属核验列的旧 schema），再建依赖列的索引。"""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(samples)")}
        if "md5" not in cols:
            self.conn.execute("ALTER TABLE samples ADD COLUMN md5 TEXT")
        for col, default in (
            ("attribution_status", "TEXT DEFAULT 'pending'"),
            ("attribution_reasons", "TEXT DEFAULT '[]'"),
            ("attribution_evidence", "TEXT DEFAULT '{}'"),
            ("page_url", "TEXT DEFAULT ''"),
            ("page_title", "TEXT DEFAULT ''"),
        ):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE samples ADD COLUMN {col} {default}")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_work_md5 ON samples(work_slug, md5)"
        )

    # ---- 归属核验（screening） ----
    def set_attribution(
        self,
        pks: list[int],
        status: str,
        reasons: list[str],
        evidence: dict[str, Any] | None = None,
        actor: str = "user",
    ) -> int:
        """批量写归属状态（verified/rejected/needs_review），append 审计日志。

        evidence 仅在提供时覆盖（自动规则可携带指标；人工操作保留原证据）。
        返回实际变更条数（状态相同的不重复写）。
        """
        if status not in ATTRIBUTION_STATUSES:
            raise ValueError(f"非法归属状态：{status}")
        changed = 0
        now = _now()
        for pk in pks:
            row = self.conn.execute(
                "SELECT attribution_status FROM samples WHERE pk = ?", (pk,)
            ).fetchone()
            if row is None:
                continue
            old = row["attribution_status"] or "pending"
            sets = [
                "attribution_status = ?",
                "attribution_reasons = ?",
            ]
            values: list[Any] = [
                status,
                json.dumps(reasons, ensure_ascii=False),
            ]
            if evidence is not None:
                sets.append("attribution_evidence = ?")
                values.append(json.dumps(evidence, ensure_ascii=False))
            sets.append("updated_at = ?")
            values.append(now)
            values.append(pk)
            self.conn.execute(
                f"UPDATE samples SET {', '.join(sets)} WHERE pk = ?", values
            )
            self.conn.execute(
                """
                INSERT INTO attribution_log
                (sample_pk, old_status, new_status, actor, reasons, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pk,
                    old,
                    status,
                    actor,
                    json.dumps(reasons, ensure_ascii=False),
                    now,
                ),
            )
            changed += 1
        self.conn.commit()
        return changed

    def last_attribution_actor(self, pk: int) -> str | None:
        rows = self.conn.execute(
            "SELECT actor FROM attribution_log WHERE sample_pk = ? ORDER BY pk DESC LIMIT 1",
            (pk,),
        ).fetchall()
        return rows[0]["actor"] if rows else None

    def attribution_stats(self) -> dict[str, dict[str, int]]:
        """每作品 × 归属状态计数。"""
        rows = self.conn.execute(
            "SELECT work_slug, attribution_status, COUNT(*) AS n FROM samples "
            "GROUP BY work_slug, attribution_status"
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["work_slug"], {})[row["attribution_status"] or "pending"] = row["n"]
        return out

    def purge_samples(self, items: list[dict[str, Any]]) -> int:
        """删除 samples 行（连带归属日志），并写 purge_log 留痕。

        items 每项需含 pk/filename/md5/source_url/page_title/reason
        （文件删除由调用方负责，本方法只管数据库）。
        """
        now = _now()
        for it in items:
            self.conn.execute(
                "DELETE FROM attribution_log WHERE sample_pk = ?", (it["pk"],)
            )
            self.conn.execute("DELETE FROM samples WHERE pk = ?", (it["pk"],))
            self.conn.execute(
                """
                INSERT INTO purge_log
                (work_slug, filename, original_file, md5, source_url, page_title, reason, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(it.get("work_slug", "")),
                    str(it.get("filename", "")),
                    str(it.get("original_file", "")),
                    it.get("md5"),
                    str(it.get("source_url", "") or ""),
                    str(it.get("page_title", "") or ""),
                    str(it.get("reason", "")),
                    now,
                ),
            )
        self.conn.commit()
        return len(items)

    def purge_log(self, work_slug: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM purge_log"
        params: tuple[Any, ...] = ()
        if work_slug:
            sql += " WHERE work_slug = ?"
            params = (work_slug,)
        rows = self.conn.execute(sql + " ORDER BY pk DESC", params).fetchall()
        return [dict(r) for r in rows]

    def attribution_log(self, sample_pk: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM attribution_log WHERE sample_pk = ? ORDER BY pk",
            (sample_pk,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["reasons"] = json.loads(item.get("reasons") or "[]")
            except (json.JSONDecodeError, TypeError):
                item["reasons"] = []
            out.append(item)
        return out

    def set_page_evidence(self, pk: int, page_url: str, page_title: str) -> bool:
        """回填来源页证据（仅当原先为空时写入），返回是否写入。"""
        row = self.conn.execute(
            "SELECT page_url, page_title FROM samples WHERE pk = ?", (pk,)
        ).fetchone()
        if row is None:
            return False
        if row["page_url"] or row["page_title"]:
            return False
        self.conn.execute(
            "UPDATE samples SET page_url = ?, page_title = ?, updated_at = ? WHERE pk = ?",
            (page_url, page_title, _now(), pk),
        )
        self.conn.commit()
        return True

    def close(self) -> None:
        self.conn.close()

    # ---- 候选入库 ----
    def register_candidates(
        self, work_slug: str, items: list[CandidateItem], candidates_dir: Path | None
    ) -> int:
        """登记候选样本（已存在路径则跳过），返回新增数量。"""
        added = 0
        for item in items:
            digest: str | None = None
            if candidates_dir is not None:
                digest = _file_md5(candidates_dir / work_slug / item.filename)
            try:
                self.conn.execute(
                    """
                    INSERT INTO samples
                    (work_slug, image_path, filename, width, height,
                     status, source_type, source_url, md5, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (
                        work_slug,
                        f"{work_slug}/{item.filename}",
                        item.filename,
                        item.width,
                        item.height,
                        item.source_type,
                        item.source_url,
                        digest,
                        _now(),
                    ),
                )
                added += 1
            except sqlite3.IntegrityError:
                continue  # 重复登记幂等
        self.conn.commit()
        return added

    def sync_candidates(self, work_slug: str, candidates_dir: Path) -> dict[str, int]:
        """把 candidates 目录中的 cand_* 图片同步入库（以内容 md5 为身份，幂等）。

        解决的问题：`vcrb clean` 每次重跑都会按保留顺序重新编号
        （cand_0003.jpg 可能换了一张图）。以文件名为身份会把旧标注错配到
        新图；以 md5 为身份则：
        - md5 已存在 → 跳过；若原文件已消失（重编号），把该行改指向新
          文件名，已有标注原样保留；
        - md5 不存在但文件名已被另一张图占用 → 磁盘改名
          cand_XXXX_{md5前6}.jpg 再登记，避免覆盖他行；
        - 其余 → 新增 pending 行。

        返回 {"registered", "relinked", "renamed", "skipped"}。
        """
        import json as _json

        result = {"registered": 0, "relinked": 0, "renamed": 0, "skipped": 0}
        if not candidates_dir.exists():
            return result

        # 候选目录的 manifest（clean 产物，含 source 溯源）
        manifest: dict[str, dict[str, Any]] = {}
        manifest_path = candidates_dir / "manifest.jsonl"
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and entry.get("file"):
                    manifest[str(entry["file"])] = entry

        files = sorted(
            p for p in candidates_dir.iterdir()
            if p.name.startswith("cand_") and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for path in files:
            digest = _file_md5(path)
            if digest is None:
                continue
            row = self.conn.execute(
                "SELECT * FROM samples WHERE work_slug = ? AND md5 = ?",
                (work_slug, digest),
            ).fetchone()
            if row is not None:
                old = candidates_dir / row["filename"]
                if row["filename"] == path.name:
                    result["skipped"] += 1
                elif old.exists():
                    # 同内容两个文件名同时在磁盘上（罕见）：保留原行不重复登记
                    result["skipped"] += 1
                else:
                    self.conn.execute(
                        "UPDATE samples SET filename = ?, image_path = ?, updated_at = ? WHERE pk = ?",
                        (path.name, f"{work_slug}/{path.name}", _now(), row["pk"]),
                    )
                    self.conn.commit()
                    result["relinked"] += 1
                # 顺手回填缺失的来源页证据（manifest 更新后重跑 sync 即可补上）
                if not (row["page_url"] or row["page_title"]):
                    meta0 = manifest.get(path.name) or manifest.get(row["filename"]) or {}
                    if meta0.get("page_url") or meta0.get("page_title"):
                        self.conn.execute(
                            "UPDATE samples SET page_url = ?, page_title = ?, updated_at = ? WHERE pk = ?",
                            (
                                str(meta0.get("page_url", "")),
                                str(meta0.get("page_title", "")),
                                _now(),
                                row["pk"],
                            ),
                        )
                        self.conn.commit()
                continue

            name_row = self.conn.execute(
                "SELECT * FROM samples WHERE work_slug = ? AND filename = ?",
                (work_slug, path.name),
            ).fetchone()
            target_name = path.name
            if name_row is not None:
                if name_row["md5"] is None:
                    # 旧库行无指纹：若其文件与当前同内容则补指纹并跳过
                    old_digest = _file_md5(candidates_dir / name_row["filename"])
                    if old_digest == digest:
                        self.conn.execute(
                            "UPDATE samples SET md5 = ? WHERE pk = ?", (digest, name_row["pk"])
                        )
                        self.conn.commit()
                        result["skipped"] += 1
                        continue
                # 文件名冲突且内容不同 → 磁盘改名后登记
                target_name = f"{path.stem}_{digest[:6]}{path.suffix.lower()}"
                path.rename(candidates_dir / target_name)
                path = candidates_dir / target_name
                result["renamed"] += 1

            meta = manifest.get(target_name) or manifest.get(path.name) or {}
            width, height = _image_size(path)
            try:
                self.conn.execute(
                    """
                    INSERT INTO samples
                    (work_slug, image_path, filename, width, height,
                     status, source_type, source_url, page_url, page_title, md5, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_slug,
                        f"{work_slug}/{target_name}",
                        target_name,
                        width,
                        height,
                        str(meta.get("source_type", "unknown")),
                        str(meta.get("source_url", "")),
                        str(meta.get("page_url", "")),
                        str(meta.get("page_title", "")),
                        digest,
                        _now(),
                    ),
                )
                self.conn.commit()
                result["registered"] += 1
            except sqlite3.IntegrityError:
                result["skipped"] += 1
        return result

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
            for key in ("character_aliases", "tags", "attribution_reasons"):
                try:
                    item[key] = json.loads(item.get(key) or "[]")
                except (json.JSONDecodeError, TypeError):
                    item[key] = []
            for key in ("target_bbox", "attribution_evidence"):
                try:
                    item[key] = json.loads(item[key]) if item.get(key) else None
                except (json.JSONDecodeError, TypeError):
                    item[key] = None
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
    target_per_group: int = DEFAULT_TARGET_PER_GROUP,
    target_per_work: int = DEFAULT_TARGET_PER_WORK,
) -> Path:
    """把 verified 样本导出为匿名化正式集（dataset.json + 重编码图片）。

    - 文件名统一 {slug}_{0001}.jpg，与角色身份无关；
    - 重存 JPEG 清除 EXIF；bbox 保持归一化（坐标与图片内容一一对应）；
    - 配额达成情况写入 dataset.json 的 quota 字段，缺口如实展示；
    - 每组按登记顺序优先取至配额，再用其余合格样本补足每部总量上限；
    - 导出门禁：标注 status=verified 且 **作品归属 attribution_status=verified**
      才可进入正式集（归属 rejected/needs_review/pending 一律隔离）。
    """
    if target_per_group < 1 or target_per_work < 1:
        raise ValueError("每组与每部的采样目标必须为正整数")
    ensure_dirs()
    out_dir = out_dir or FINAL_DIR
    candidates_dir = candidates_dir or out_dir.parent / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_by_slug = {w.slug: w for w in works}

    samples_out: list[dict[str, Any]] = []
    quota: dict[str, dict[str, int]] = {}
    excluded_quarantine: dict[str, int] = {}
    eligible_per_work: dict[str, int] = {}
    gap_per_work: dict[str, int] = {}
    quota_shortfall: dict[str, dict[str, int]] = {}
    for slug, work in work_by_slug.items():
        verified_rows = [r for r in store.verified() if r["work_slug"] == slug]
        rows = [
            r for r in verified_rows
            if (r.get("attribution_status") or "pending") == "verified"
        ]
        excluded_quarantine[slug] = len(verified_rows) - len(rows)
        rows.sort(key=lambda r: r["pk"])  # 稳定顺序保证 sample_id 可复现
        eligible_per_work[slug] = len(rows)
        grouped = {group: [] for group in SAMPLING_GROUPS}
        for row in rows:
            group = sampling_group(
                row["character_importance"] or "supporting",
                row["visual_difficulty"] or "normal",
                bool(row["is_multi_person"]),
            )
            grouped[group].append(row)
        selected = []
        for position in range(target_per_group):
            for group in SAMPLING_GROUPS:
                if position < len(grouped[group]) and len(selected) < target_per_work:
                    selected.append(grouped[group][position])
        selected_pks = {row["pk"] for row in selected}
        for row in rows:
            if len(selected) >= target_per_work:
                break
            if row["pk"] not in selected_pks:
                selected.append(row)
        selected.sort(key=lambda r: r["pk"])
        groups = {group: 0 for group in SAMPLING_GROUPS}
        for idx, row in enumerate(selected, start=1):
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
        gap_per_work[slug] = max(0, target_per_work - len(selected))
        quota_shortfall[slug] = {
            group: max(0, target_per_group - count) for group, count in groups.items()
        }

    payload = {
        "exported_at": _now(),
        "total": len(samples_out),
        "quota_per_group": quota,
        "target_per_group": target_per_group,
        "target_per_work": target_per_work,
        "eligible_per_work": eligible_per_work,
        "gap_per_work": gap_per_work,
        "quota_shortfall": quota_shortfall,
        "quarantined_not_exported": excluded_quarantine,
        "samples": samples_out,
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir / "dataset.json"
