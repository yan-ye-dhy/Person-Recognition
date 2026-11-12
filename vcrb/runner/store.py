"""实验结果持久化（SQLite results/runs.db）。

幂等键：(run_id, model_id, experiment, sample_id)。
- status=ok 的记录在续跑时直接跳过（避免重复付费调用）；
- status=api_failure 的记录可通过 --retry-failed 重试；
- 每条记录同时追加写入 raw JSONL（完整请求/响应快照，不含 API Key）。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..paths import RAW_RESPONSES_DIR, RUNS_DB, ensure_dirs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    label TEXT DEFAULT '',
    experiments TEXT NOT NULL,      -- JSON list ["A","B"]
    models TEXT NOT NULL,           -- JSON list of model ids
    prompt_version TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    fairness_meta TEXT DEFAULT '{}',-- JSON: 画框说明/参数差异/图片预处理
    status TEXT NOT NULL DEFAULT 'running',
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS records (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    experiment TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    work_slug TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    ok INTEGER NOT NULL,
    http_status INTEGER,
    latency_ms INTEGER,
    error_type TEXT,
    error_message TEXT,
    served_model TEXT,
    usage TEXT DEFAULT '{}',
    raw_json TEXT NOT NULL,         -- 完整请求响应快照（脱敏）
    raw_file TEXT NOT NULL,         -- raw JSONL 路径
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, experiment, sample_id)
);
CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);
CREATE TABLE IF NOT EXISTS verify (
    model_id TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error_type TEXT,
    error_message TEXT,
    served_model TEXT,
    raw_text_head TEXT DEFAULT '',
    PRIMARY KEY (model_id, checked_at)
);
CREATE TABLE IF NOT EXISTS review_overrides (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    experiment TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, experiment, sample_id)
);
"""


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, db_path: Path = RUNS_DB) -> None:
        ensure_dirs()
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- runs ----
    def create_run(
        self,
        run_id: str,
        experiments: list[str],
        models: list[str],
        prompt_version: str,
        dataset_hash: str,
        dataset_path: str,
        fairness_meta: dict[str, Any],
        label: str = "",
        note: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runs (run_id, created_at, label, experiments, models,
                              prompt_version, dataset_hash, dataset_path,
                              fairness_meta, status, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                run_id,
                _now(),
                label,
                json.dumps(experiments),
                json.dumps(models),
                prompt_version,
                dataset_hash,
                dataset_path,
                json.dumps(fairness_meta, ensure_ascii=False),
                note,
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
        if not rows:
            return None
        run = dict(rows[0])
        run["experiments"] = json.loads(run["experiments"])
        run["models"] = json.loads(run["models"])
        run["fairness_meta"] = json.loads(run["fairness_meta"] or "{}")
        return run

    def latest_run_id(self) -> str | None:
        rows = self.conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchall()
        return rows[0]["run_id"] if rows else None

    def finish_run(self, run_id: str, status: str = "done", note: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, note = ? WHERE run_id = ?",
            (status, note, run_id),
        )
        self.conn.commit()

    # ---- records ----
    def upsert_record(self, record: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO records
            (run_id, model_id, experiment, sample_id, work_slug, attempt, ok,
             http_status, latency_ms, error_type, error_message, served_model,
             usage, raw_json, raw_file, created_at)
            VALUES (:run_id, :model_id, :experiment, :sample_id, :work_slug,
                    :attempt, :ok, :http_status, :latency_ms, :error_type,
                    :error_message, :served_model, :usage, :raw_json,
                    :raw_file, :created_at)
            """,
            record,
        )
        self.conn.commit()

    def existing_ok_keys(self, run_id: str) -> set[tuple[str, str, str]]:
        rows = self.conn.execute(
            "SELECT model_id, experiment, sample_id FROM records WHERE run_id = ? AND ok = 1",
            (run_id,),
        ).fetchall()
        return {(r["model_id"], r["experiment"], r["sample_id"]) for r in rows}

    def failed_keys(self, run_id: str) -> set[tuple[str, str, str]]:
        rows = self.conn.execute(
            "SELECT model_id, experiment, sample_id FROM records WHERE run_id = ? AND ok = 0",
            (run_id,),
        ).fetchall()
        return {(r["model_id"], r["experiment"], r["sample_id"]) for r in rows}

    def records(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM records WHERE run_id = ? ORDER BY model_id, experiment, sample_id",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["ok"] = bool(item["ok"])
            try:
                item["usage"] = json.loads(item.get("usage") or "{}")
            except json.JSONDecodeError:
                item["usage"] = {}
            try:
                item["raw"] = json.loads(item.get("raw_json") or "{}")
            except json.JSONDecodeError:
                item["raw"] = {}
            out.append(item)
        return out

    def run_progress(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(ok) AS ok_count,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failed
            FROM records WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "ok": row["ok_count"] or 0,
            "failed": row["failed"] or 0,
        }

    # ---- raw jsonl ----
    @staticmethod
    def raw_file_for(run_id: str, model_id: str, experiment: str, base: Path | None = None) -> Path:
        base = base or RAW_RESPONSES_DIR
        safe_model = model_id.replace("/", "_")
        return base / run_id / f"{safe_model}_{experiment}.jsonl"

    def append_raw(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- verify ----
    def save_verify(
        self,
        model_id: str,
        ok: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        served_model: str | None = None,
        raw_text_head: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO verify
            (model_id, checked_at, ok, error_type, error_message, served_model, raw_text_head)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (model_id, _now(), int(ok), error_type, error_message, served_model, raw_text_head[:300]),
        )
        self.conn.commit()

    def latest_verify(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT v.* FROM verify v
            JOIN (SELECT model_id, MAX(checked_at) AS t FROM verify GROUP BY model_id) latest
              ON v.model_id = latest.model_id AND v.checked_at = latest.t
            """
        ).fetchall()
        return {r["model_id"]: dict(r) for r in rows}

    # ---- overrides ----
    def set_override(
        self, run_id: str, model_id: str, experiment: str, sample_id: str,
        outcome: str, note: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO review_overrides
            (run_id, model_id, experiment, sample_id, outcome, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, model_id, experiment, sample_id, outcome, note, _now()),
        )
        self.conn.commit()

    def overrides(self, run_id: str) -> dict[tuple[str, str, str], dict[str, str]]:
        rows = self.conn.execute(
            "SELECT * FROM review_overrides WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {
            (r["model_id"], r["experiment"], r["sample_id"]): {
                "outcome": r["outcome"],
                "note": r["note"],
            }
            for r in rows
        }
