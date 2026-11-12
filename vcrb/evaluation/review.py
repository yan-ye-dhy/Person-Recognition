"""人工复核：列出待复核样本、改写评分结论（不触碰原始响应）。"""

from __future__ import annotations

from typing import Any

from ..runner.store import RunStore
from .scoring import OUTCOMES, ScoreResult

REVIEWABLE_OUTCOMES = [o for o in OUTCOMES if o != "not_run"]


def needs_review_records(result: ScoreResult) -> list[dict[str, Any]]:
    """待复核 = 模糊匹配命中（相似但未精确/别名命中）的样本。"""
    return [r.to_row() for r in result.records if r.needs_review]


def error_records(result: ScoreResult) -> list[dict[str, Any]]:
    return [r.to_row() for r in result.records if r.outcome not in {"correct", "not_run"}]


def set_override(
    store: RunStore,
    run_id: str,
    model_id: str,
    experiment: str,
    sample_id: str,
    outcome: str,
    note: str,
) -> None:
    """人工改判（outcome 必须是合法枚举）。重新 score 后自动生效。"""
    if outcome not in REVIEWABLE_OUTCOMES:
        raise ValueError(f"非法 outcome：{outcome}（合法值 {REVIEWABLE_OUTCOMES}）")
    store.set_override(run_id, model_id, experiment, sample_id, outcome, note)


def list_overrides(store: RunStore, run_id: str) -> list[dict[str, str]]:
    return [
        {"model_id": m, "experiment": e, "sample_id": s, **v}
        for (m, e, s), v in store.overrides(run_id).items()
    ]
