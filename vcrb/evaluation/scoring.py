"""评分与统计。

双口径指标（用户批准的调整 5）：
- 核心选型指标：有效响应识别准确率 = correct / (planned - api_failure - not_run)
- 并列展示：端到端正确率 = correct / planned；API 失败率 = api_failure / planned
所有分母不剔除任何未正确识别的响应（拒答/解析失败均计入有效响应）。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.schema import Sample
from ..runner.store import RunStore
from .matcher import CharacterMatcher
from .parser import looks_like_refusal, parse_response

OUTCOMES = [
    "correct",
    "wrong_name",
    "actor_name",
    "uncertain",
    "refused",
    "parse_error",
    "api_failure",
    "not_run",
]

OUTCOME_LABELS = {
    "correct": "正确",
    "wrong_name": "错误识别",
    "actor_name": "答演员名",
    "uncertain": "无法识别",
    "refused": "拒答",
    "parse_error": "格式错误",
    "api_failure": "API失败",
    "not_run": "未执行",
}


# ---------------------------------------------------------------- 统计学工具

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% 置信区间（纯 Python）。n<=0 时返回 (0, 0)。"""
    if n <= 0:
        return 0.0, 0.0
    k = max(0, min(k, n))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar(success_a: list[bool], success_b: list[bool]) -> dict[str, Any]:
    """A/B 配对差异检验（同一批样本）。返回 b/c 计数、统计量与 p 值。"""
    if len(success_a) != len(success_b):
        raise ValueError("配对样本长度不一致")
    b = sum(1 for a, c in zip(success_a, success_b) if (not a) and c)
    c = sum(1 for a, d in zip(success_a, success_b) if a and (not d))
    n = b + c
    result: dict[str, Any] = {"b": b, "c": c, "n_discordant": n}
    if n == 0:
        result.update({"statistic": 0.0, "p_value": 1.0, "method": "no_discordant_pairs"})
        return result
    try:  # scipy 可选依赖：精确二项检验
        from scipy.stats import binomtest  # type: ignore

        p = float(binomtest(min(b, c), n, 0.5).pvalue)
        result.update({"statistic": min(b, c), "p_value": p, "method": "binomial_exact(scipy)"})
        return result
    except ImportError:
        pass
    if n <= 1000:  # 精确二项（纯 Python）
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
        result.update({"statistic": min(b, c), "p_value": p, "method": "binomial_exact"})
    else:  # 正态近似（连续性校正卡方）
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = math.erfc(math.sqrt(chi2 / 2))
        result.update({"statistic": chi2, "p_value": p, "method": "chi2_normal_approx"})
    return result


# ---------------------------------------------------------------- 评分记录

@dataclass
class ScoredRecord:
    run_id: str
    model_id: str
    experiment: str
    sample_id: str
    work_slug: str
    gt_character: str
    predicted: str
    outcome: str
    needs_review: bool = False
    detail: str = ""
    overridden: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "experiment": self.experiment,
            "sample_id": self.sample_id,
            "work_slug": self.work_slug,
            "gt_character": self.gt_character,
            "predicted": self.predicted,
            "outcome": self.outcome,
            "outcome_label": OUTCOME_LABELS.get(self.outcome, self.outcome),
            "needs_review": self.needs_review,
            "detail": self.detail,
            "overridden": self.overridden,
        }


@dataclass
class CellMetrics:
    planned: int
    counts: Counter
    accuracy_valid: float
    accuracy_valid_ci: tuple[float, float]
    accuracy_e2e: float
    accuracy_e2e_ci: tuple[float, float]
    api_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "counts": {k: self.counts.get(k, 0) for k in OUTCOMES},
            "accuracy_valid": self.accuracy_valid,
            "accuracy_valid_ci95": list(self.accuracy_valid_ci),
            "accuracy_e2e": self.accuracy_e2e,
            "accuracy_e2e_ci95": list(self.accuracy_e2e_ci),
            "api_failure_rate": self.api_failure_rate,
        }


def _cell_metrics(records: list[ScoredRecord]) -> CellMetrics:
    counts = Counter(r.outcome for r in records)
    planned = len(records)
    api_failure = counts.get("api_failure", 0)
    not_run = counts.get("not_run", 0)
    valid = planned - api_failure - not_run
    correct = counts.get("correct", 0)
    return CellMetrics(
        planned=planned,
        counts=counts,
        accuracy_valid=correct / valid if valid else 0.0,
        accuracy_valid_ci=wilson_ci(correct, valid),
        accuracy_e2e=correct / planned if planned else 0.0,
        accuracy_e2e_ci=wilson_ci(correct, planned),
        api_failure_rate=api_failure / planned if planned else 0.0,
    )


@dataclass
class ScoreResult:
    run_id: str
    prompt_version: str
    dataset_hash: str
    records: list[ScoredRecord]
    metrics: dict[tuple[str, str], CellMetrics] = field(default_factory=dict)
    by_work: dict[tuple[str, str], dict[str, CellMetrics]] = field(default_factory=dict)
    by_importance: dict[tuple[str, str], dict[str, CellMetrics]] = field(default_factory=dict)
    by_difficulty: dict[tuple[str, str], dict[str, CellMetrics]] = field(default_factory=dict)
    by_multi_person: dict[tuple[str, str], dict[str, CellMetrics]] = field(default_factory=dict)
    per_character: dict[tuple[str, str], dict[str, dict[str, Any]]] = field(default_factory=dict)
    ab_tests: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def key(m: str, e: str) -> str:
            return f"{m}|{e}"

        return {
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "dataset_hash": self.dataset_hash,
            "metrics": {key(*k): v.to_dict() for k, v in self.metrics.items()},
            "by_work": {
                key(*k): {w: m.to_dict() for w, m in v.items()}
                for k, v in self.by_work.items()
            },
            "by_importance": {
                key(*k): {w: m.to_dict() for w, m in v.items()}
                for k, v in self.by_importance.items()
            },
            "by_difficulty": {
                key(*k): {w: m.to_dict() for w, m in v.items()}
                for k, v in self.by_difficulty.items()
            },
            "by_multi_person": {
                key(*k): {w: m.to_dict() for w, m in v.items()}
                for k, v in self.by_multi_person.items()
            },
            "per_character": {key(*k): v for k, v in self.per_character.items()},
            "ab_tests": self.ab_tests,
            "records": [r.to_row() for r in self.records],
        }


# ---------------------------------------------------------------- 主评分流程


def score_run(
    store: RunStore,
    run_id: str,
    samples: list[Sample],
    matcher: CharacterMatcher,
    settings: dict[str, Any],
    apply_overrides: bool = True,
) -> ScoreResult:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"未知 run_id：{run_id}")
    refusal_keywords = list(
        settings.get("scoring", {}).get("refusal_keywords", []) or []
    ) or None  # None → 使用 parser 默认表

    samples_by_id = {s.sample_id: s for s in samples}
    records_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in store.records(run_id):
        records_by_key[(rec["model_id"], rec["experiment"], rec["sample_id"])] = rec

    overrides = store.overrides(run_id) if apply_overrides else {}
    scored: list[ScoredRecord] = []
    for model_id in run["models"]:
        for exp in run["experiments"]:
            for sample in samples:
                key = (model_id, exp, sample.sample_id)
                rec = records_by_key.get(key)
                sr = _classify_record(run_id, model_id, exp, sample, rec, matcher, refusal_keywords)
                ov = overrides.get(key)
                if ov:
                    sr.detail = f"{sr.detail} | 人工复核: {sr.outcome}→{ov['outcome']}({ov['note']})"
                    sr.outcome = ov["outcome"]
                    sr.needs_review = False
                    sr.overridden = True
                scored.append(sr)

    result = ScoreResult(
        run_id=run_id,
        prompt_version=run.get("prompt_version", ""),
        dataset_hash=run.get("dataset_hash", ""),
        records=scored,
    )
    _annotate_dimensions(result, samples)
    _aggregate(result)
    return result


def _classify_record(
    run_id: str,
    model_id: str,
    exp: str,
    sample: Sample,
    rec: dict[str, Any] | None,
    matcher: CharacterMatcher,
    refusal_keywords: list[str] | None,
) -> ScoredRecord:
    sr = ScoredRecord(
        run_id=run_id,
        model_id=model_id,
        experiment=exp,
        sample_id=sample.sample_id,
        work_slug=sample.work_slug,
        gt_character=sample.target_character,
        predicted="",
        outcome="not_run",
    )
    if rec is None:
        return sr
    if not rec["ok"]:
        sr.outcome = "api_failure"
        sr.detail = f"{rec.get('error_type')}: {(rec.get('error_message') or '')[:120]}"
        return sr
    raw_text = (rec.get("raw", {}).get("response", {}) or {}).get("raw_text", "")
    parsed = parse_response(raw_text)
    refusal = looks_like_refusal(raw_text, refusal_keywords)
    sr.predicted = parsed.character_name or ""

    if not parsed.parse_ok:
        sr.outcome = "refused" if refusal else "parse_error"
        sr.detail = parsed.error or "（无 JSON）"
        return sr
    if refusal and parsed.character_name is None:
        sr.outcome = "refused"
        sr.detail = "命中拒答关键词且未给出角色名"
        return sr
    if parsed.is_uncertain or parsed.character_name is None:
        sr.outcome = "uncertain"
        sr.detail = parsed.reason[:120]
        return sr

    match = matcher.match(parsed.character_name, sample.work_slug, sample.target_character, sample.character_aliases)
    if match.is_correct:
        sr.outcome = "correct"
        sr.detail = f"匹配方式={match.kind}"
    elif match.kind == "actor":
        sr.outcome = "actor_name"
        sr.detail = f"回答了演员名（{sr.predicted}）而非角色名"
    else:
        sr.outcome = "wrong_name"
        sr.needs_review = match.needs_review
        detail_map = {
            "actor_of_other": "回答了其他角色的演员",
            "other_character": "回答了其他角色",
            "fuzzy": f"与正确答案高度相似（{match.similarity:.2f}），转人工复核",
            "none": "未命中任何角色",
            "empty": "回答为空",
        }
        sr.detail = detail_map.get(match.kind, match.kind)
    return sr


def _aggregate(result: ScoreResult) -> None:
    records = result.records
    models = sorted({r.model_id for r in records})
    exps = sorted({r.experiment for r in records})

    result.metrics = {
        (m, e): _cell_metrics([r for r in records if r.model_id == m and r.experiment == e])
        for m in models
        for e in exps
    }

    dims: dict[str, dict[str, Any]] = getattr(result, "_dim", {}) or {}

    def slice_by(attr: str, target: dict) -> None:
        grouped: dict[tuple[str, str], dict[Any, list[ScoredRecord]]] = {}
        for r in records:
            value = dims.get(r.sample_id, {}).get(attr)
            if value is None:
                value = r.work_slug if attr == "work_slug" else "未知"
            grouped.setdefault((r.model_id, r.experiment), {}).setdefault(value, []).append(r)
        target.update({k: {str(sk): _cell_metrics(sv) for sk, sv in v.items()} for k, v in grouped.items()})

    slice_by("work_slug", result.by_work)
    slice_by("_importance", result.by_importance)
    slice_by("_difficulty", result.by_difficulty)
    slice_by("_multi", result.by_multi_person)

    # 按角色统计 + 宏平均
    for m in models:
        for e in exps:
            chars: dict[str, dict[str, int]] = {}
            for r in records:
                if r.model_id == m and r.experiment == e:
                    ck = f"{r.work_slug}::{r.gt_character}"
                    bucket = chars.setdefault(ck, {"n": 0, "correct": 0})
                    bucket["n"] += 1
                    if r.outcome == "correct":
                        bucket["correct"] += 1
            detailed = {
                ck: {**v, "accuracy": (v["correct"] / v["n"]) if v["n"] else 0.0}
                for ck, v in chars.items()
            }
            macro = (
                sum(v["accuracy"] for v in detailed.values()) / len(detailed)
                if detailed
                else 0.0
            )
            result.per_character[(m, e)] = {"characters": detailed, "macro_accuracy": macro}

    # A/B 配对检验（每个模型）
    for m in models:
        a_map = {r.sample_id: r for r in records if r.model_id == m and r.experiment == "A"}
        b_map = {r.sample_id: r for r in records if r.model_id == m and r.experiment == "B"}
        shared = sorted(set(a_map) & set(b_map))
        pair = [
            (a_map[s], b_map[s])
            for s in shared
            if a_map[s].outcome not in {"api_failure", "not_run"}
            and b_map[s].outcome not in {"api_failure", "not_run"}
        ]
        if not pair:
            continue
        succ_a = [a.outcome == "correct" for a, _ in pair]
        succ_b = [b.outcome == "correct" for _, b in pair]
        test = mcnemar(succ_a, succ_b)
        acc_a = sum(succ_a) / len(pair)
        acc_b = sum(succ_b) / len(pair)
        result.ab_tests[m] = {
            "paired_n": len(pair),
            "accuracy_A_on_paired": acc_a,
            "accuracy_B_on_paired": acc_b,
            "delta_pp": (acc_b - acc_a) * 100,
            **test,
        }


def _annotate_dimensions(result: ScoreResult, samples: list[Sample]) -> None:
    """把样本三维度挂到结果上（_aggregate 使用）。"""
    dims = {
        s.sample_id: {
            "work_slug": s.work_slug,
            "_importance": s.character_importance,
            "_difficulty": s.visual_difficulty,
            "_multi": "多人同框" if s.is_multi_person else "单人/主体",
        }
        for s in samples
    }
    result._dim = dims  # type: ignore[attr-defined]


def save_score_result(result: ScoreResult, out_dir: Path) -> dict[str, Path]:
    """导出 scores：summary.json / metrics.csv / detail.csv。"""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model_id",
                "experiment",
                "planned",
                "valid_responses",
                "correct",
                "wrong_name",
                "actor_name",
                "uncertain",
                "refused",
                "parse_error",
                "api_failure",
                "not_run",
                "accuracy_valid",
                "accuracy_valid_ci95_low",
                "accuracy_valid_ci95_high",
                "accuracy_e2e",
                "api_failure_rate",
                "macro_accuracy_per_character",
            ]
        )
        for (m, e), cell in sorted(result.metrics.items()):
            counts = cell.counts
            valid = cell.planned - counts.get("api_failure", 0) - counts.get("not_run", 0)
            writer.writerow(
                [
                    m,
                    e,
                    cell.planned,
                    valid,
                    counts.get("correct", 0),
                    counts.get("wrong_name", 0),
                    counts.get("actor_name", 0),
                    counts.get("uncertain", 0),
                    counts.get("refused", 0),
                    counts.get("parse_error", 0),
                    counts.get("api_failure", 0),
                    counts.get("not_run", 0),
                    f"{cell.accuracy_valid:.4f}",
                    f"{cell.accuracy_valid_ci[0]:.4f}",
                    f"{cell.accuracy_valid_ci[1]:.4f}",
                    f"{cell.accuracy_e2e:.4f}",
                    f"{cell.api_failure_rate:.4f}",
                    f"{result.per_character.get((m, e), {}).get('macro_accuracy', 0.0):.4f}",
                ]
            )

    detail_path = out_dir / "detail.csv"
    detail_fields = [
        "run_id", "model_id", "experiment", "sample_id", "work_slug",
        "gt_character", "predicted", "outcome", "outcome_label",
        "needs_review", "detail", "overridden",
    ]
    with detail_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=detail_fields)
        writer.writeheader()
        for r in result.records:
            writer.writerow(r.to_row())

    return {
        "summary": summary_path,
        "metrics": metrics_path,
        "detail": detail_path,
    }
