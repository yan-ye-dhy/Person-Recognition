"""报告生成测试：图表 / Markdown / HTML。"""

from __future__ import annotations

import json

from tests.conftest import build_tiny_dataset, make_work
from tests.test_scoring import _record, _roster, _samples
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.evaluation.scoring import ScoreResult, score_run
from vcrb.reports.charts import render_charts
from vcrb.reports.html_report import render_html_report
from vcrb.reports.markdown_report import render_markdown_report
from vcrb.runner.store import RunStore


def _scored_run(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    run_id = "run_report1"
    store.create_run(run_id, ["A", "B"], ["m1"], "v1", "hash", "path", {})
    for exp in ("A", "B"):
        for i, s in enumerate(_samples()):
            name = s.target_character if i % 2 == 0 else None
            raw = json.dumps(
                {"character_name": name, "is_uncertain": name is None, "reason": "r"},
                ensure_ascii=False,
            )
            store.upsert_record(
                _record(run_id, "m1", exp, s.sample_id, ok=True, raw_text=raw)
            )
    result = score_run(
        store, run_id, _samples(), CharacterMatcher({"kuangbiao": _roster()}),
        {"scoring": {"refusal_keywords": []}},
    )
    return store, run_id, result


def test_render_charts_with_data(tmp_path):
    _, _, result = _scored_run(tmp_path)
    charts = render_charts(result, tmp_path / "charts")
    names = {p.name for p in charts}
    assert names == {
        "overall_accuracy.png", "ab_lift.png", "by_work.png",
        "by_importance.png", "by_difficulty.png", "error_types.png",
    }
    for p in charts:
        assert p.stat().st_size > 1000


def test_render_charts_placeholder(tmp_path):
    empty = ScoreResult(run_id="r", prompt_version="v1", dataset_hash="h", records=[])
    charts = render_charts(empty, tmp_path / "charts")
    assert len(charts) == 6
    assert all(p.exists() for p in charts)


def test_markdown_report_sections_and_red_line(tmp_path):
    _, run_id, result = _scored_run(tmp_path)
    md = render_markdown_report(
        result,
        tmp_path / "report.md",
        models_info={"m1": {"vendor": "Mock", "tier": "standard", "display_name": "M1", "official_series": ""}},
        run_meta={"created_at": "2026-01-01", "label": "test"},
        chart_names=["overall_accuracy.png"],
    )
    text = md.read_text(encoding="utf-8")
    assert "影视角色识别评测报告" in text
    assert "双口径" in text
    assert "同厂商档位对比" in text
    assert "人工复核记录" in text
    assert "实验结论" in text
    assert "![overall_accuracy.png]" in text
    assert "run_report1" in text


def test_markdown_report_empty_shows_pending(tmp_path):
    empty = ScoreResult(run_id="r2", prompt_version="v1", dataset_hash="h", records=[])
    md = render_markdown_report(empty, tmp_path / "report2.md")
    text = md.read_text(encoding="utf-8")
    assert "待测" in text
    assert "不生成任何模型优劣结论" in text  # 红线声明


def test_html_error_browser(tmp_path):
    dataset_dir = tmp_path / "dataset"
    samples, _works = build_tiny_dataset(dataset_dir, n_per_work=2)
    # 构造错误记录：预测错名
    store = RunStore(tmp_path / "runs.db")
    run_id = "run_html1"
    store.create_run(run_id, ["A"], ["m1"], "v1", "h", "p", {})
    for s in samples:
        raw = json.dumps(
            {"character_name": "错误角色名", "is_uncertain": False, "reason": "x"},
            ensure_ascii=False,
        )
        store.upsert_record(_record(run_id, "m1", "A", s.sample_id, ok=True, raw_text=raw))
    from tests.conftest import write_dataset_json

    write_dataset_json(dataset_dir, samples)
    result = score_run(
        store, run_id, samples,
        CharacterMatcher({"kuangbiao": _roster()}),
        {"scoring": {"refusal_keywords": []}},
    )
    # 样本包含琅琊榜，但 roster 只有狂飙 → 该作品匹配走 none → wrong_name
    out = render_html_report(result, store, dataset_dir, tmp_path / "errors.html")
    text = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64" in text
    assert "错误角色名" in text
    assert "wrong_name" in text
    assert "CARDS" in text
