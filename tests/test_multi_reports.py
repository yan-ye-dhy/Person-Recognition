"""Reports expose task-specific denominators and synthetic-result provenance."""

import csv
import json
import re
from xml.etree import ElementTree

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.reports import bars, save_reports
from tests.test_multi_scoring import make_result


def test_reports_split_ab_c_show_mock_and_export_matching_metrics(tmp_path):
    matcher = CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [CharacterEntry("安欣"), CharacterEntry("李响")])})
    result, store, _samples = make_result(tmp_path, matcher)
    files = save_reports(result, tmp_path / "reports/multi/mock_demo", store.get_run("multi_mock"))
    summary = json.loads(files["summary.json"].read_text())
    assert summary["is_mock"] is True and summary["benchmark_protocol"] == "multi-v2"
    with files["metrics_ab.csv"].open(encoding="utf-8-sig", newline="") as f:
        ab = list(csv.DictReader(f))
    with files["metrics_c.csv"].open(encoding="utf-8-sig", newline="") as f:
        c = list(csv.DictReader(f))
    assert {r["experiment"] for r in ab} == {"A", "B"} and {r["experiment"] for r in c} == {"C"}
    assert all(r["person_accuracy_denominator"] == "2" for r in ab)
    assert c[0]["localization_precision_denominator"] == c[0]["end_to_end_recall_denominator"] == "2"
    with files["detail.csv"].open(encoding="utf-8-sig", newline="") as f:
        detail = list(csv.DictReader(f))
    assert len(detail) == 6 and {r["outcome"] for r in detail} == {"correct"}
    text = files["index.html"].read_text()
    for expected in ("MOCK / 合成响应演示", "不是任何真实模型成绩", "指定编号人物身份识别", "自主发现、定位及识别",
                     "不与A/B准确率混合排名", "95% CI", "P50 / P95", "费用", "分作品", "1/2/3/4+"):
        assert expected in text
    charts = [ElementTree.fromstring(svg) for svg in re.findall(r'<svg\b[\s\S]*?</svg>', text)]
    assert len(charts) == 3
    ns = {"svg": "http://www.w3.org/2000/svg"}
    titles = [chart.find("svg:title", ns).text for chart in charts]
    assert "A/B人物准确率" in titles[0] and "A/B图片完全正确率" in titles[1]
    assert "C任务F1" in titles[2] and all("MOCK" in title for title in titles)
    assert [len(chart.findall("svg:rect", ns)) for chart in charts] == [2, 2, 2]
    assert "persons" in summary["quality"] and files["ab_effects.csv"].is_file()
    store.close()


def test_svg_missing_metric_has_label_without_zero_bar_and_escapes_model_name():
    text = bars([('model<script>|A', None), ('real-zero|B', 0.0), ('scored|A', .5)], "A/B人物准确率", is_mock=True)
    chart = ElementTree.fromstring(re.search(r'<svg\b[\s\S]*?</svg>', text)[0])
    ns = {"svg": "http://www.w3.org/2000/svg"}
    rects = chart.findall("svg:rect", ns)
    assert len(rects) == 2 and {r.get("data-label") for r in rects} == {"real-zero|B", "scored|A"}
    assert any(t.text == "无可评分GT" for t in chart.findall("svg:text", ns))
    assert '<script>' not in text and '&lt;script&gt;' in text


def test_report_escapes_model_output_and_never_overwrites_v1(tmp_path):
    matcher = CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [CharacterEntry("安欣"), CharacterEntry("李响")])})
    result, store, _samples = make_result(tmp_path, matcher)
    result.records[0]["person_details"][0].update(predicted='<script>alert("x")</script>', outcome="wrong_name")
    files = save_reports(result, tmp_path / "multi/safe")
    text = files["index.html"].read_text()
    assert '<script>alert' not in text and '&lt;script&gt;' in text
    old = tmp_path / "reports/2026-11-13"
    old.mkdir(parents=True)
    marker = old / "index.html"
    marker.write_text("old report remains")
    with pytest.raises(ValueError, match="v1 report"):
        save_reports(result, old)
    assert marker.read_text() == "old report remains"
    store.close()


@pytest.mark.parametrize("status,not_run,show_banner", [("running", 0, True), ("limited", 0, True), ("done", 1, True), ("done", 0, False)])
def test_real_partial_report_has_nonfinal_banner_without_changing_scores(tmp_path, status, not_run, show_banner):
    matcher = CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [CharacterEntry("安欣"), CharacterEntry("李响")])})
    result, store, _samples = make_result(tmp_path, matcher)
    result.is_mock = False
    next(iter(result.metrics.values()))["not_run"] = not_run
    before = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
    files = save_reports(result, tmp_path / "multi/real", {"status": status})
    text = files["index.html"].read_text()
    assert ("真实评测进行中 / 未全执行：非最终结论" in text) is show_banner
    assert "MOCK / 合成响应演示" not in text
    assert json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) == before
    store.close()
