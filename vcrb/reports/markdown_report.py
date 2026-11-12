"""Markdown 总报告生成。

结论章节只基于真实数据填充；无数据时明确显示「待测」，
绝不生成虚构排行榜或模型优劣结论（用户验收红线）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.scoring import OUTCOME_LABELS, ScoreResult


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ci(lo: float, hi: float) -> str:
    return f"[{_pct(lo)}–{_pct(hi)}]"


def _cell_row(m: str, exp: str, result: ScoreResult, models_info: dict[str, Any]) -> list[str]:
    cell = result.metrics.get((m, exp))
    info = models_info.get(m, {})
    name = info.get("display_name", m)
    if not cell or cell.planned == 0:
        return [name, "待测", "待测"]
    return [
        name,
        f"{_pct(cell.accuracy_valid)} {_ci(*cell.accuracy_valid_ci)}",
        f"{cell.counts.get('correct', 0)}/{cell.planned}",
    ]


def render_markdown_report(
    result: ScoreResult,
    out_path: Path,
    dataset_stats: dict[str, Any] | None = None,
    models_info: dict[str, Any] | None = None,
    run_meta: dict[str, Any] | None = None,
    overrides: list[dict[str, str]] | None = None,
    chart_names: list[str] | None = None,
) -> Path:
    models_info = models_info or {}
    models = sorted({m for m, _ in result.metrics})
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("# 国产多模态大模型影视角色识别评测报告")
    add()
    meta = run_meta or {}
    add(f"- Run ID：`{result.run_id}`")
    add(f"- Prompt 版本：`{result.prompt_version}`")
    add(f"- 数据集快照哈希：`{result.dataset_hash[:16]}…`")
    if meta.get("created_at"):
        add(f"- 运行时间：{meta['created_at']}")
    if meta.get("label"):
        add(f"- 标签：{meta['label']}")
    add(f"- 参与模型数：{len(models)}；样本数：{dataset_stats.get('total', '待测') if dataset_stats else '待测'}")
    add()

    # ---- 数据集概览 ----
    add("## 1. 数据集概览")
    add()
    if dataset_stats and dataset_stats.get("by_work"):
        add("| 作品 | 样本数 | 主角 | 重要配角 | 普通配角 | 复杂画面 | 多人同框 |")
        add("|---|---|---|---|---|---|---|")
        for slug, bucket in dataset_stats["by_work"].items():
            add(
                f"| {slug} | {bucket.get('total', 0)} "
                f"| {bucket.get('main_character', 0)} "
                f"| {bucket.get('key_supporting_character', 0)} "
                f"| {bucket.get('supporting_character', 0)} "
                f"| {bucket.get('hard_case', 0)} "
                f"| {bucket.get('multi_person', 0)} |"
            )
    else:
        add("数据集尚未导出（待采集与人工标注）。")
    add()

    # ---- 总排行榜 ----
    add("## 2. 模型总体排行榜（核心指标：有效响应识别准确率，95% Wilson CI）")
    add()
    add("> 有效响应 = 计划样本 − API失败 − 未执行；拒答与格式错误均计入分母（模型行为）。")
    add("> 端到端正确率与 API 失败率见第 3 节双口径表。")
    add()
    add("| 模型 | 厂商/档位 | A组准确率 | B组准确率 | B−A 提升 | A/B 配对检验 |")
    add("|---|---|---|---|---|---|")
    def _rank_key(m: str) -> float:
        cell = result.metrics.get((m, "B")) or result.metrics.get((m, "A"))
        return -cell.accuracy_valid if cell and cell.planned else 0.0

    ranked = sorted(models, key=_rank_key)
    for m in ranked:
        info = models_info.get(m, {})
        a_cell = result.metrics.get((m, "A"))
        b_cell = result.metrics.get((m, "B"))
        vendor_tier = f"{info.get('vendor', '?')}/{info.get('tier', '?')}"
        a_txt = f"{_pct(a_cell.accuracy_valid)} {_ci(*a_cell.accuracy_valid_ci)}" if a_cell and a_cell.planned else "待测"
        b_txt = f"{_pct(b_cell.accuracy_valid)} {_ci(*b_cell.accuracy_valid_ci)}" if b_cell and b_cell.planned else "待测"
        if a_cell and b_cell and a_cell.planned and b_cell.planned:
            lift = f"{(b_cell.accuracy_valid - a_cell.accuracy_valid) * 100:+.1f}pp"
            test = result.ab_tests.get(m)
            p_txt = f"p={test['p_value']:.3f}（{test['method']}）" if test else "—"
        else:
            lift, p_txt = "待测", "—"
        add(f"| {info.get('display_name', m)} | {vendor_tier} | {a_txt} | {b_txt} | {lift} | {p_txt} |")
    add()

    # ---- 双口径 ----
    add("## 3. 双口径指标（调整 5：网络故障不得扭曲模型能力评价）")
    add()
    add("| 模型 | 实验 | 计划 | 有效响应 | 正确 | 端到端正确率 | 有效响应准确率 | API失败率 |")
    add("|---|---|---|---|---|---|---|---|")
    for m in models:
        for exp in ("A", "B"):
            cell = result.metrics.get((m, exp))
            if not cell or not cell.planned:
                add(f"| {models_info.get(m, {}).get('display_name', m)} | {exp} | 0 | — | — | 待测 | 待测 | — |")
                continue
            valid = cell.planned - cell.counts.get("api_failure", 0) - cell.counts.get("not_run", 0)
            add(
                f"| {models_info.get(m, {}).get('display_name', m)} | {exp} "
                f"| {cell.planned} | {valid} | {cell.counts.get('correct', 0)} "
                f"| {_pct(cell.accuracy_e2e)} | {_pct(cell.accuracy_valid)} "
                f"| {_pct(cell.api_failure_rate)} |"
            )
    add()

    # ---- 错误类型 ----
    add("## 4. 结果分类明细")
    add()
    kinds = ["correct", "wrong_name", "actor_name", "uncertain", "refused", "parse_error", "api_failure", "not_run"]
    add("| 模型 | 实验 | " + " | ".join(OUTCOME_LABELS[k] for k in kinds) + " |")
    add("|---|---|" + "---|" * len(kinds))
    for m in models:
        for exp in ("A", "B"):
            cell = result.metrics.get((m, exp))
            counts = cell.counts if cell else {}
            add(
                f"| {models_info.get(m, {}).get('display_name', m)} | {exp} | "
                + " | ".join(str(counts.get(k, 0)) for k in kinds)
                + " |"
            )
    add()

    # ---- 分作品 ----
    add("## 5. 分作品准确率（有效响应口径）")
    add()
    work_slugs: list[str] = []
    for cell_map in result.by_work.values():
        for w in cell_map:
            if w not in work_slugs:
                work_slugs.append(w)
    if work_slugs:
        add("| 模型 | " + " | ".join(work_slugs) + " |")
        add("|---|" + "---|" * len(work_slugs))
        for m in models:
            row = [models_info.get(m, {}).get("display_name", m)]
            for w in work_slugs:
                cell = result.by_work.get((m, "B"), {}).get(w)
                row.append(f"{_pct(cell.accuracy_valid)}" if cell and cell.planned else "待测")
            add("| " + " | ".join(row) + " |")
    else:
        add("待测。")
    add()

    # ---- 分维度 ----
    for title, field, order in (
        ("6. 分人物重要度准确率（B组）", "by_importance", ["main", "key_supporting", "supporting"]),
        ("7. 分视觉难度准确率（B组）", "by_difficulty", ["normal", "side_face", "occluded", "blurry", "costume_change"]),
    ):
        add(f"## {title}")
        add()
        slice_data: dict = getattr(result, field)
        keys = order or sorted({k for cm in slice_data.values() for k in cm})
        if any(slice_data.values()):
            add("| 模型 | " + " | ".join(keys) + " |")
            add("|---|" + "---|" * len(keys))
            for m in models:
                row = [models_info.get(m, {}).get("display_name", m)]
                for k in keys:
                    cell = slice_data.get((m, "B"), {}).get(k)
                    row.append(f"{_pct(cell.accuracy_valid)}" if cell and cell.planned else "—")
                add("| " + " | ".join(row) + " |")
        else:
            add("待测。")
        add()

    # ---- 分多人/单人 ----
    add("## 8. 多人同框 vs 单人画面（B组）")
    add()
    if any(result.by_multi_person.values()):
        add("| 模型 | " + " | ".join(sorted({k for cm in result.by_multi_person.values() for k in cm})) + " |")
        add("|---|---|---|")
        for m in models:
            keys = sorted({k for cm in result.by_multi_person.values() for k in cm})
            row = [models_info.get(m, {}).get("display_name", m)]
            for k in keys:
                cell = result.by_multi_person.get((m, "B"), {}).get(k)
                row.append(f"{_pct(cell.accuracy_valid)}" if cell and cell.planned else "—")
            add("| " + " | ".join(row) + " |")
    else:
        add("待测。")
    add()

    # ---- 按角色 ----
    add("## 9. 按角色统计（防高频角色掩盖短板）")
    add()
    any_char = any(v.get("characters") for v in result.per_character.values())
    if any_char:
        add("| 模型 | 实验 | 角色宏平均准确率 | 覆盖角色数 | 最弱角色（准确率最低的前 5） |")
        add("|---|---|---|---|---|")
        for m in models:
            for exp in ("A", "B"):
                pc = result.per_character.get((m, exp))
                if not pc or not pc.get("characters"):
                    continue
                chars = pc["characters"]
                worst = sorted(chars.items(), key=lambda kv: (kv[1]["accuracy"], kv[1]["n"]))[:5]
                worst_txt = "；".join(
                    f"{k.split('::')[-1]} {v['correct']}/{v['n']}" for k, v in worst
                )
                add(
                    f"| {models_info.get(m, {}).get('display_name', m)} | {exp} "
                    f"| {_pct(pc['macro_accuracy'])} | {len(chars)} | {worst_txt} |"
                )
    else:
        add("待测。")
    add()

    # ---- 同厂档位对比 ----
    add("## 10. 同厂商档位对比（tier 仅为项目分组标签，性能以实测为准）")
    add()
    vendors: dict[str, list[str]] = {}
    for m in models:
        vendors.setdefault(models_info.get(m, {}).get("vendor", "?"), []).append(m)
    if any(vendors.values()):
        add("| 厂商 | 模型 | 档位标签 | A组 | B组 |")
        add("|---|---|---|---|---|")
        for vendor in sorted(vendors):
            for m in vendors[vendor]:
                info = models_info.get(m, {})
                a_cell = result.metrics.get((m, "A"))
                b_cell = result.metrics.get((m, "B"))
                a_txt = _pct(a_cell.accuracy_valid) if a_cell and a_cell.planned else "待测"
                b_txt = _pct(b_cell.accuracy_valid) if b_cell and b_cell.planned else "待测"
                add(
                    f"| {vendor} | {info.get('display_name', m)} | {info.get('tier', '?')} "
                    f"| {a_txt} | {b_txt} |"
                )
    else:
        add("待测。")
    add()

    # ---- 图表 ----
    if chart_names:
        add("## 11. 可视化图表")
        add()
        for name in chart_names:
            add(f"![{name}]({name})")
        add()

    # ---- 公平性记录 ----
    add("## 12. 公平性与参数差异记录")
    add()
    fairness = (run_meta or {}).get("fairness_meta", {}) if run_meta else {}
    if isinstance(fairness, str):
        import json as _json

        try:
            fairness = _json.loads(fairness)
        except _json.JSONDecodeError:
            fairness = {"note": fairness}
    for key, value in (fairness or {}).items():
        if key == "per_model_notes" and isinstance(value, dict):
            add("### 各模型参数差异")
            add()
            for mid, note in value.items():
                add(f"- **{mid}**：`extra_body={note.get('extra_body')}`；"
                    f"传temperature={'是' if note.get('passes_temperature') else '否'}；"
                    f"{note.get('notes') or ''}")
        else:
            add(f"- **{key}**：{value}")
    add()

    # ---- 复核 ----
    add("## 13. 人工复核记录")
    add()
    if overrides:
        add("| 模型 | 实验 | 样本 | 改判为 | 备注 |")
        add("|---|---|---|---|---|")
        for ov in overrides:
            add(
                f"| {ov['model_id']} | {ov['experiment']} | {ov['sample_id']} "
                f"| {OUTCOME_LABELS.get(ov['outcome'], ov['outcome'])} | {ov['note']} |"
            )
    else:
        add("（暂无人工改判记录）")
    add()

    # ---- 结论 ----
    add("## 14. 实验结论")
    add()
    has_data = any(
        cell.planned and cell.counts.get("correct", 0) > 0 for cell in result.metrics.values()
    )
    if not has_data:
        add("> ⚠️ 本报告暂无真实实验数据，以下结论栏位保持「待测」。")
        add("> 项目红线：无真实结果不生成任何模型优劣结论。")
        add()
    conclusions: list[str] = []
    # 总体最佳（B组）
    best = None
    for m in models:
        cell = result.metrics.get((m, "B"))
        if cell and cell.planned:
            if best is None or cell.accuracy_valid > best[1]:
                best = (m, cell.accuracy_valid)
    if best:
        conclusions.append(
            f"- 总体（B组）最佳：{models_info.get(best[0], {}).get('display_name', best[0])}，"
            f"有效响应准确率 {_pct(best[1])}。"
        )
    # 分作品最佳
    if result.by_work:
        for w in sorted(work_slugs):
            best_w = None
            for m in models:
                cell = result.by_work.get((m, "B"), {}).get(w)
                if cell and cell.planned and (best_w is None or cell.accuracy_valid > best_w[1]):
                    best_w = (m, cell.accuracy_valid)
            if best_w:
                conclusions.append(
                    f"- {w} 最佳（B组）：{models_info.get(best_w[0], {}).get('display_name', best_w[0])} {_pct(best_w[1])}"
                )
    # A/B 提升结论
    sig = [
        (m, t)
        for m, t in result.ab_tests.items()
        if t.get("p_value") is not None and t["p_value"] < 0.05
    ]
    if result.ab_tests:
        avg_lift = sum(t["delta_pp"] for t in result.ab_tests.values()) / len(result.ab_tests)
        conclusions.append(
            f"- 作品名称平均影响：{avg_lift:+.1f}pp（{len(sig)}/{len(result.ab_tests)} 个模型达到统计显著 p<0.05）。"
        )
    if conclusions:
        lines.extend(conclusions)
    else:
        add("待真实实验数据后自动生成。")
    add()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
