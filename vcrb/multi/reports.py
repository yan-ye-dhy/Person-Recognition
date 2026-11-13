"""Portable v2 HTML/CSV/JSON reports with separate identification and C tasks."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .scoring import MultiScoreResult


def flat(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            result.update(flat(item, name + "_"))
        elif isinstance(item, list):
            result[name] = json.dumps(item, ensure_ascii=False)
        else:
            result[name] = item
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def rate(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "—"
    n, d, value = metric["numerator"], metric["denominator"], metric["value"]
    if value is None:
        return f"— ({n}/{d})"
    ci = metric.get("ci95")
    suffix = f"<small>95% CI {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%</small>" if ci else ""
    return f"{100 * value:.2f}% ({n}/{d}){suffix}"


def number(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else esc(value)


def table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="scroll"><table><thead><tr>' + ''.join(f"<th>{esc(h)}</th>" for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f"<td>{cell}</td>" for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def bars(items: list[tuple[str, float | None]], title: str, *, is_mock: bool = False) -> str:
    """Inline SVG, including explicit missing denominators instead of zero bars."""
    heading = title + (" · MOCK 合成演示" if is_mock else "")
    height = max(90, 58 + 34 * len(items))
    x, width = 300, 440
    parts = [f'<section class="chart-panel"><h3>{esc(heading)}</h3>',
             f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(heading)}" viewBox="0 0 900 {height}">',
             f'<title>{esc(heading)}</title><desc>同一任务内指标对比；无可评分GT的项目没有柱形，不作为零分。</desc>']
    for tick in (0, 25, 50, 75, 100):
        point = x + width * tick / 100
        parts.append(f'<line x1="{point}" y1="34" x2="{point}" y2="{height - 12}" stroke="#d8e2e9"/><text x="{point}" y="22" text-anchor="middle" font-size="13" fill="#526575">{tick}%</text>')
    if not items:
        parts.append('<text x="300" y="64" fill="#755b19">无可评分GT</text>')
    for index, (label, value) in enumerate(items):
        y = 48 + 34 * index
        parts.append(f'<text x="8" y="{y + 16}" font-size="14" fill="#172b3b">{esc(label)}</text>')
        if value is None:
            parts.append(f'<text class="missing-metric" x="{x + 6}" y="{y + 16}" font-size="14" fill="#755b19">无可评分GT</text>')
            continue
        color = "#278163" if "端到端" in label else "#dd8422" if label.endswith("|B") else "#2475ad"
        bar_width = width * max(0., min(1., value))
        parts.append(f'<rect class="metric-bar" data-label="{esc(label)}" x="{x}" y="{y}" width="{bar_width:.3f}" height="23" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + width + 12}" y="{y + 16}" font-size="14" fill="#172b3b">{100 * value:.2f}%</text>')
    parts.append('</svg></section>')
    return ''.join(parts)


def metric_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"model_id": key.rsplit("|", 1)[0], **flat(value)} for key, value in sorted(metrics.items())]


def render_html(result: MultiScoreResult, path: Path, run_meta: dict[str, Any] | None = None) -> None:
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>多人物 A/B/C 评测</title><style>body{font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#172b3b;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #d5dfe5;padding:8px;vertical-align:top}small{display:block;color:#526575}h2{margin-top:34px}.scroll{overflow:auto}.notice{background:#fff2c6;padding:16px;border:2px solid #d6a23c}code{overflow-wrap:anywhere}.chart-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.chart-panel{background:white;border:1px solid #d5dfe5;padding:12px;margin:16px 0}.chart-panel svg{width:100%;height:auto;display:block}@media(max-width:500px){.chart-pair{grid-template-columns:1fr}}</style>',
             f'<h1>多人物 A/B/C 评测 · {esc(result.run_id)}</h1>']
    if result.is_mock:
        parts.append('<div class="notice"><strong>MOCK / 合成响应演示：不是任何真实模型成绩。</strong>本报告仅验证多人物代码链路，不代表真实API调用、模型能力或真实费用。</div>')
    else:
        status = (run_meta or {}).get("status")
        planned = sum(v.get("planned_images", 0) for v in result.metrics.values())
        not_run = sum(v.get("not_run", 0) for v in result.metrics.values())
        if not_run or (status and status not in {"done", "completed"}):
            parts.append(
                '<div class="notice"><strong>真实评测进行中 / 未全执行：非最终结论。</strong>'
                f'当前状态：{esc(status or "部分任务未执行")}；已保存响应 {planned - not_run} / {planned}，'
                f'尚未执行 {not_run}。当前数据仅作过程记录，不能据此形成最终模型优劣结论；'
                'API失败照实列出，未执行任务仍保留在全部计划口径的分母中。</div>'
            )
    if result.quality["identity_counts"].get("pending", 0):
        parts.append('<div class="notice">仍有待复核身份；未核实名字没有作为真值。迁移样本不等于已完成多人物标注，正式付费评测须另行通过标注门禁并获用户确认。</div>')
    parts.append('<p><a href="summary.json">完整JSON</a> · <a href="metrics_ab.csv">A/B指标CSV</a> · <a href="metrics_c.csv">C指标CSV</a> · <a href="detail.csv">逐人物CSV</a> · <a href="images.csv">逐图片CSV</a> · <a href="by_work.csv">分作品</a> · <a href="by_person_count.csv">分人数</a> · <a href="by_face_orientation.csv">按人物朝向</a> · <a href="by_visibility.csv">按人物可见性</a></p>')
    parts.append(f'<p>数据版本 <code>{esc(result.dataset_hash)}</code>；图片 {result.quality["samples"]}，人物框 {result.quality["persons"]}，身份状态 {esc(result.quality["identity_counts"])}；前景标注完整图片 {result.quality["foreground_complete_samples"]}。</p>')
    parts.append('<h2>A/B：指定编号人物身份识别</h2><p>A无作品名，B提供作品名；两组复用同一编号红框图。人物准确率仅以verified且approved身份为真值，HTTP成功但格式错误/拒答仍留在有效分母。端到端口径把API失败和未执行也留在分母。未知身份只单列null、猜测与漏答；pending不计普通姓名真值。</p>')
    ab = [(key, value) for key, value in sorted(result.metrics.items()) if value["experiment"] in {"A", "B"}]
    parts.append(table(["模型", "实验", "人物准确率（有效响应）", "人物准确率（全部计划）", "图片完全正确", "unknown null / 猜测 / 缺失错误", "格式错误率 / 拒答率"], [
        [esc(key.rsplit('|', 1)[0]), value["experiment"], rate(value["person_accuracy"]), rate(value["person_accuracy_e2e"]),
         rate(value["image_exact_accuracy"]), f'{value["unknown_null"]} / {value["unknown_wrong_guess"]} / {value["unknown_missing_or_error"]}',
         rate(value["format_error_rate"]) + ' / ' + rate(value["refusal_rate"])] for key, value in ab]))
    parts.append('<div class="chart-pair">' + bars([(key, value["person_accuracy"]["value"]) for key, value in ab], "A/B人物准确率（与C任务分开）", is_mock=result.is_mock)
                 + bars([(key, value["image_exact_accuracy"]["value"]) for key, value in ab], "A/B图片完全正确率", is_mock=result.is_mock) + '</div>')
    parts.append('<h3>作品名称带来的A/B配对差异</h3>')
    parts.append(table(["模型", "B−A人物准确率", "配对核实人物数", "McNemar端到端p值"], [
        [esc(model), number(value["person_accuracy_delta_B_minus_A"]), str(value["paired_verified_targets"]),
         number(value["mcnemar_e2e"]["p_value"])] for model, value in sorted(result.ab_effects.items())]))
    parts.append('<h2>C：自主发现、定位及识别</h2><p>C只收无框原图与作品名，不提供GT框、数量或编号。以下主定位与端到端P/R/F1仅纳入foreground_complete图片；不完整前景标注另外给诊断计数，不能把漏标背景自动算误检。最大匹配先最大化匹配数再最大化IoU，姓名不参与配对；一个预测和一个真值均最多匹配一次。</p>')
    cs = [(key, value) for key, value in sorted(result.metrics.items()) if value["experiment"] == "C"]
    parts.append(table(["模型", "定位P / R / F1", "匹配后身份准确率", "端到端P / R / F1", "图片完全正确", "主定位完整图 / 全部图", "完整图漏检 / 误检 / 身份错", "unknown null / 猜测 / 缺失错误", "ignore抑制 / 未评背景"], [
        [esc(key.rsplit('|', 1)[0]), rate(value["localization"]["precision"]) + ' / ' + rate(value["localization"]["recall"]) + '<br>F1 ' + number(value["localization"]["f1"]),
         rate(value["matched_identity_accuracy"]), rate(value["end_to_end"]["precision"]) + ' / ' + rate(value["end_to_end"]["recall"]) + '<br>F1 ' + number(value["end_to_end"]["f1"]),
         rate(value["image_exact_accuracy"]), f'{value["localization_eligible_images"]} / {value["planned_images"]}',
         f'{value["localization"]["fn"]} / {value["localization"]["fp"]} / {value["identity_errors"]}',
         f'{value["unknown_null"]} / {value["unknown_wrong_guess"]} / {value["unknown_missing_or_error"]}',
         f'{value["ignored_predictions"]} / {value["unassessed_predictions"]}'] for key, value in cs]))
    parts.append(bars([(key + ' 定位F1', value["localization"]["f1"]) for key, value in cs] + [(key + ' 端到端F1', value["end_to_end"]["f1"]) for key, value in cs], 'C任务F1；不与A/B准确率混合排名', is_mock=result.is_mock))
    parts.append('<h2>可用性、延迟、usage与费用</h2><p>延迟单位毫秒，P50/P95依据有记录的实际请求耗时；限频排队/退避是否计入由适配器记录口径决定。配置单价且有输入/输出usage时按每百万token价格估算CNY，此估算不是完整账单。未提供价格、usage或费用的记录显示缺失，不以0冒充免费。</p>')
    parts.append(table(["模型/实验", "API成功", "有效 / 已格式化 / 计划图片", "P50 / P95 ms（记录数）", "usage（记录数）", "已提供费用（记录数）", "API失败原因"], [
        [esc(key), rate(v["api_success_rate"]), f'{v["valid_responses"]} / {v["formatted_responses"]} / {v["planned_images"]}',
         f'{number(v["latency_ms_p50"])} / {number(v["latency_ms_p95"])} ({v["latency_records"]})',
         esc(v["usage"]) + f' ({v["usage_records"]})', esc(v["estimated_cost_by_currency"]) + f' ({v["cost_records"]})',
         esc(v["error_types"])] for key, v in sorted(result.metrics.items())]))
    for heading, data in [("分作品", result.by_work), ("单人/多人", result.by_single_multi), ("1/2/3/4+人物", result.by_person_count)]:
        parts.append(f'<h2>{heading}</h2>')
        parts.append(table(["模型/实验", "分组", "任务", "已核实人物", "A/B人物准确率", "C定位F1", "C端到端F1", "图片完全正确"], [
            [esc(key), esc(group), esc(v["task"]), str(v["verified_targets"]), rate(v.get("person_accuracy")),
             number(v.get("localization", {}).get("f1")), number(v.get("end_to_end", {}).get("f1")), rate(v["image_exact_accuracy"])]
            for key, groups in sorted(data.items()) for group, v in sorted(groups.items())]))
    parts.append('<h2>按人物朝向与可见性分组</h2>')
    if not result.by_face_orientation:
        parts.append('<p>该运行没有冻结的分组标签，未自动借用当前工作区的新标签。旧运行指标照常保留。</p>')
    else:
        parts.append('<p>以下以真值人物为统计单位；同一张图可含不同朝向，因此不同组的涉及图片数不可相加。'
                     'A/B分别列HTTP成功口径及全部计划口径；C按完整标注图中的真值人物统计定位召回、匹配后身份准确率及端到端召回。'
                     'API失败和未执行目标仍留在全部计划分母。未匹配的额外预测没有真值朝向，不强行分组；各组不计算定位精确率或整图全对率。'
                     '整图全对率继续在单人/多人和作品分组中呈现。少量样本的组仅供参考；分组标签不发送给模型。</p>')
        parts.append('<p>分组标签状态：' + esc(result.stratification.get("status")) + '；内容哈希 <code>'
                     + esc(result.stratification.get("content_hash")) + '</code>。</p>')
        labels = {"front": "正脸", "three_quarter": "斜侧脸", "profile": "侧脸", "not_visible": "面部不可见/无法判断",
                  "clear": "清晰无遮挡", "occluded_or_blurry": "明显遮挡/模糊"}
        for heading, data in (("人物朝向", result.by_face_orientation), ("人物可见性", result.by_visibility)):
            parts.append(f'<h3>{heading}</h3>')
            parts.append(table(["模型/实验", "分组", "涉及图片 / 人物 / 核实身份", "API失败 / 未执行人物", "格式错误 / 拒答人物",
                                "A/B人物准确率（HTTP成功）", "A/B人物准确率（全部计划）", "C定位召回", "C匹配后身份准确率", "C端到端召回"], [
                [esc(key), esc(labels.get(group, group)), f'{v["sample_count"]} / {v["person_targets"]} / {v["verified_targets"]}',
                 f'{v["api_failure_persons"]} / {v["not_run_persons"]}', f'{v["format_error_persons"]} / {v["refused_persons"]}',
                 rate(v.get("person_accuracy")), rate(v.get("person_accuracy_e2e")), rate(v.get("localization_recall")),
                 rate(v.get("matched_identity_accuracy")), rate(v.get("end_to_end_recall"))]
                for key, groups in sorted(data.items()) for group, v in sorted(groups.items())]))
    parts.append('<h2>错误与待核示例</h2><p>以下最多展示200条；完整逐人物结果见detail.csv。unknown与pending单独标识，不把模型多数意见作为真值。</p>')
    cases = [[esc(r["model_id"]), esc(r["experiment"]), esc(r["sample_id"]), esc(p.get("person_id")),
              esc(p.get("identity_category")), esc(p.get("gt_character")), esc(p.get("predicted")), esc(p["outcome"])]
             for r in result.records for p in r["person_details"] if p["outcome"] not in {"correct", "unknown_null", "ignored_prediction"}][:200]
    parts.append(table(["模型", "实验", "图片", "编号", "真值状态", "核实角色", "预测", "结果"], cases))
    parts.append('<h2>分母、匹配与置信区间</h2><ul>')
    for key, value in result.policy.items():
        parts.append(f'<li><strong>{esc(key)}</strong>：{esc(value)}</li>')
    parts.append('<li>图片完全正确率只纳入无pending身份的图片；C另须前景完整。未知人物需明确返回null，额外编号/误检/漏检/格式错误均使图片不完全正确。</li><li>匹配后身份准确率以几何匹配成功的已核实人物为分母；C端到端只以已核实身份为姓名真值，匹配到unknown/pending的预测从姓名分母剔除。</li><li>95%区间为Wilson二项区间，未校正同角色或同图片内相关性；F1非二项比例，不伪造Wilson区间。零分母显示缺失。</li></ul>')
    if run_meta:
        metadata = run_meta.get("fairness_meta", {})
        selected = {k: metadata[k] for k in ("generation_params", "image_encoding", "benchmark_protocol", "is_mock", "stratification") if k in metadata}
        parts.append('<h2>运行参数</h2><pre>' + esc(json.dumps(selected, ensure_ascii=False, indent=2)) + '</pre>')
    parts.append('</html>')
    path.write_text(''.join(parts), encoding="utf-8")


def save_reports(result: MultiScoreResult, out_dir: Path, run_meta: dict[str, Any] | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    if "2026-11-13" in out_dir.resolve().parts:
        raise ValueError("Multi-v2 reports cannot overwrite the published v1 report")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {name: out_dir / name for name in ("summary.json", "metrics.csv", "metrics_ab.csv", "metrics_c.csv", "detail.csv", "images.csv", "by_work.csv", "by_person_count.csv", "by_single_multi.csv", "by_face_orientation.csv", "by_visibility.csv", "ab_effects.csv", "index.html")}
    files["summary.json"].write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    all_metrics = metric_rows(result.metrics)
    write_csv(files["metrics.csv"], all_metrics)
    write_csv(files["metrics_ab.csv"], [r for r in all_metrics if r["experiment"] in {"A", "B"}])
    write_csv(files["metrics_c.csv"], [r for r in all_metrics if r["experiment"] == "C"])
    details, images = [], []
    for row in result.records:
        images.append(flat({k: v for k, v in row.items() if k != "person_details"}))
        for person in row["person_details"]:
            details.append(flat({"run_id": result.run_id, "model_id": row["model_id"], "experiment": row["experiment"],
                                 "sample_id": row["sample_id"], "work_slug": row["work_slug"], **person}))
    write_csv(files["detail.csv"], details)
    write_csv(files["images.csv"], images)
    for name, grouped in (("by_work.csv", result.by_work), ("by_person_count.csv", result.by_person_count), ("by_single_multi.csv", result.by_single_multi),
                          ("by_face_orientation.csv", result.by_face_orientation), ("by_visibility.csv", result.by_visibility)):
        write_csv(files[name], [{"model_id": key.rsplit("|", 1)[0], "group": group, **flat(metric)}
                               for key, groups in sorted(grouped.items()) for group, metric in sorted(groups.items())])
    write_csv(files["ab_effects.csv"], [{"model_id": model, **flat(value)} for model, value in sorted(result.ab_effects.items())])
    render_html(result, files["index.html"], run_meta)
    return files
