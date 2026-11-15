"""Export offline, per-image Qwen Max error evidence without rescoring or API calls.

Run with: .venv/bin/python scripts/export_qwenmax_review.py
Original run snapshots stay unchanged; explicit user corrections are applied as a review overlay.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcrb.evaluation.parser import extract_json_text
from vcrb.multi.parsing import bbox_valid, parse_response
from vcrb.multi.identity_review import build_identity_review

RUN_ID = "multi_v2_20261113_075729_2e4f15"
MODEL_ID = "qwen3.8-max"
DEFAULT_OUTPUT = ROOT / "reports/2026-11-15-qwenmax-review"
LABELS = {
    "correct": "识别正确", "wrong_name": "角色姓名错误", "uncertain": "未确定身份（null）",
    "actor_name": "回答演员姓名", "missing_id": "漏答编号", "extra_id": "多答编号",
    "missed_detection": "漏检主要人物", "false_positive": "额外误检",
    "parse_error": "格式不合法，未正常匹配", "ignored_prediction": "背景预测已忽略",
    "unassessed_prediction": "未评估预测", "unknown_null": "未知身份正确保留null",
    "unknown_guess": "对未知身份猜测姓名", "pending_unscored": "待确认身份未评分",
    "api_failure": "接口失败", "refused": "拒绝回答", "not_run": "尚未执行",
}
POSE = {"front": "正脸", "three_quarter": "斜侧脸", "profile": "侧脸", "not_visible": "面部不可见"}
VISIBILITY = {"clear": "清晰", "occluded_or_blurry": "遮挡或模糊"}
SUCCESS = {"correct", "unknown_null", "ignored_prediction", "pending_unscored", "unassessed_prediction"}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + \
        '</tbody></table></div>'


def raw_persons(text: str) -> list[dict]:
    try:
        data = json.loads(extract_json_text(text))
    except (ValueError, TypeError):
        return []
    return data.get("persons", []) if isinstance(data, dict) and isinstance(data.get("persons"), list) else []


def bbox_issue(box: Any) -> str:
    if bbox_valid(box):
        return "坐标本身符合0–1约束"
    if not isinstance(box, list) or len(box) != 4:
        return "bbox应为含4个数值的数组"
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in box):
        return "bbox含非数值坐标"
    if any(v < 0 or v > 1 for v in box):
        ordering = "，且坐标顺序无效" if box[0] >= box[2] or box[1] >= box[3] else ""
        return f"坐标超出0–1范围{ordering}；本次要求归一化坐标，原评测未自动换算"
    return "坐标顺序不符合x1<x2、y1<y2"


def explain(detail: dict, experiment: str) -> str:
    outcome, gt, predicted = detail["outcome"], detail.get("gt_character"), detail.get("predicted")
    if outcome == "wrong_name":
        prefix = "位置已匹配，" if experiment == "C" else ""
        return f'{prefix}应为“{gt}”，回答“{predicted}”；未命中当次可接受别名。'
    if outcome == "correct":
        return "该人物识别正确；本图其他人物或输出格式存在问题。"
    if outcome == "uncertain":
        return f'应识别“{gt}”，模型返回null（无法确定）。'
    if outcome == "parse_error":
        return "整条响应格式不合法，原评测没有对该人物正常匹配或判定姓名；请结合下面原始条目查看。"
    if outcome == "actor_name":
        return f'应回答角色“{gt}”，实际回答演员姓名“{predicted}”。'
    if outcome == "ignored_prediction":
        return "落入原标注的背景忽略区，未算误检，也不影响本图是否全对。"
    return {
        "missing_id": "模型遗漏图片中已有编号。", "extra_id": "模型增加了图片中不存在的编号。",
        "missed_detection": "没有合法预测框与该主要人物匹配（IoU门槛0.5）。",
        "false_positive": "多出的预测未匹配主要人物，也没有被背景忽略区抑制。",
        "unknown_null": "该人物真值身份未知，模型保留null。",
        "unknown_guess": "真值身份未知，模型仍给出了确定姓名。",
        "pending_unscored": "身份尚待确认，不作为已核实名字评分。",
        "unassessed_prediction": "前景标注不完整，该额外预测未进入主定位评分。",
    }.get(outcome, LABELS.get(outcome, outcome))


def categories(row: dict) -> list[str]:
    if not row["parse_ok"]:
        return ["parse_error"]
    result = {p["outcome"] for p in row["person_details"] if p["outcome"] not in SUCCESS}
    if row.get("issues") and not result:
        result.add("format_issue")
    return sorted(result)


def stats(rows: list[dict]) -> dict:
    failed = [r for r in rows if r["image_correct"] is False]
    valid = [r for r in rows if r["parse_ok"] and r["api_state"] == "ok" and not r["refused"]]
    return {
        "images": len(rows), "error_images": len(failed),
        "fully_correct_images": sum(r["image_correct"] is True for r in rows),
        "invalid_format_images": sum(not r["parse_ok"] for r in rows),
        "valid_error_images": sum(r["parse_ok"] for r in failed),
        "verified_targets": sum(r["verified_count"] for r in rows),
        "correct_targets": sum(r["identity_correct"] for r in rows),
        "matched_verified_targets": sum(r["matched_verified"] for r in rows),
        "person_outcomes": dict(Counter(p["outcome"] for r in rows for p in r["person_details"] if p["identity_category"] != "extra")),
        "error_image_categories_nonexclusive": dict(Counter(c for r in failed for c in categories(r))),
        "legal_output_localization": {suffix: sum(r[f"localization_{suffix}"] for r in valid) for suffix in ("tp", "fp", "fn")},
        "ignored_background_predictions": sum(r["ignored_predictions"] for r in rows),
    }


def overlay(sample: dict, row: dict, predictions: list[dict], image_size: list[int]) -> str:
    # SVG is a human review layer; the embedded JPEG bytes are the unmarked API input.
    height = 1000 * image_size[1] / image_size[0]
    parts = [f'<svg class="overlay" viewBox="0 0 1000 {height:.6f}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">']
    def box(bbox, label, color, cls, dashed=False, lower=False):
        x1, y1, x2, y2 = bbox[0] * 1000, bbox[1] * height, bbox[2] * 1000, bbox[3] * height
        label_width = max(48, len(label) * 18 + 12)
        lx = max(0, min(x1, 1000 - label_width))
        ly = max(0, min(y2 - 28 if lower else y1, height - 28))
        dash = ' stroke-dasharray="9 5"' if dashed else ''
        return f'<g class="{cls}"><rect x="{x1:.3f}" y="{y1:.3f}" width="{x2-x1:.3f}" height="{y2-y1:.3f}" fill="none" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke"{dash}/><rect x="{lx:.3f}" y="{ly:.3f}" width="{label_width}" height="28" fill="{color}"/><text x="{lx+5:.3f}" y="{ly+21:.3f}" fill="white" font-size="18" font-weight="700">{esc(label)}</text></g>'
    for p in sample["persons"]:
        parts.append(box(p["bbox"], f'真值{p["person_id"]}', '#087f5b', 'gt-box'))
    if row["parse_ok"] and not row["refused"]:
        outcomes = {p["prediction_index"]: p["outcome"] for p in row["person_details"] if p.get("prediction_index") is not None}
        for i, p in enumerate(predictions):
            ignored = outcomes.get(i) == "ignored_prediction"
            if bbox_valid(p.get("bbox")):
                parts.append(box(p["bbox"], f'预测{i+1}' + ('·忽略' if ignored else ''), '#64748b' if ignored else '#b45309', 'pred-box', True, True))
    parts.append('</svg>')
    return ''.join(parts)


CSS = """
:root{color-scheme:light;--ink:#182e40;--muted:#566b7c;--border:#d4e0e8;--accent:#075d86}*{box-sizing:border-box}body{margin:0;background:#edf3f7;color:var(--ink);font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1320px;margin:30px auto;padding:0 22px}h1{font-size:29px;line-height:1.35;margin:0 0 12px}h2{font-size:22px;margin:0 0 12px}h3{font-size:19px;margin:0}p{margin:10px 0}small,.muted{color:var(--muted)}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f9;padding:15px;border-radius:8px;font:14px/1.6 ui-monospace,monospace;max-height:520px;overflow:auto}.intro,.panel,.case{border:1px solid var(--border);background:white;border-radius:14px;padding:22px;margin-bottom:20px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.kpi{padding:14px 16px;background:#f0f6fa;border-radius:10px}.kpi b{display:block;font-size:29px}.kpi span{color:var(--muted);font-size:14px}.notice{border-left:4px solid #d97706;background:#fff8e9;padding:13px 17px;margin:14px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 12px;border:1px solid var(--border);vertical-align:top;text-align:left}th{background:#eff5f9;white-space:nowrap}td small{display:block}td pre{margin:0;min-width:150px;padding:6px;font-size:12px}.toolbar{position:sticky;top:0;z-index:5;background:#f7fafdf5;border:1px solid var(--border);border-radius:10px;padding:14px;margin:24px 0;display:flex;gap:12px;flex-wrap:wrap;align-items:end}.toolbar label{display:flex;flex-direction:column;font-size:13px;gap:4px}input,select,button{font:inherit;border:1px solid #b7cbd8;border-radius:7px;padding:8px 12px;color:var(--ink);background:white}input[type=search]{min-width:270px}button{cursor:pointer}button:hover{border-color:var(--accent);color:var(--accent)}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #65a9d8;outline-offset:3px}.case-head{display:flex;gap:12px;justify-content:space-between;align-items:start;flex-wrap:wrap}.tags{display:flex;gap:8px;flex-wrap:wrap}.tag{display:inline-block;border-radius:20px;padding:3px 11px;background:#fff1e5;color:#92400e;font-size:13px}.bad{color:#a42e22;font-weight:600}.good{color:#087f5b;font-weight:600}.neutral{color:var(--muted)}.picture-controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:13px 0}.picture-controls label{cursor:pointer}.picture-controls input{accent-color:#087f5b;margin-right:5px}.gt-legend{color:#087f5b}.pred-legend{color:#b45309}.image-wrap{position:relative;background:#13202a;line-height:0;border-radius:8px;overflow:hidden}.input-image{display:block;width:100%;height:auto}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.hide-gt .gt-box,.hide-pred .pred-box{display:none}.caption{font-size:13px;color:var(--muted);margin:8px 0 15px}summary{cursor:pointer;color:var(--accent);padding:10px 0;font-weight:600}details{margin-top:10px}details details{margin-left:8px}.case [data-person-outcome]{margin-top:14px}.case[hidden]{display:none}#empty{padding:25px;text-align:center}dialog{max-width:96vw;width:1500px;max-height:95vh;border:0;border-radius:12px;padding:15px;box-shadow:0 20px 90px #0008}dialog::backdrop{background:#0c1727c9}.dialog-head{display:flex;justify-content:space-between;align-items:center;gap:20px;padding-bottom:12px}.dialog-content{overflow:auto;max-height:80vh}.dialog-content .image-wrap{width:100%}a{color:var(--accent)}.top-link{display:inline-block;margin:15px 0}footer{margin:25px 0 50px;font-size:13px;color:var(--muted)}@media(max-width:700px){main{padding:0 12px}.intro,.panel,.case{padding:15px}.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:24px}.toolbar{position:static}input[type=search]{min-width:190px}}@media print{body{background:white}.toolbar,.picture-controls,dialog,.top-link{display:none}.case{break-inside:avoid}details{display:block}.case[hidden]{display:none}}
"""

JS = """
const cards = [...document.querySelectorAll('.case')];
const search = document.getElementById('search');
const work = document.getElementById('work');
const type = document.getElementById('error-type');
function filterCards(){
 let count = 0;
 for(const card of cards){
   const shown = (!work.value || card.dataset.work === work.value) &&
     (!type.value || card.dataset.categories.split(' ').includes(type.value)) &&
     (!search.value.trim() || card.dataset.search.includes(search.value.trim().toLocaleLowerCase()));
   card.hidden = !shown; count += Number(shown);
 }
 document.getElementById('visible-count').textContent = `显示 ${count} / ${cards.length} 张`;
 document.getElementById('empty').hidden = count !== 0;
}
search.addEventListener('input', filterCards);
work.addEventListener('change', filterCards);
type.addEventListener('change', filterCards);
document.getElementById('reset').addEventListener('click',()=>{search.value='';work.value='';type.value='';filterCards();});
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{
 input.closest('.case').querySelector('.image-wrap').classList.toggle(`hide-${input.dataset.layer}`,!input.checked);
}));
const dialog = document.getElementById('zoom-dialog');
document.querySelectorAll('.zoom').forEach(button=>button.addEventListener('click',()=>{
 const card = button.closest('.case');
 document.getElementById('zoom-title').textContent = card.querySelector('h2').textContent;
 const content = document.getElementById('zoom-content'); content.replaceChildren(card.querySelector('.image-wrap').cloneNode(true));
 dialog.showModal();
}));
document.getElementById('close-zoom').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
filterCards();
"""


def detail_table(sample: dict, row: dict) -> str:
    persons = {p["person_id"]: p for p in sample["persons"]}
    rows = []
    for p in row["person_details"]:
        gt = persons.get(p.get("person_id"), {})
        aliases = gt.get("character_aliases", [])
        gt_name = esc(p.get("gt_character"))
        if aliases:
            gt_name += '<small>可接受别名：' + esc('、'.join(aliases)) + '</small>'
        outcome = p["outcome"]
        predicted = "未正常匹配；原始姓名见下表" if outcome == "parse_error" else "null" if p.get("predicted") is None else p["predicted"]
        css = "good" if outcome in {"correct", "unknown_null"} else "neutral" if outcome in SUCCESS else "bad"
        ident = f'真值{p["person_id"]}' if p.get("person_id") is not None else '额外预测'
        if p.get("prediction_index") is not None:
            ident += f'<small>对应预测{p["prediction_index"]+1}</small>'
        values = [ident, gt_name, esc(predicted), f'<span class="{css}">{esc(LABELS.get(outcome,outcome))}</span>',
                  esc(explain(p, row["experiment"])), esc(POSE.get(p.get("face_orientation"), "—")) + '<small>' + esc(VISIBILITY.get(p.get("visibility"), "—")) + '</small>']
        if row["experiment"] == "C":
            values.append(f'{p["iou"]:.4f}' if p.get("iou") is not None else '—')
        rows.append(values)
    headers = ["人物对应", "复核真值角色", "模型回答", "复核后判定", "具体说明", "朝向 / 可见性"]
    if row["experiment"] == "C":
        headers.append("匹配IoU")
    return '<div data-person-outcome>' + table(headers, rows) + '</div>'


def prediction_table(row: dict, predictions: list[dict]) -> str:
    if row["experiment"] != "C":
        return ''
    mapped = {p["prediction_index"]: p for p in row["person_details"] if p.get("prediction_index") is not None}
    rows = []
    for i, p in enumerate(predictions):
        if not isinstance(p, dict):
            rows.append([str(i+1), esc(p), '—', '条目不是对象'])
            continue
        box = p.get("bbox")
        if not row["parse_ok"]:
            explanation = bbox_issue(box) + '；整条响应未进入正常匹配，姓名字段仅作原文展示。'
        else:
            matched = mapped.get(i, {})
            explanation = f'匹配真值人物{matched["person_id"]}' if matched.get("person_id") is not None else LABELS.get(matched.get("outcome"), "未匹配")
        rows.append([str(i+1), esc(p.get("character") if p.get("character") is not None else 'null'),
                     '<code>' + esc(json.dumps(box, ensure_ascii=False)) + '</code>', esc(explanation)])
    return '<details class="predictions"' + (' open' if not row["parse_ok"] else '') + '><summary>逐条预测原文与坐标（' + str(len(predictions)) + '条）</summary>' + table(["预测序号", "原始姓名字段", "原始bbox", "格式与匹配说明"], rows) + '</details>'


def render_case(index: int, sample: dict, row: dict, raw: dict, image_data: bytes) -> str:
    predictions = raw_persons(raw["response"]["raw_text"])
    cats = categories(row)
    search_terms = [sample["sample_id"], sample["work_name"]] + [str(p.get(k) or '') for p in row["person_details"] for k in ("gt_character", "predicted")] + [str(p.get("character") or '') for p in predictions if isinstance(p, dict)]
    parts = [f'<article class="case" id="{esc(sample["sample_id"])}" data-work="{esc(sample["work_slug"])}" data-categories="{esc(" ".join(cats))}" data-search="{esc(" ".join(search_terms).lower())}" data-parse-ok="{str(row["parse_ok"]).lower()}">',
             f'<div class="case-head"><div><h2>{index:03d} · {esc(sample["sample_id"])} <small>{esc(sample["work_name"])}</small></h2><p class="muted">主要人物 {row["person_count"]} 人 · 正确 {row["identity_correct"]} 人 · 该图未整图全对</p></div><div class="tags">',
             ''.join(f'<span class="tag">{esc(LABELS.get(c,c))}</span>' for c in cats), '</div></div>']
    parts.append('<div class="picture-controls"><button class="zoom" type="button">放大图片</button>')
    if row["experiment"] == "C":
        parts.append('<label class="gt-legend"><input type="checkbox" data-layer="gt" checked>绿色实线：真值框</label><label class="pred-legend"><input type="checkbox" data-layer="pred" checked>橙色虚线：合法预测框</label>')
    parts.append('</div><div class="image-wrap"><img class="input-image" loading="lazy" decoding="async" alt="' + esc(sample["sample_id"]) + ' · 当次API输入图片" src="data:image/jpeg;base64,' + base64.b64encode(image_data).decode('ascii') + '">')
    if row["experiment"] == "C":
        parts.append(overlay(sample, row, predictions, raw["request"]["image_meta"]["encoded_size"]))
    parts.append('</div><p class="caption">')
    parts.append('底图为当次实际提交的无框JPEG；叠加框仅供本报告人工复核，未发送给模型。灰框代表被忽略的背景预测。' if row["experiment"] == "C" else '上图为当次实际提交的编号红框JPEG；作品名通过文字提示提供，答案没有提供给模型。')
    parts.append('</p>')
    if not row["parse_ok"]:
        parts.append('<div class="notice"><strong>格式失败：</strong>本图原始坐标不符合0–1归一化要求。原评测整条响应判为格式失败，未自动除以1000；姓名字段即使写出，也不能据此断言姓名识别错误。<br>解析问题：<code>' + esc('；'.join(row["issues"])) + '</code></div>')
    parts.append(detail_table(sample, row))
    parts.append(prediction_table(row, predictions))
    parts.append('<details class="raw-response"><summary>查看模型原始回答</summary><pre>' + esc(raw["response"]["raw_text"]) + '</pre></details>')
    request = raw["request"]
    settings = {k: request.get(k) for k in ("model", "temperature", "max_tokens", "max_completion_tokens", "extra_body", "tools") if k in request}
    settings.update({k: raw["response"].get(k) for k in ("served_model", "http_status", "finish_reason")})
    parts.append('<details class="request"><summary>当次提示词与调用参数</summary><h3>System</h3><pre>' + esc(request["system"]) + '</pre><h3>User</h3><pre>' + esc(request["user"]) + '</pre><h3>参数与图片指纹</h3><pre>' + esc(dump({**settings,"image_meta":request["image_meta"]})) + '</pre></details>')
    parts.append('<a class="top-link" href="#top">返回筛选栏 ↑</a></article>')
    return ''.join(parts)


def render_corrections_notice(corrections: list[dict]) -> str:
    if not corrections:
        return ''
    items=''.join('<li><code>'+esc(c['sample_id'])+'</code> 人物'+str(c['person_id'])+'：'+esc(c['old_character'])+' → '+esc(c['new_character'])+'</li>' for c in corrections)
    return '<div class="notice"><strong>已应用用户确认的标注更正：</strong><ul>'+items+'</ul>仅重新统计受影响的目标，原始图片、框、回答和历史评分记录保留；当前列表按更正后的身份判定。</div>'


def render_report(experiment: str, rows: list[dict], samples: dict, raw_records: dict,
                  gallery: dict, gallery_dir: Path, summary: dict) -> tuple[str, dict]:
    failed = sorted([r for r in rows if r["image_correct"] is False], key=lambda r: (list(gallery).index(r["sample_id"])))
    totals = stats(rows)
    works = dict.fromkeys((r["work_slug"], samples[r["sample_id"]]["work_name"]) for r in rows)
    title = f'Qwen Max · {experiment}组逐图错误复核'
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title><style>' + CSS + '</style></head><body><main>',
             '<header class="intro"><h1>' + esc(title) + '</h1><p>' + ('编号红框 + 提供作品名，识别每个编号的角色。' if experiment == "B" else '无框原图 + 提供作品名，自主定位并识别主要角色。') + '</p><p>本文件列全本组未整图全对的图片，每张展示全部已标主要人物及对应错误；直接双击即可离线查看。</p>',
             '<div class="kpis">' + ''.join('<div class="kpi"><b>' + esc(v) + '</b><span>' + esc(label) + '</span></div>' for v, label in [
                 (f'{totals["error_images"]} / {totals["images"]}', '错误图片 / 全部图片'),
                 (totals['invalid_format_images'], '格式不合法图片'),
                 (totals['valid_error_images'], '合法输出但未整图全对'),
                 (f'{totals["correct_targets"]} / {totals["verified_targets"]}', '全部目标中识别正确人物')]) + '</div>',
             '<p class="muted">图片数与人物数分别统计；多人图仅一个人物出错，整图也不算全对。' + ('C组“全部目标中正确人物”含格式失败影响；匹配后身份准确率的分母另为' + str(totals["matched_verified_targets"]) + '人。' if experiment == "C" else '') + '</p>',
             render_corrections_notice(summary['corrections']),
             '<details><summary>运行与统计来源</summary><pre>' + esc(dump({"run_id":RUN_ID,"model_id":MODEL_ID,"prompt_version":summary["prompt_version"],"source_dataset_hash":summary["dataset_hash"],"review_dataset_hash":summary["review_dataset_hash"],"truth":"当次冻结标注叠加明确的用户更正；仅重算受影响图片，原始评分和回答保留","corrections":summary["corrections"],"new_api_calls":0})) + '</pre></details></header>',
             '<section class="panel"><h2>按作品查看低分分布</h2>']
    by_work = {slug: stats([r for r in rows if r["work_slug"] == slug]) for slug, _ in works}
    work_table = []
    for slug, name in works:
        st = by_work[slug]
        correct, targets = st['correct_targets'], st['verified_targets']
        work_table.append([esc(name), f'{st["error_images"]} / {st["images"]}', str(st['invalid_format_images']), str(st['valid_error_images']),
                           f'{correct} / {targets}（{correct/targets:.2%}）' if targets else '—'])
    parts.append(table(["作品", "错误图 / 全部图", "格式失败图", "合法输出但未全对图", "全部目标中正确人物"], work_table))
    parts.append('<div class="notice"><strong>《功夫女足》已确认的现象：</strong>B组40张图片、50个人物均未识别正确：48个姓名错误、2个返回null，接口调用和返回格式均正常。C组同样40张未全对，其中8张为格式失败；其余合法输出中，39个人物姓名错误、1个返回null。<br>这说明低分包含明确的角色命名错误，不能仅归因于无框定位。知识覆盖、网页与API模型版本或联网能力差异仍待网页版对照验证，现有数据不足以确定根因。</div>')
    if experiment == 'C':
        loc = totals['legal_output_localization']
        parts.append(f'<p>本组合法输出共{sum(r["parse_ok"] for r in rows)}张：按原IoU≥0.5规则，定位匹配{loc["tp"]}人、漏检{loc["fn"]}人、额外误检{loc["fp"]}人，另有{totals["ignored_background_predictions"]}个背景预测被忽略。格式失败的{totals["invalid_format_images"]}张不属于这项合法输出统计，原总分仍保留它们的影响。</p>')
    outcomes = totals['person_outcomes']
    parts.append('<p>逐人物原判定：' + '；'.join(esc(LABELS.get(k,k)) + f' {v}人' for k,v in outcomes.items()) + '。背景忽略预测单列，不算人物识别错误。</p></section>')
    cat_counts = Counter(c for r in failed for c in categories(r))
    parts.append('<div class="toolbar" id="top"><label>作品<select id="work"><option value="">全部作品</option>' + ''.join(f'<option value="{esc(slug)}">{esc(name)}</option>' for slug,name in works) + '</select></label><label>错误类型（同图可有多类）<select id="error-type"><option value="">全部错误</option>' + ''.join(f'<option value="{esc(k)}">{esc(LABELS.get(k,k))} · {v}张</option>' for k,v in sorted(cat_counts.items())) + '</select></label><label>搜索编号、真值或模型回答<input id="search" type="search" placeholder="例如 gongfunvzu_0003 或 徐风"></label><button id="reset" type="button">重置筛选</button><strong id="visible-count"></strong></div><p id="empty" hidden>没有符合筛选条件的图片。</p>')
    exported = []
    for index, row in enumerate(failed, 1):
        sid = row['sample_id']
        sample, raw = samples[sid], raw_records[(experiment, sid)]
        entry = gallery[sid]['experiments'][experiment]
        image_data = (gallery_dir / entry['image']).read_bytes()
        expected = raw['request']['image_meta']['encoded_sha256']
        if digest(image_data) != expected or entry['encoded_sha256'] != expected:
            raise ValueError(f'Image differs from actual API request: {experiment}/{sid}')
        if sample['image_sha256'] != raw['request']['image_meta']['source_sha256']:
            raise ValueError(f'Frozen annotation/image fingerprint mismatch: {sid}')
        gt_by_id = {p['person_id']:p for p in sample['persons']}
        for detail in row['person_details']:
            if detail['identity_category'] == 'verified' and gt_by_id[detail['person_id']]['character'] != detail['gt_character']:
                raise ValueError(f'Frozen truth differs from score row: {sid}')
        parsed = parse_response(raw['response']['raw_text'], experiment, list(gt_by_id))
        if parsed.parse_ok != row['parse_ok']:
            raise ValueError(f'Parser result differs from stored score: {experiment}/{sid}')
        parts.append(render_case(index, sample, row, raw, image_data))
        exported.append({'sample_id':sid,'work_slug':row['work_slug'],'categories':categories(row),'parse_ok':row['parse_ok'],
                         'encoded_sha256':expected,'person_details':row['person_details']})
    parts.append('<footer>本文件使用既有真实回答、当次冻结标注及用户明确更正生成；原始运行记录保留，未再次调用模型。</footer></main><dialog id="zoom-dialog"><div class="dialog-head"><strong id="zoom-title"></strong><button id="close-zoom" type="button">关闭（Esc）</button></div><div id="zoom-content" class="dialog-content"></div></dialog><script>' + JS + '</script></body></html>')
    return ''.join(parts), {'totals':totals,'by_work':by_work,'error_images':exported}


def export_review(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    summary_path = ROOT / 'reports/2026-11-13-multi-v2/summary.json'
    dataset_path = ROOT / 'results/multi' / RUN_ID / 'dataset/dataset.json'
    gallery_dir = ROOT / 'results/multi/input_gallery'
    context = build_identity_review(ROOT)
    summary = {**context['summary'], 'review_dataset_hash':context['review_dataset_hash'], 'corrections':context['corrections']}
    if summary['run_id'] != RUN_ID or summary.get('is_mock'):
        raise ValueError('Expected the approved completed real run')
    samples = context['samples']
    gallery = {r['sample_id']:r for r in json.loads((gallery_dir/'manifest.json').read_text(encoding='utf-8'))['rows']}
    rows = [r for r in context['rows'] if r['experiment'] in {'B','C'}]
    if len(rows) != 320 or len({(r['experiment'],r['sample_id']) for r in rows}) != 320:
        raise ValueError('Expected 160 unique tasks per group')
    raw_records = context['raw_records']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest = {'run_id':RUN_ID,'model_id':MODEL_ID,'generated_at':datetime.now(timezone.utc).isoformat(),
                'new_api_calls':0,'dataset_hash':summary['dataset_hash'],'review_dataset_hash':context['review_dataset_hash'],
                'corrections':context['corrections'],
                'source_summary_sha256':digest(summary_path.read_bytes()),'frozen_dataset_sha256':digest(dataset_path.read_bytes()),'experiments':{}}
    # Validate/build both before writing either report.
    reports = {}
    for experiment in ('B','C'):
        selected=[r for r in rows if r['experiment']==experiment]
        expected_count=sum(r['image_correct'] is False for r in selected)
        content, metadata = render_report(experiment,selected,samples,raw_records,gallery,gallery_dir,summary)
        if metadata['totals']['error_images'] != expected_count:
            raise ValueError(f'Unexpected error count in {experiment}')
        reports[experiment] = content
        metadata['original_totals'] = stats([r for r in context['original_rows'] if r['experiment']==experiment])
        manifest['experiments'][experiment] = metadata
    for experiment, content in reports.items():
        path = output_dir / f'qwenmax_{experiment}_errors.html'
        path.write_text(content,encoding='utf-8')
        manifest['experiments'][experiment]['file'] = path.name
        manifest['experiments'][experiment]['file_sha256'] = digest(path.read_bytes())
    (output_dir/'review_manifest.json').write_text(dump(manifest)+'\n',encoding='utf-8')
    (output_dir/'使用说明.txt').write_text(
        'Qwen Max 逐图复核与网页版测试包\n\n'
        '1. 双击 qwenmax_identity_summary.html：Qwen Max姓名识别统计，含全部四部与去掉功夫女足的三部对照。\n'
        f'2. 双击 qwenmax_B_errors.html：更正标注后B组全部{manifest["experiments"]["B"]["totals"]["error_images"]}张错误图片。\n'
        f'3. 双击 qwenmax_C_errors.html：C组全部{manifest["experiments"]["C"]["totals"]["error_images"]}张错误图片；格式失败单列。\n'
        '两份HTML均内嵌图片，不需要启动服务器；可按作品、错误类型、编号和姓名筛选。\n'
        '若编辑器只显示源代码，请在文件管理器中找到文件并选择用浏览器打开。\n\n'
        '网页版截图入口为 gongfunvzu_web_test_15.html，共15张，每张A/B/C并排及人工标注。\n'
        '点击图片放大后只截取图片区域；角色对照用于测试后核对，提示词可从页面复制。\n'
        '逐张新建对话，保留完整回答与模型设置。\n\n'
        '用户已确认的标注更正详见annotation_corrections.json，报告已应用全部更正。\n'
        '原始运行快照、原始回答和历史报告保留。\n'
        '姓名命中率忽略坐标与编号，但仍将漏答、null、错名纳入分母；不会剔除格式失败图片来提高分数。\n'
        '分享HTML时发送完整文件，对方下载后用浏览器打开；报告图片内嵌，不需服务器或账号。\n'
        '本次未产生新的模型调用。\n',encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export_review(args.output)
    print(dump({'output':str(args.output),'error_images':{e:v['totals']['error_images'] for e,v in manifest['experiments'].items()},'new_api_calls':0}))


if __name__ == '__main__':
    main()
