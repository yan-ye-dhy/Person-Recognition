"""Local, read-only A/B/C input gallery using the runner's actual JPEG encoder."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from ..config import Work, load_works
from .runner import InputBuilder
from .schema import dataset_hash, dataset_stats, load_dataset, validate_dataset
from .stratification import SIDECAR_NAME, load_stratification, stratification_hash

DEFAULT_OUTPUT = Path("results/multi/input_gallery")


def _jpeg(data_url: str) -> bytes:
    header, separator, encoded = data_url.partition(",")
    if not separator or header != "data:image/jpeg;base64":
        raise ValueError("Gallery expects the runner's JPEG data URL")
    return base64.b64decode(encoded, validate=True)


def generate_gallery(dataset_path: Path | str, output_dir: Path | str = DEFAULT_OUTPUT) -> Path:
    """Export all current annotations as side-by-side model inputs; make no calls.

    A/B reference one identical JPEG file. C is the original-image request JPEG,
    with no boxes or labels. Human review metadata appears only in HTML/JSON and
    is never painted onto any model input. Original images and annotations are
    never overwritten. ``index.html`` works directly or via a local HTTP server.
    """
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if output == dataset_path.parent or output.is_relative_to(dataset_path.parent):
        raise ValueError("Gallery output must be separate from the annotation dataset directory")
    dataset_bytes = dataset_path.read_bytes()
    samples = load_dataset(dataset_path)
    errors, stats = validate_dataset(samples, dataset_path.parent)
    if errors:
        raise ValueError("Gallery input validation failed: " + "; ".join(errors[:8]))
    if not samples:
        raise ValueError("Gallery requires at least one sample")
    tags_path = dataset_path.parent / SIDECAR_NAME
    tags_payload = load_stratification(tags_path, samples) if tags_path.is_file() else None
    person_tags = {(r["sample_id"], r["person_id"]): r for r in (tags_payload or {}).get("persons", [])}
    works = {w.slug: w for w in load_works()}
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    rows = []
    with TemporaryDirectory(prefix="multi-gallery-numbered-") as marked:
        builder = InputBuilder(dataset_path.parent, marked)
        for sample in samples:
            work = works.get(sample.work_slug) or Work(sample.work_slug, sample.work_name, sample.work_category, "影视作品", None)
            source = (dataset_path.parent / sample.image_path).resolve()
            prompts, encoded, metas = {}, {}, {}
            for experiment in ("A", "B", "C"):
                prompt, data_url, meta = builder(sample, experiment, work)
                prompts[experiment], encoded[experiment], metas[experiment] = prompt, _jpeg(data_url), meta
            if encoded["A"] != encoded["B"]:
                raise ValueError(f"{sample.sample_id}: A/B request images differ")
            if metas["C"]["input_file_sha256"] != sample.image_sha256:
                raise ValueError(f"{sample.sample_id}: C did not use the unmarked source")
            # IDs are displayed as text only; filenames use a stable safe digest
            # so a user-supplied ID can never escape the gallery output directory.
            stem = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:20]
            ab_file, c_file = f"images/{stem}_AB.jpg", f"images/{stem}_C.jpg"
            (output / ab_file).write_bytes(encoded["A"])
            (output / c_file).write_bytes(encoded["C"])
            entries = {}
            for experiment in ("A", "B", "C"):
                entries[experiment] = {
                    "image": c_file if experiment == "C" else ab_file,
                    "encoded_sha256": hashlib.sha256(encoded[experiment]).hexdigest(),
                    "annotation_mode": metas[experiment]["annotation_mode"],
                    "user_prompt": prompts[experiment],
                }
            rows.append({
                "sample_id": sample.sample_id, "work_slug": sample.work_slug, "work_name": sample.work_name,
                "annotated_person_count": len(sample.persons),
                "verified_identity_count": sum(p.identity_status == "verified" and p.review_status == "approved" for p in sample.persons),
                "unknown_identity_count": sum(p.identity_status == "unknown" for p in sample.persons),
                "pending_identity_count": sum(p.review_status != "approved" or p.identity_status == "needs_review" for p in sample.persons),
                "ignore_region_count": len(sample.ignore_regions), "review_status": sample.review_status,
                "foreground_complete": sample.foreground_complete,
                "human_annotations": {"persons": [
                    {"person_id": p.person_id, "character": p.character,
                     "identity_status": p.identity_status, "review_status": p.review_status,
                     "note": p.note,
                     **{k: person_tags[(sample.sample_id, p.person_id)][k]
                        for k in ("face_orientation", "visibility")
                        if (sample.sample_id, p.person_id) in person_tags}}
                    for p in sample.persons
                ]},
                "source_sha256": sample.image_sha256,
                "source_image_link": quote(Path(os.path.relpath(source, output)).as_posix(), safe="/"),
                "source_size": [sample.image_width, sample.image_height], "experiments": entries,
            })
    # Detect concurrent edits rather than publishing a misleading dataset binding.
    if dataset_path.read_bytes() != dataset_bytes:
        raise ValueError("Annotations changed while generating the gallery; regenerate from the saved version")
    manifest = {
        "kind": "multi_person_input_gallery", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(), "annotation_hash": dataset_hash(samples),
        "samples": len(rows), "quality": stats, "api_calls": 0,
        "notice": "仅本地输入与草稿标注预览；不是模型测试成绩。A/B共用实际编码JPEG，C为无框原图的实际编码JPEG。",
        "human_metadata_only": ["annotated_person_count", "review_status", "foreground_complete", "identity_counts", "human_annotations"],
        "rows": rows,
    }
    page = _html(rows, stats)
    gate_path = dataset_path.parent / "real_call_gate.json"
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        gate = {}
    manifest["real_call_gate_state"] = gate.get("state", "unconfirmed")
    if tags_payload is not None:
        manifest["stratification"] = {"hash": stratification_hash(tags_payload),
                                      "persons": len(person_tags), "purpose": "human_review_and_statistics_only"}
    manifest["annotation_policy"] = json.loads(dataset_bytes).get("annotation_policy", {})
    notices = []
    confirmed = gate.get("state") == "confirmed_by_user" and gate.get("confirmed_dataset_hash") == manifest["annotation_hash"]
    if not confirmed:
        notices.append("真实模型调用已暂停：等你确认这版标注后，再统一调用。")
    else:
        page = page.replace("已代标 · 待用户确认", "用户已确认")
        page = page.replace("已由助手完成当前范围的代标与独立证据复核，等待用户确认", "当前范围的角色标注已经用户确认")
        notices.append("你已确认当前角色标注并授权模型测试。朝向与可见性标签仅供统计，不发送给模型。")
    if manifest["annotation_policy"].get("scope"):
        notices.append("当前标注范围：" + manifest["annotation_policy"]["scope"])
    if str(manifest["annotation_policy"].get("kuangbiao_refresh", "")).startswith("in progress"):
        notices.append("《狂飙》正在用新MP4重新选图；可先检查另外三部，重抽完成后会更新同一入口。")
    if notices:
        page = page.replace("</h1>", "</h1>" + "".join(f'<p class="notice">{escape(message)}</p>' for message in notices), 1)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text(page, encoding="utf-8")
    return output / "index.html"


def _html(rows, stats):
    works = {r["work_slug"]: r["work_name"] for r in rows}
    options = '<option value="">全部作品</option>' + "".join(
        f'<option value="{escape(slug, quote=True)}">{escape(name)}</option>' for slug, name in works.items()
    )
    cards = []
    captions = {"A": "编号红框 · 不提供作品名", "B": "编号红框 · 提供作品名", "C": "无框原图 · 提供作品名"}
    for row in rows:
        sid = escape(row["sample_id"], quote=True)
        cells = []
        for experiment in ("A", "B", "C"):
            entry = row["experiments"][experiment]
            title = f'{experiment} · {row["sample_id"]}'
            cells.append(
                f'<section class="experiment"><h3>{experiment} <span>{captions[experiment]}</span></h3>'
                f'<a class="image-link" href="{entry["image"]}" target="_blank" rel="noopener" data-title="{escape(title, quote=True)}">'
                f'<img src="{entry["image"]}" alt="{escape(title, quote=True)}" loading="lazy" decoding="async" width="1600" height="670"></a>'
                f'<a class="zoom-link" href="{entry["image"]}" target="_blank" rel="noopener">放大{experiment}输入图 ↗</a></section>'
            )
        approved = row["review_status"] == "approved" and row["foreground_complete"]
        status = "已代标 · 待用户确认" if approved else "草稿 · 待补框/复核"
        detail = f'已标 {row["annotated_person_count"]} 人 · 身份已核 {row["verified_identity_count"]} · 待核 {row["pending_identity_count"]} · unknown {row["unknown_identity_count"]}'
        person_labels = []
        for person in row.get("human_annotations", {}).get("persons", []):
            verified = person["identity_status"] == "verified" and person["review_status"] == "approved"
            unknown = person["identity_status"] == "unknown" and person["review_status"] == "approved"
            identity = person.get("character") or ("身份未知" if unknown else "身份待核")
            state = "已核实" if verified else ("保留 null" if unknown else "待复核")
            label_class = "verified" if verified else ("unknown" if unknown else "pending")
            note = f'<span class="person-note">{escape(person["note"])}</span>' if person.get("note") else ""
            face_labels = {"front": "正脸", "three_quarter": "斜侧脸", "profile": "纯侧脸", "not_visible": "面部不可见/难判断"}
            visibility_labels = {"clear": "清晰", "occluded_or_blurry": "遮挡/模糊"}
            tags = " · ".join(value for value in (face_labels.get(person.get("face_orientation")), visibility_labels.get(person.get("visibility"))) if value)
            tag_label = f'<span class="person-note">统计分类：{escape(tags)}</span>' if tags else ""
            person_labels.append(
                f'<li class="person-label {label_class}"><strong>{person["person_id"]} {escape(identity)}</strong>'
                f'<span class="person-state"> · {state}</span>{tag_label}{note}</li>'
            )
        human_annotations = (
            '<section class="human-annotations" aria-label="编号与人工角色对照">'
            '<h3>编号与角色对照 <span>仅供人工审核；不绘入图片、不发送给模型</span></h3>'
            f'<ul>{"".join(person_labels)}</ul></section>'
        )
        prompt_details = "".join(f'<h4>实验 {experiment}</h4><pre>{escape(row["experiments"][experiment]["user_prompt"])}</pre>' for experiment in ("A", "B", "C"))
        cards.append(
            f'<article class="sample" id="{sid}" data-sample="{sid}" data-work="{escape(row["work_slug"], quote=True)}" data-persons="{row["annotated_person_count"]}">'
            f'<header class="sample-header"><div><h2>{sid} <small>{escape(row["work_name"])}</small></h2><p>{detail}（仅人审信息）</p></div>'
            f'<div class="sample-actions"><span class="badge {"approved" if approved else "draft"}">{status}</span> '
            f'<a href="{escape(row["source_image_link"], quote=True)}" target="_blank" rel="noopener">查看未改动原始大图 ↗</a></div></header>'
            f'<div class="experiments">{"".join(cells)}</div>'
            f'{human_annotations}'
            f'<details><summary>查看 A/B/C 实际输入提示词</summary>{prompt_details}</details></article>'
        )
    total = len(rows)
    multi = sum(r["annotated_person_count"] >= 2 for r in rows)
    all_reviewed = stats.get("draft_samples", 0) == 0 and stats.get("foreground_incomplete", 0) == 0
    notice = (
        "已由助手完成当前范围的代标与独立证据复核，等待用户确认；本页不是模型成绩。"
        if all_reviewed else
        "当前为标注草稿预览，前景人物尚未全部补框、身份仍有待复核；不能当作正式模型成绩。"
    )
    notice += "编号角色对照、人数和审核状态仅供人审，不绘入图片，也不发送给模型。筛选“多人”按当前已标注框 ≥2 判断。"
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全部 A/B/C 实验图片与标注</title><style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#182633;background:#edf1f5}*{box-sizing:border-box}body{margin:0}a{color:#135d92}main{max-width:1680px;margin:auto;padding:22px}h1{margin:0 0 10px;font-size:27px}p{line-height:1.65}.intro{background:#fff;border-radius:12px;padding:23px;margin-bottom:15px}.intro p{margin:7px 0}.notice{border-left:4px solid #d78b23;padding:8px 14px;background:#fff6e8;border-radius:4px}.toolbar{position:sticky;top:0;z-index:3;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px;background:#f9fbfdf5;border:1px solid #d7e0e9;border-radius:9px;backdrop-filter:blur(8px);margin:14px 0}label{display:flex;gap:7px;align-items:center;font-size:14px}select,input,button{font:inherit;padding:9px 10px;border:1px solid #b4c1cd;border-radius:6px;background:white}input{min-width:225px}button{cursor:pointer}#result-count{font-weight:650;margin-left:auto}.sample{background:white;border:1px solid #d7e0e9;border-radius:11px;padding:16px;margin:14px 0;scroll-margin-top:95px}.sample[hidden]{display:none}.sample-header{display:flex;justify-content:space-between;align-items:start;gap:15px;margin-bottom:12px}h2{font-size:18px;margin:0}h2 small{font-size:15px;color:#566877;margin-left:10px}.sample-header p{font-size:13px;color:#5b6b78;margin:5px 0}.sample-actions{display:flex;gap:10px;align-items:center;font-size:13px;flex-wrap:wrap}.badge{display:inline-block;padding:4px 9px;border-radius:20px;white-space:nowrap}.draft{background:#fff0d6;color:#855000}.approved{background:#e3f4e8;color:#245e37}.experiments{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}h3{font-size:19px;margin:0 0 7px}h3 span{font-size:13px;font-weight:450;color:#506474}.image-link{display:block;background:#16202b;overflow:hidden;border-radius:5px}.image-link img{display:block;width:100%;height:auto;aspect-ratio:1600/670;object-fit:contain}.zoom-link{display:block;font-size:12px;margin-top:7px}details{margin-top:12px;font-size:13px;color:#4c5f6d}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:12px;border-radius:5px;color:#2c3c49;font-size:12px}#empty{padding:35px;text-align:center}dialog{border:0;border-radius:10px;padding:15px;max-width:96vw;max-height:96vh;background:#101820;color:white}dialog::backdrop{background:#000b}dialog img{display:block;max-width:90vw;max-height:80vh;object-fit:contain}dialog header{display:flex;justify-content:space-between;gap:25px;align-items:center;margin-bottom:10px}dialog a{color:#9ed5ff}dialog button{padding:5px 12px}.footer{font-size:13px;color:#5b6b78;margin:22px 0}@media(max-width:900px){.sample-header{display:block}.sample-actions{margin-top:8px}.experiments{gap:6px}h3 span{display:block;font-size:11px}main{padding:10px}.sample{padding:10px}.toolbar{gap:8px}input{min-width:170px}#result-count{margin-left:0}h1{font-size:23px}}@media(max-width:560px){.experiments{grid-template-columns:1fr}h3 span{display:inline}.toolbar{position:static}}
.human-annotations{margin-top:15px;padding:14px;border:1px solid #b8cddd;border-radius:8px;background:#f4f9fd}.human-annotations h3{font-size:16px}.human-annotations h3 span{display:inline-block;margin-left:8px;font-size:12px}.human-annotations ul{display:flex;flex-wrap:wrap;gap:9px;list-style:none;padding:0;margin:11px 0 0}.person-label{padding:8px 11px;border:1px solid #c8d5df;border-radius:7px;background:white;max-width:100%;font-size:14px}.person-label.verified{border-color:#90bfa1;background:#edf8f0}.person-label.unknown{background:#f1f4f7}.person-label.pending{border-color:#dfbd7f;background:#fff6e7}.person-state{font-size:13px;color:#526675}.person-note{display:block;font-size:12px;line-height:1.5;color:#5b6b78;margin-top:4px;overflow-wrap:anywhere}
</style></head><body><main><section class="intro"><h1>全部 A/B/C 实验图片与标注</h1>
<p>共 <strong>''' + str(total) + ''' 张</strong>，每行并排展示三组实际模型输入。点击图片可放大，也可打开未改动的原始大图。</p>
<p><strong>A / B 使用完全相同的编号红框 JPEG</strong>，唯一实验变量是提示词是否提供作品名称。<strong>C 使用无框原图</strong>，由模型自主发现人物；不向模型提示 GT 人数、框坐标或身份。</p>
<p class="notice">''' + escape(notice) + '''</p>
<p>当前已标 ''' + str(stats["persons"]) + ''' 人，''' + str(multi) + ''' 张已有多人框，''' + str(stats["draft_samples"]) + ''' 张仍为草稿。<a href="manifest.json" target="_blank" rel="noopener">查看生成清单与输入哈希</a>。</p></section>
<section class="toolbar" aria-label="图片筛选"><label>作品 <select id="work-filter">''' + options + '''</select></label>
<label>人数 <select id="people-filter"><option value="all">全部图片</option><option value="multi">只看已标多人（≥2）</option></select></label>
<label>编号 <input id="sample-search" type="search" placeholder="搜索 sample_id，例如 0012" autocomplete="off"></label>
<button id="reset-filters" type="button">清空筛选</button><output id="result-count" aria-live="polite">''' + str(total) + ' / ' + str(total) + ''' 张</output></section>
<div id="sample-list">''' + "\n".join(cards) + '''</div><p id="empty" hidden>没有匹配的图片，请调整作品或编号筛选。</p>
<p class="footer">本页只在本地生成和查看图片，不调用模型、不修改标注或原图。A/B/C 图片均来自与运行器相同的编码流程。</p>
<dialog id="lightbox"><header><span id="lightbox-title"></span><a id="lightbox-link" target="_blank" rel="noopener">新窗口打开大图 ↗</a><button id="close-lightbox" type="button">关闭 ×</button></header><img id="lightbox-image" alt="放大的实验输入图"></dialog>
<script>
(()=>{const cards=[...document.querySelectorAll('.sample')],work=document.getElementById('work-filter'),people=document.getElementById('people-filter'),search=document.getElementById('sample-search'),count=document.getElementById('result-count'),empty=document.getElementById('empty');
function filter(){const query=search.value.trim().toLowerCase();let shown=0;for(const card of cards){const visible=(!work.value||card.dataset.work===work.value)&&(people.value!=='multi'||Number(card.dataset.persons)>=2)&&(!query||card.dataset.sample.toLowerCase().includes(query));card.hidden=!visible;if(visible)shown++;}count.textContent=`${shown} / ${cards.length} 张`;empty.hidden=shown!==0;}
for(const element of [work,people,search])element.addEventListener(element===search?'input':'change',filter);
document.getElementById('reset-filters').addEventListener('click',()=>{work.value='';people.value='all';search.value='';filter();});
const params=new URLSearchParams(location.search);if(params.has('work'))work.value=params.get('work');if(params.get('people')==='multi')people.value='multi';search.value=params.get('q')||'';filter();
const dialog=document.getElementById('lightbox'),image=document.getElementById('lightbox-image');for(const link of document.querySelectorAll('.image-link'))link.addEventListener('click',event=>{if(!dialog.showModal)return;event.preventDefault();image.src=link.href;document.getElementById('lightbox-title').textContent=link.dataset.title;document.getElementById('lightbox-link').href=link.href;dialog.showModal();});document.getElementById('close-lightbox').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
})();
</script></main></body></html>'''
