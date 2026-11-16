"""Export one offline contact sheet of all currently reviewed annotations.

Only the unmarked C JPEGs are reused from the input gallery. All names, boxes
and optional-person rules come from the current validated dataset, because the
input gallery's old human_annotations and marked A/B images can be stale.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_c_search import annotation_review_context, atomic_text
from vcrb.multi.schema import dataset_hash, load_dataset, validate_dataset

OUTPUT = ROOT / 'reports/2026-11-16-annotation-review/gallery_160.html'


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#213547;font:12px/1.45 system-ui,'Microsoft YaHei',sans-serif}main{max-width:2100px;margin:auto;padding:16px}.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}h1{font-size:21px;margin:0}.version{font-size:11px;color:#087f5b;margin:3px 0 0}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-left:auto}select,input,button{font:inherit;color:inherit;background:white;border:1px solid #cad8e1;border-radius:6px;padding:6px 8px}input{width:180px;max-width:100%}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #7bbce8;outline-offset:2px}#count{color:#617787;white-space:nowrap;font-size:11px}.grid{--columns:5;display:grid;grid-template-columns:repeat(var(--columns),minmax(0,1fr));gap:10px;align-items:start}.card{min-width:0;background:white;border:1px solid #d8e2e9;border-radius:8px;padding:8px}.card[hidden]{display:none}.card-title{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:6px;white-space:nowrap}.card-title b{font-size:11px}.card-title span{color:#6b7c88;font-size:10px}.thumb{display:block;width:100%;border:0;border-radius:5px;padding:0;overflow:hidden;cursor:zoom-in}.picture{display:block;position:relative;line-height:0;background:#13202a}.picture img{display:block;width:100%;height:auto}.picture svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.box-number{position:absolute;min-width:14px;background:#087f5b;color:white;padding:1px 3px;font:700 10px/1.2 system-ui;border-radius:0 0 3px 0;pointer-events:none}.people{display:flex;flex-wrap:wrap;gap:4px 8px;margin-top:6px;font-size:11px}.person{display:inline-flex;align-items:center;gap:4px}.person b{color:#087f5b;font-size:10px}.person em{font-size:10px;font-style:normal;color:#6b7c88}.quota{font-size:10px;color:#887039;margin:4px 0 0}#empty{color:#617787;text-align:center;margin:40px}dialog{width:min(1450px,96vw);max-height:96vh;overflow:auto;border:0;border-radius:10px;padding:12px}dialog::backdrop{background:#132536c9}.dialog-head{display:flex;align-items:center;gap:8px;margin-bottom:10px}.dialog-head h2{font-size:16px;margin:0 auto 0 0}.dialog-head button{white-space:nowrap}.dialog-head button:disabled{opacity:.4;cursor:default}#zoom-picture{max-height:78vh;overflow:auto}#zoom-picture .box-number{font-size:15px;min-width:20px}#zoom-info .people{font-size:15px;gap:7px 16px}#zoom-info .person b{font-size:14px}#zoom-info .quota,#zoom-info .person em{font-size:12px}@media(max-width:900px){main{padding:10px}.controls{margin-left:0}.top{gap:8px}h1{font-size:19px}.grid{--columns:3;gap:8px}.card{padding:6px}.card-title{flex-wrap:wrap;gap:0}.dialog-head{flex-wrap:wrap}.dialog-head h2{flex-basis:100%}}@media(max-width:580px){.grid{--columns:2;gap:6px}.controls{gap:6px}input{width:135px}.card-title b{font-size:10px}.card-title span{font-size:9px}.people{gap:3px 6px;font-size:10px}.version{font-size:10px}.top{margin-bottom:10px}}@media print{body{background:white}main{padding:0}.controls{display:none}.grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.card{break-inside:avoid}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

JS = """
const cards=[...document.querySelectorAll('.card')],work=document.getElementById('work'),query=document.getElementById('search'),density=document.getElementById('density'),grid=document.querySelector('.grid'),dialog=document.getElementById('zoom');
function layout(){const maximum=innerWidth<580?2:innerWidth<900?3:innerWidth<1200?4:6;grid.style.setProperty('--columns',Math.min(Number(density.value),maximum));}
function filter(){let n=0;for(const card of cards){card.hidden=!!(work.value&&card.dataset.work!==work.value)||!card.dataset.search.includes(query.value.trim().toLowerCase());n+=Number(!card.hidden);}document.getElementById('count').textContent=`${n} / ${cards.length} 张`;document.getElementById('empty').hidden=!!n;}
let current=null;
function show(card){current=card;document.getElementById('zoom-title').textContent=card.dataset.label;document.getElementById('zoom-picture').replaceChildren(card.querySelector('.picture').cloneNode(true));const info=document.getElementById('zoom-info');info.replaceChildren(card.querySelector('.people').cloneNode(true));const quota=card.querySelector('.quota');if(quota)info.append(quota.cloneNode(true));const visible=cards.filter(c=>!c.hidden),i=visible.indexOf(card);document.getElementById('prev').disabled=i<=0;document.getElementById('next').disabled=i>=visible.length-1;}
function step(delta){const visible=cards.filter(c=>!c.hidden),i=visible.indexOf(current),next=visible[i+delta];if(next)show(next);}
for(const card of cards)card.querySelector('.thumb').addEventListener('click',()=>{show(card);dialog.showModal();});
work.addEventListener('change',filter);query.addEventListener('input',filter);density.addEventListener('change',layout);addEventListener('resize',layout);document.getElementById('reset').addEventListener('click',()=>{work.value='';query.value='';filter();});document.getElementById('close').addEventListener('click',()=>dialog.close());document.getElementById('prev').addEventListener('click',()=>step(-1));document.getElementById('next').addEventListener('click',()=>step(1));dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});dialog.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}});layout();filter();
"""


def esc(value):
    return escape(str(value), quote=True)


def picture(sample, data, size):
    """Keep the JPEG bytes and native ratio; draw current boxes as SVG."""
    height = 1000 * size[1] / size[0]
    optional = set(sample.get('evaluation_policy', {}).get('optional_person_ids', []))
    boxes, numbers = [], []
    for person in sample['persons']:
        x1, y1, x2, y2 = person['bbox']
        pid = person['person_id']
        dash = ' stroke-dasharray="5 3"' if pid in optional else ''
        boxes.append(f'<rect data-person-id="{pid}" x="{x1*1000:.6f}" y="{y1*height:.6f}" width="{(x2-x1)*1000:.6f}" height="{(y2-y1)*height:.6f}" fill="none" stroke="#087f5b" stroke-width="2" vector-effect="non-scaling-stroke"{dash}/>')
        numbers.append(f'<span class="box-number" style="left:{x1*100:.6f}%;top:{min(y1*100, 94):.6f}%">{pid}</span>')
    return ('<span class="picture"><img loading="lazy" decoding="async" width="' + str(size[0]) + '" height="' + str(size[1])
            + '" alt="' + esc(sample['sample_id']) + '" src="data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')
            + '"><svg viewBox="0 0 1000 ' + f'{height:.6f}' + '" aria-hidden="true">' + ''.join(boxes) + '</svg>' + ''.join(numbers) + '</span>')


def export_gallery(root=ROOT, output=OUTPUT):
    root, output = Path(root).resolve(), Path(output)
    dataset_dir = root / 'data/v2'
    samples = load_dataset(dataset_dir / 'dataset.json')
    errors, _stats = validate_dataset(samples, dataset_dir, for_run=True)
    if errors:
        raise ValueError('Current dataset is not fully approved: ' + '; '.join(errors[:3]))
    _base, review = annotation_review_context(root, samples)
    digest = dataset_hash(samples)
    if not review.get('applied') or review.get('current_dataset_hash') != digest:
        raise ValueError('A matching confirmed review revision is required')
    counts = Counter(sample.work_slug for sample in samples)
    if len(samples) != 160 or len(counts) != 4 or set(counts.values()) != {40}:
        raise ValueError('Expected the final four-work, 40-images-per-work dataset')
    gallery_root = root / 'results/multi/input_gallery'
    gallery_rows = json.loads((gallery_root / 'manifest.json').read_text())['rows']
    gallery = {row['sample_id']: row for row in gallery_rows}
    if len(gallery_rows) != 160 or set(gallery) != {s.sample_id for s in samples}:
        raise ValueError('Input gallery has duplicate, missing or foreign sample IDs')
    works = {sample.work_slug: sample.work_name for sample in samples}
    revision = str(review.get('revision_id') or '当前确认版本').rsplit('_', 1)[-1]
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="dataset-sha256" content="' + digest + '"><title>160张最终标注图</title><style>' + CSS + '</style></head><body><main><header class="top"><div><h1>160张最终标注图</h1><p class="version" title="标注指纹：' + digest + '">最终复核 ' + esc(revision) + ' · 点击图片放大</p></div><div class="controls"><select id="work" aria-label="筛选作品"><option value="">全部作品 · 160张</option>']
    parts += ['<option value="' + esc(slug) + '">' + esc(name) + ' · ' + str(counts[slug]) + '张</option>' for slug, name in works.items()]
    parts.append('</select><input id="search" type="search" placeholder="编号 / 角色姓名" aria-label="搜索编号或角色"><select id="density" aria-label="排列密度"><option value="4">宽松</option><option value="5" selected>标准</option><option value="6">紧凑</option></select><button id="reset" type="button">重置</button><span id="count">160 / 160 张</span></div></header><div class="grid">')
    people_count = 0
    for item in samples:
        sample = item.to_dict()
        sid = sample['sample_id']
        row, entry = gallery[sid], gallery[sid]['experiments']['C']
        if row['source_sha256'] != sample['image_sha256'] or entry['annotation_mode'] != 'unmarked':
            raise ValueError('Source image or unmarked C mode differs: ' + sid)
        image_path = (gallery_root / entry['image']).resolve()
        if not image_path.is_relative_to(gallery_root):
            raise ValueError('Image path outside the input gallery')
        data = image_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['encoded_sha256']:
            raise ValueError('C JPEG fingerprint differs: ' + sid)
        with Image.open(BytesIO(data)) as im:
            size = im.size
        policy = sample.get('evaluation_policy', {})
        optional = set(policy.get('optional_person_ids', []))
        label = sid + ' · ' + sample['work_name']
        names = [str(p['character'] or '未知') for p in sample['persons']]
        search = ' '.join([sid, sample['work_name']] + names).lower()
        parts.append('<article class="card" id="' + esc(sid) + '" data-work="' + esc(sample['work_slug']) + '" data-label="' + esc(label) + '" data-search="' + esc(search) + '"><div class="card-title"><b>' + esc(sid) + '</b><span>' + esc(sample['work_name']) + '</span></div><button class="thumb" type="button" aria-label="放大 ' + esc(label) + '">' + picture(sample, data, size) + '</button><div class="people">')
        for person in sample['persons']:
            suffix = '<em>可选</em>' if person['person_id'] in optional else ''
            parts.append('<span class="person"><b>' + str(person['person_id']) + '</b>' + esc(person['character'] or '未知') + suffix + '</span>')
        parts.append('</div>')
        quota = policy.get('required_name_count')
        if quota is not None and quota < len(sample['persons']) - len(optional):
            parts.append('<p class="quota">' + str(len(sample['persons'])) + '人任选' + str(quota) + '人</p>')
        parts.append('</article>')
        people_count += len(sample['persons'])
    parts.append('</div><p id="empty" hidden>没有匹配的图片</p></main><dialog id="zoom" aria-labelledby="zoom-title"><div class="dialog-head"><h2 id="zoom-title"></h2><button id="prev" type="button">上一张</button><button id="next" type="button">下一张</button><button id="close" type="button">关闭（Esc）</button></div><div id="zoom-picture"></div><div id="zoom-info"></div></dialog><script>' + JS + '</script></body></html>')
    content = ''.join(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(output, content)
    return {'output': str(output), 'images': len(samples), 'person_boxes': people_count,
            'works': dict(counts), 'dataset_hash': digest, 'bytes': len(content.encode()), 'model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    print(json.dumps(export_gallery(output=parser.parse_args().output), ensure_ascii=False))
