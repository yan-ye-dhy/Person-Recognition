"""Create a compact offline review of matching answers flagged by the benchmark."""
from __future__ import annotations

import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_qwenmax_review import overlay

REPORT = ROOT / 'reports/2026-11-15-c-search'
LABELS = {'deepseek-flash': 'DeepSeek', 'qwen3.8-max': 'Qwen Max'}


class InputImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.images = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article' and 'case' in attrs.get('class', '').split():
            self.current = attrs['id']
        if tag == 'img' and self.current and self.current not in self.images:
            self.images[self.current] = attrs['src']

    def handle_endtag(self, tag):
        if tag == 'article':
            self.current = None


def select_pending_consensus(summary):
    models = [model for model in ('deepseek-flash', 'qwen3.8-max') if model in summary['models']]
    reviewed = set((summary.get('annotation_review') or {}).get('reviewed_sample_ids', []))
    selected = []
    for case in summary['cases']:
        if case['sample_id'] in reviewed:
            continue
        groups = {}
        for model in models:
            result = case['results'][model]['new']
            names = result['names']
            if result['state'] != 'ok' or not (not names['image_names_complete'] or names['extra_names'] > 0):
                continue
            # Compare full name multisets, ignoring order but retaining duplicates.
            signature = tuple(sorted(str(p['character']).strip() for p in names['predictions'] if p['character'] is not None))
            if signature:
                groups.setdefault(signature, []).append(model)
        matches = [(names, members) for names, members in groups.items() if len(members) >= 2]
        if matches:
            names, members = max(matches, key=lambda item: len(item[1]))
            selected.append({'case': case, 'names': names, 'models': members})
    selected.sort(key=lambda row: row['case']['sample_id'])
    return models, selected, reviewed


def export_consensus(report_dir=REPORT):
    report_dir = Path(report_dir)
    summary = json.loads((report_dir / 'summary.json').read_text())
    models, selected, reviewed = select_pending_consensus(summary)
    images = InputImages()
    images.feed((report_dir / 'index.html').read_text())
    esc = lambda value: escape(str(value), quote=True)
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepSeek / Qwen Max同答待核</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#203244;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:20px}header{margin-bottom:20px}h1{font-size:25px;margin:0}h2{font-size:21px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:18px}.card{background:white;padding:15px;border:1px solid #d5dfe7;border-radius:12px;min-width:0}.card h3{margin:0 0 10px;font-size:19px}.image{position:relative;line-height:0;cursor:zoom-in}.image img{width:100%;height:auto;display:block}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.answer{background:#e9f5ee;padding:10px;border-radius:7px}.note{color:#6c5321;background:#fff5df;padding:9px;border-radius:7px;font-size:14px}.muted{font-size:13px;color:#637386}p{margin:9px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:6px 8px;border-bottom:1px solid #e3e9ee;text-align:left;vertical-align:top}.match{font-weight:700;color:#087f5b}dialog{width:min(1500px,96vw);max-height:96vh;border:0;border-radius:10px;padding:14px}dialog .image{cursor:default}dialog button{float:right;margin-bottom:10px;padding:6px 18px}dialog::backdrop{background:#152536b3}a{color:#075d86}details>summary{cursor:pointer;padding:12px 0;font-size:20px;font-weight:700}@media(max-width:650px){main{padding:10px}.grid{grid-template-columns:1fr}h1{font-size:22px}}
</style><main><header><h1>DeepSeek / Qwen Max同答待核</h1>''']
    parts.append(f'<p><b>仅这两个模型同答且仍有姓名问题：{len(selected)}张。</b>姓名忽略顺序、保留重复次数；只看必答名额未满足或多报目标外姓名，不把纯坐标问题当姓名错误。</p><p class="muted">当前标注已应用{len(reviewed)}张用户确认；这些已审核样本不再列为待核疑问，模型在其中的真实错误仍保留于主报告。绿色框为当前标注，点击放大。共同答案只是人工复核线索，不自动改写真值。</p></header>')
    manifest = []
    for count, title, anchor in [(2, '尚未审核的两模型同答问题', 'two')]:
        rows = [row for row in selected if len(row['models']) == count]
        parts.append(f'<h2 id="{anchor}">{title} · {len(rows)}张</h2><div class="grid">')
        for row in rows:
            case = row['case']; sample = case['sample']; sid = case['sample_id']
            if sid not in images.images:
                raise ValueError('Pending consensus input missing from current error report: ' + sid)
            source = images.images[sid]
            policy = sample.get('evaluation_policy', {})
            optional = set(policy.get('optional_person_ids', []))
            quota = policy.get('required_name_count', len(sample['persons']) - len(optional))
            truth = '；'.join(f"{p['person_id']} {p['character']}" + ('（可选，可不答）' if p['person_id'] in optional else '') for p in sample['persons'])
            names = '、'.join(row['names'])
            parts.append(f'<article class="card" id="{esc(sid)}"><h3>{esc(sid)} · {esc(sample["work_name"])}</h3><div class="image"><img loading="lazy" alt="{esc(sid)}" src="{source}">{overlay(sample, {"parse_ok": False}, [], [sample["image_width"], sample["image_height"]])}</div>')
            parts.append(f'<p><b>当前标注（{len(sample["persons"])}个实框 / 姓名必答{quota}名）：</b>{esc(truth)}</p><p class="answer"><b>两模型共同回答：</b>{esc(names)}</p>')
            parts.append('<table><tr><th>模型</th><th>实际回答</th></tr>')
            for model in models:
                result = case['results'][model]['new']
                actual = '、'.join('未知' if p['character'] is None else str(p['character']) for p in result['names']['predictions']) or '（无姓名）'
                cls = ' class="match"' if model in row['models'] else ''
                parts.append(f'<tr{cls}><td>{esc(LABELS.get(model,model))}</td><td>{esc(actual)}</td></tr>')
            parts.append('</table>')
            reasons = [r.get('reason', '') for r in sample.get('ignore_regions', []) if isinstance(r, dict) and r.get('reason')]
            if reasons:
                parts.append('<p class="note"><b>当前忽略区说明：</b>' + esc('；'.join(reasons)) + '</p>')
            parts.append('</article>')
            manifest.append({'sample_id':sid, 'matching_models':row['models'], 'common_names':list(row['names']), 'current_characters':[p['character'] for p in sample['persons']]})
        parts.append('</div>')
    parts.append('''</main><dialog id="zoom"><button id="close">关闭</button><div id="large"></div></dialog><script>const d=document.getElementById('zoom');for(const el of document.querySelectorAll('.card .image'))el.addEventListener('click',()=>{document.getElementById('large').replaceChildren(el.cloneNode(true));d.showModal()});document.getElementById('close').onclick=()=>d.close();d.addEventListener('click',e=>{if(e.target===d)d.close()});</script></html>''')
    content = ''.join(parts)
    output = report_dir / 'consensus_review.html'
    output.write_text(content, encoding='utf-8')
    result = {'models': models, 'dataset_hash': summary['dataset_hash'], 'excluded_reviewed_sample_ids': sorted(reviewed),
              'three_model_count': 0, 'two_or_more_model_count': len(selected), 'items': manifest,
              'html_sha256': hashlib.sha256(content.encode()).hexdigest()}
    (report_dir / 'consensus_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    result = export_consensus()
    print(json.dumps({'two_models':result['two_or_more_model_count'],'html':str(REPORT/'consensus_review.html')},ensure_ascii=False))


if __name__ == '__main__':
    main()
