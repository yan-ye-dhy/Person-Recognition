"""Export fifteen annotated film samples as one offline screenshot-friendly HTML."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='multi_v2_20261113_075729_2e4f15'
MODEL_ID='qwen3.8-max'
OUTPUT=ROOT/'reports/2026-11-15-qwenmax-review/gongfunvzu_web_test_15.html'
SAMPLE_IDS=tuple('gongfunvzu_'+n for n in (
    '0003','0005','0006','0007','0008','0012','0013','0016','0020','0022','0023','0029','0030','0038','0040',
))
CAPTIONS={'A':'编号红框 · 不提供作品名','B':'编号红框 · 提供作品名','C':'无框原图 · 提供作品名'}
POSE={'front':'正脸','three_quarter':'斜侧脸','profile':'侧脸','not_visible':'面部不可见'}
VISIBILITY={'clear':'清晰','occluded_or_blurry':'遮挡或模糊'}


def esc(value):
    return html.escape(str(value if value is not None else ''),quote=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CSS="""
:root{font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;color:#183047;background:#edf3f8}*{box-sizing:border-box}body{margin:0}main{max-width:1740px;margin:auto;padding:24px}h1{font-size:29px;margin:0 0 12px}h2{font-size:21px;margin:0}h2 small{font-size:16px;color:#5a6e81;margin-left:10px}h3{font-size:23px;margin:0 0 10px}h3 span{font-size:14px;font-weight:400;color:#496984}p{line-height:1.7}.intro,.sample{background:white;border:1px solid #d2e0eb;border-radius:15px;padding:22px;margin-bottom:22px}.intro p{margin:8px 0}.notice{border-left:4px solid #148274;background:#eef9f6;padding:10px 15px}.muted{color:#5a6e81}.toolbar{position:sticky;top:0;z-index:3;display:flex;flex-wrap:wrap;gap:15px;align-items:end;background:#f7fbfff5;border:1px solid #d2e0eb;border-radius:12px;padding:15px;margin-bottom:22px}label{display:flex;flex-direction:column;gap:5px;font-size:14px}button,input,select{font:inherit;border:1px solid #acc3d5;border-radius:6px;padding:9px 12px;background:white;color:#183047}button{cursor:pointer}button:hover{color:#075d8e;border-color:#247aad}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #62a5d6;outline-offset:3px}input[type=search]{min-width:260px}#count{margin-left:auto;font-weight:700}article[hidden]{display:none}.sample-header{display:flex;justify-content:space-between;gap:15px;align-items:start;margin-bottom:18px}.sample-header p{font-size:14px;color:#536d83;margin:6px 0}.badge{font-size:13px;padding:4px 12px;background:#e7f6eb;color:#286440;border-radius:20px;white-space:nowrap}.experiments{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.image-button{display:block;width:100%;padding:0;background:#15212c;border:0;border-radius:6px;overflow:hidden}.image-button img{display:block;width:100%;height:auto}.zoom-link{display:block;padding:0;border:0;border-radius:0;margin-top:9px;text-decoration:underline;background:transparent;color:#08648e;font-size:13px}.human-annotations{background:#f4f9fd;border:1px solid #b4cfe2;border-radius:10px;padding:17px;margin-top:20px}.human-annotations h3{font-size:18px}.human-annotations h3 span{display:inline-block;margin-left:8px;font-size:13px}.people{display:flex;flex-wrap:wrap;gap:12px;list-style:none;margin:12px 0 0;padding:0}.person{border:1px solid #8ebea0;background:#eef9f1;border-radius:8px;padding:11px 14px;min-width:240px;max-width:100%;flex:0 1 480px}.person strong{font-size:17px}.person-state{font-size:14px;color:#506d82}.person-note{display:block;font-size:13px;line-height:1.65;color:#527189;margin-top:5px;overflow-wrap:anywhere}details{margin-top:16px}summary{font-size:14px;color:#476980;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 ui-monospace,monospace;background:#edf4f8;border-radius:6px;padding:13px;max-height:400px;overflow:auto}.prompt-block{margin-top:15px}.prompt-block h4{margin:0 0 8px}.copy-status{font-size:13px;color:#39715d;margin-left:10px}.footer{font-size:13px;color:#527189;margin:20px 0}body[data-mode=B] .experiment:not([data-experiment=B]),body[data-mode=C] .experiment:not([data-experiment=C]){display:none}body[data-mode=B] .experiments,body[data-mode=C] .experiments{grid-template-columns:minmax(0,1fr)}body[data-mode=B] .image-button,body[data-mode=C] .image-button{max-width:1400px}#empty{text-align:center;padding:30px}dialog{border:0;border-radius:9px;padding:14px;background:#111d29;color:white;max-width:98vw;max-height:98vh;width:max-content}dialog::backdrop{background:#08101a}dialog header{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:12px}#lightbox-image{display:block;max-width:94vw;max-height:84vh;width:auto;height:auto;object-fit:contain}dialog button{padding:6px 12px}dialog p{font-size:12px;color:#b7c8d7;margin:8px 0 0}.layout-note{font-size:13px;color:#527189}@media(max-width:1000px){main{padding:14px}.sample,.intro{padding:16px}.experiments{gap:9px}h3 span{display:block;font-size:12px}.toolbar{gap:10px}#count{margin-left:0}}@media(max-width:600px){main{padding:10px}.experiments{grid-template-columns:1fr;gap:20px}.sample-header{display:block}.badge{display:inline-block;margin-top:6px}.toolbar{position:static}input[type=search]{min-width:170px}h1{font-size:23px}h2{font-size:18px}h2 small{display:block;margin-left:0;margin-top:4px}.person{min-width:0;width:100%;flex-basis:100%}.human-annotations h3 span{margin-left:0;display:block}.intro,.sample{padding:13px}}@media print{.toolbar,.zoom-link,.footer,dialog{display:none}.sample{break-inside:avoid}.experiments{grid-template-columns:repeat(3,minmax(0,1fr))}}
"""

JS="""
const cards=[...document.querySelectorAll('.sample')];
const search=document.getElementById('search'),people=document.getElementById('people'),mode=document.getElementById('display-mode');
function filter(){let shown=0;const q=search.value.trim().toLocaleLowerCase();for(const card of cards){const n=Number(card.dataset.people);const visible=(!q||card.dataset.search.includes(q))&&(people.value==='all'||(people.value==='single'&&n===1)||(people.value==='multi'&&n>1));card.hidden=!visible;shown+=Number(visible);}document.getElementById('count').textContent=`${shown} / ${cards.length} 张`;document.getElementById('empty').hidden=shown!==0;}
search.addEventListener('input',filter);people.addEventListener('change',filter);
mode.addEventListener('change',()=>{document.body.dataset.mode=mode.value;});
document.getElementById('reset').addEventListener('click',()=>{search.value='';people.value='all';mode.value='ABC';document.body.dataset.mode='ABC';filter();});
const dialog=document.getElementById('lightbox'),image=document.getElementById('lightbox-image');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{const source=document.getElementById(button.dataset.zoom);image.src=source.src;image.alt='实验输入图片';document.getElementById('lightbox-title').textContent=button.dataset.title;dialog.showModal();}));
document.getElementById('close-lightbox').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{const text=document.getElementById(button.dataset.copy).textContent;const status=button.parentElement.querySelector('.copy-status');let copied=false;try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text);copied=true;}}catch(_){}if(!copied){const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();copied=document.execCommand('copy');area.remove();}status.textContent=copied?'已复制':'请手动选中下方文字复制';}));
filter();
"""


def render_card(index: int, sample: dict, tags: dict, inputs: dict, system_prompt: str) -> str:
    sid=sample['sample_id']
    search=' '.join([sid]+[p['character'] or '' for p in sample['persons']]).lower()
    parts=[f'<article class="sample" id="{sid}" data-sample="{sid}" data-people="{len(sample["persons"])}" data-search="{esc(search)}"><header class="sample-header"><div><h2>{index:02d} · {sid}<small>功夫女足</small></h2><p>已标{len(sample["persons"])}人 · 身份已核{len(sample["persons"])} · 编号与角色对照仅供人工核对</p></div><span class="badge">现有已确认标注</span></header><div class="experiments">']
    for experiment,title in CAPTIONS.items():
        entry=inputs[experiment]
        image_id=f'{sid}_{experiment}'
        zoom_title=f'样本{index:02d} · {experiment}组输入图'
        width,height=entry['size']
        parts.append(f'<section class="experiment" data-experiment="{experiment}"><h3>{experiment} <span>{title}</span></h3><button class="image-button" type="button" data-zoom="{image_id}" data-title="{esc(zoom_title)}" aria-label="放大样本{index:02d}的{experiment}组输入图片"><img id="{image_id}" src="data:image/jpeg;base64,{entry["base64"]}" width="{width}" height="{height}" loading="lazy" decoding="async" alt="样本{index:02d} · {experiment}组输入图片" data-sha256="{entry["sha256"]}"></button><button class="zoom-link" type="button" data-zoom="{image_id}" data-title="{esc(zoom_title)}">放大{experiment}输入图 ↗</button></section>')
    parts.append('</div><section class="human-annotations"><h3>编号与角色对照 <span>仅供人工审核；姓名不绘入图片、不用于模型输入</span></h3><ul class="people">')
    for person in sample['persons']:
        tag=tags[(sid,person['person_id'])]
        classification=POSE[tag['face_orientation']]+' · '+VISIBILITY[tag['visibility']]
        parts.append(f'<li class="person" data-person-id="{person["person_id"]}" data-character="{esc(person["character"])}"><strong>{person["person_id"]} {esc(person["character"])}</strong><span class="person-state"> · 已核实</span><span class="person-note">统计分类：{classification}</span><span class="person-note">{esc(person.get("note",""))}</span></li>')
    parts.append('</ul></section><details class="prompts"><summary>查看 A/B/C 实际输入提示词（可复制到网页版）</summary><p class="layout-note">下方复制版合并了原API的system与user内容，适合网页单输入框；网页模型、默认系统提示和设置可能与API不同。逐张新建对话，先保持联网与思考设置一致，再核对答案。</p>')
    for experiment in CAPTIONS:
        prompt_id=f'prompt_{sid}_{experiment}'
        combined=system_prompt+'\n\n'+inputs[experiment]['user_prompt']
        parts.append(f'<div class="prompt-block"><h4>{experiment}组</h4><button type="button" data-copy="{prompt_id}">复制{experiment}组提示词</button><span class="copy-status" role="status"></span><pre id="{prompt_id}">{esc(combined)}</pre></div>')
    parts.append('</details></article>')
    return ''.join(parts)


def export_html(output: Path=OUTPUT) -> dict:
    current_path=ROOT/'data/v2/dataset.json'
    current_bytes=current_path.read_bytes()
    samples={r['sample_id']:r for r in json.loads(current_bytes)['samples']}
    frozen={r['sample_id']:r for r in json.loads((ROOT/'results/multi'/RUN_ID/'dataset/dataset.json').read_text())['samples']}
    gallery_path=ROOT/'results/multi/input_gallery/manifest.json'
    gallery={r['sample_id']:r for r in json.loads(gallery_path.read_text())['rows']}
    tags={(r['sample_id'],r['person_id']):r for r in json.loads((ROOT/'data/v2/stratification.json').read_text())['persons']}
    cards=[];rows=[];roles=Counter();counts=Counter()
    with sqlite3.connect((ROOT/'results/multi/runs.db').as_uri()+'?mode=ro',uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        for index,sid in enumerate(SAMPLE_IDS,1):
            sample=samples[sid]
            if sample['work_slug']!='gongfunvzu' or sample['review_status']!='approved' or not sample['foreground_complete']:
                raise ValueError(f'Expected approved film sample: {sid}')
            if sample['image_sha256']!=frozen[sid]['image_sha256']:
                raise ValueError(f'Current image differs from completed test: {sid}')
            counts[len(sample['persons'])]+=1
            frozen_people={p['person_id']:p for p in frozen[sid]['persons']}
            for person in sample['persons']:
                if person['identity_status']!='verified' or person['review_status']!='approved':
                    raise ValueError(f'Expected confirmed role: {sid}')
                old=frozen_people[person['person_id']]
                tag=tags[(sid,person['person_id'])]
                if old['bbox']!=person['bbox'] or old['annotation_id']!=person['annotation_id']:
                    raise ValueError(f'Existing numbered input no longer matches current geometry: {sid}')
                if tag['image_sha256']!=sample['image_sha256'] or tag['annotation_id']!=person['annotation_id']:
                    raise ValueError(f'Person tag binding mismatch: {sid}')
                if tag['visibility']!='clear' or tag['face_orientation'] not in ('front','three_quarter'):
                    raise ValueError(f'Expected clear frontal/three-quarter sample: {sid}')
                roles[person['character']]+=1
            inputs={};system_prompt=None
            for experiment in CAPTIONS:
                raw_rows=connection.execute('SELECT raw_json FROM records WHERE run_id=? AND model_id=? AND experiment=? AND sample_id=?',(RUN_ID,MODEL_ID,experiment,sid)).fetchall()
                if len(raw_rows)!=1:
                    raise ValueError(f'Expected one saved request: {sid}/{experiment}')
                request=json.loads(raw_rows[0][0])['request']
                entry=gallery[sid]['experiments'][experiment]
                image_bytes=(gallery_path.parent/entry['image']).read_bytes()
                actual=sha(image_bytes)
                if actual!=request['image_meta']['encoded_sha256'] or actual!=entry['encoded_sha256'] or sample['image_sha256']!=request['image_meta']['source_sha256']:
                    raise ValueError(f'Input image SHA mismatch: {sid}/{experiment}')
                if entry['user_prompt']!=request['user']:
                    raise ValueError(f'Prompt mismatch: {sid}/{experiment}')
                if system_prompt is not None and system_prompt!=request['system']:
                    raise ValueError('Unexpected system prompt difference')
                system_prompt=request['system']
                inputs[experiment]={'base64':base64.b64encode(image_bytes).decode('ascii'),'sha256':actual,'size':request['image_meta']['encoded_size'],'user_prompt':request['user']}
            if inputs['A']['sha256']!=inputs['B']['sha256']:
                raise ValueError('A/B must use identical numbered JPEGs')
            cards.append(render_card(index,sample,tags,inputs,system_prompt))
            rows.append({'sample_id':sid,'persons':[{'person_id':p['person_id'],'character':p['character'],'bbox':p['bbox'],'annotation_id':p['annotation_id'],'face_orientation':tags[(sid,p['person_id'])]['face_orientation'],'visibility':tags[(sid,p['person_id'])]['visibility']} for p in sample['persons']],
                         'inputs':{e:{k:v for k,v in entry.items() if k!='base64'} for e,entry in inputs.items()}})
    if len(cards)!=15 or counts!=Counter({1:11,2:4}) or len(roles)!=5:
        raise ValueError('Expected 15 cases: eleven single, four double, five roles')
    manifest={'run_id':RUN_ID,'reference_model_id':MODEL_ID,'generated_at':datetime.now(timezone.utc).isoformat(),'sample_ids':list(SAMPLE_IDS),'sample_count':15,'embedded_image_count':45,'person_targets':sum(roles.values()),'person_count_distribution':dict(counts),'role_counts':dict(roles),'dataset_file_sha256':sha(current_bytes),'new_api_calls':0,'rows':rows}
    content='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>功夫女足 · 15张图片与人工标注</title><style>'+CSS+'</style></head><body data-mode="ABC"><main><section class="intro"><h1>功夫女足 · 15张图片与人工标注</h1><p>15个样本，11张单人、4张双人，共19个人物目标，覆盖5个角色。每张并排展示A/B编号红框图、C无框图，下方给出我们的已确认标注。</p><p>A/B使用完全相同的图片，文字提示是否提供作品名不同；C使用无框图片。全部图片与原API测试输入一致，姓名仅在下方人工对照中显示。</p><p class="notice"><strong>截图测试：</strong>点击任意图片或“放大输入图”，只截图片区域；下方角色对照用于你核对答案。A组不提供作品名，B/C提示词已包含作品名。</p><p class="muted">本页为清晰角色的定向测试样本，不能替代全40张正式分数。图片和提示词已内嵌，完整HTML可以独立离线打开、分享；本次未调用模型。</p></section><section class="toolbar" aria-label="图片筛选"><label>显示方式<select id="display-mode"><option value="ABC">A / B / C 并排</option><option value="B">只看编号图（A/B相同）</option><option value="C">只看C无框原图</option></select></label><label>人数<select id="people"><option value="all">全部图片</option><option value="single">单人图</option><option value="multi">多人图</option></select></label><label>编号或角色<input type="search" id="search" placeholder="例如 0012 或 钰珑"></label><button id="reset" type="button">重置筛选</button><output id="count" aria-live="polite">15 / 15 张</output></section>'+''.join(cards)+'<p id="empty" hidden>没有匹配图片，请调整筛选。</p><details class="footer"><summary>生成清单与来源</summary><pre>'+esc(json.dumps({k:v for k,v in manifest.items() if k!='rows'},ensure_ascii=False,indent=2))+'</pre></details></main><dialog id="lightbox" aria-label="只显示实验输入图片的放大窗口"><header><strong id="lightbox-title"></strong><button id="close-lightbox" type="button">关闭（Esc）</button></header><img id="lightbox-image" alt="实验输入图片"><p>截图时只取图片区域，角色对照未显示在此窗口。</p></dialog><script>'+JS+'</script></body></html>'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(content,encoding='utf-8')
    manifest['html_file']=output.name;manifest['html_sha256']=sha(output.read_bytes())
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    result=export_html(args.output)
    print(json.dumps({'output':str(args.output),'samples':result['sample_count'],'people':result['person_targets'],'embedded_images':result['embedded_image_count'],'new_api_calls':0},ensure_ascii=False))
