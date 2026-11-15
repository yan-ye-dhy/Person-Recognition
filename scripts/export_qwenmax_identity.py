"""Export a shareable Qwen Max name-only review using saved real responses."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcrb.multi.identity_review import build_identity_review
from vcrb.multi.parsing import bbox_valid
from scripts.export_qwenmax_review import (
    CSS, DEFAULT_OUTPUT, MODEL_ID, RUN_ID, POSE, VISIBILITY,
    detail_table, dump, esc, overlay, prediction_table, raw_persons, table,
)

EXPERIMENTS = {'A':'编号红框，不提供作品名', 'B':'编号红框，提供作品名', 'C':'无框原图，提供作品名'}


def aggregate(names: list[dict], strict: list[dict]) -> dict:
    targets = sum(r['target_count'] for r in names)
    correct = sum(r['correct'] for r in names)
    returned = sum(r['named_predictions'] for r in names)
    return {
        'images':len(names), 'target_count':targets, 'correct':correct,
        'accuracy':correct/targets if targets else None,
        'named_predictions':returned, 'extra_names':sum(r['extra_names'] for r in names),
        'name_precision':correct/returned if returned else None,
        'all_target_names_matched_images':sum(r['correct']==r['target_count'] for r in names),
        'strict_correct':sum(r['identity_correct'] for r in strict),
        'strict_targets':sum(r['verified_count'] for r in strict),
        'strict_invalid_images':sum(not r['parse_ok'] for r in strict),
        'strict_image_correct':sum(r['image_correct'] is True for r in strict),
    }


def rate(n: int, d: int, unit: str='人') -> str:
    return f'<strong>{n/d:.2%}</strong><small>{n} / {d} {esc(unit)}</small>' if d else '—'


def make_metrics(context: dict) -> dict:
    names, rows = context['name_rows'], context['rows']
    works = dict.fromkeys((s['work_slug'],s['work_name']) for s in context['samples'].values())
    scopes = {
        'all_four':{'label':'全部四部','slugs':[s for s,_ in works]},
        'three_works':{'label':'去掉《功夫女足》的三部','slugs':[s for s,_ in works if s!='gongfunvzu']},
    }
    result = {'scopes':{}, 'by_work':{}}
    for key, scope in scopes.items():
        result['scopes'][key] = {**scope, 'experiments':{}}
        for exp in EXPERIMENTS:
            predicate = lambda r:r['experiment']==exp and r['work_slug'] in scope['slugs']
            result['scopes'][key]['experiments'][exp] = aggregate([r for r in names if predicate(r)],[r for r in rows if predicate(r)])
    for slug, title in works:
        result['by_work'][slug] = {'label':title, 'experiments':{}}
        for exp in EXPERIMENTS:
            predicate = lambda r:r['experiment']==exp and r['work_slug']==slug
            result['by_work'][slug]['experiments'][exp] = aggregate([r for r in names if predicate(r)],[r for r in rows if predicate(r)])
    return result


def make_format_comparison(context: dict) -> dict:
    """Keep protocol penalties, ignored formatting, and deleted rows distinct."""
    names={(r['experiment'],r['sample_id']):r for r in context['name_rows']}
    rows=[r for r in context['rows'] if r['experiment']=='C']
    scopes={}
    for key in ('all_four','three_works'):
        selected=[r for r in rows if key=='all_four' or r['work_slug']!='gongfunvzu']
        legal=[r for r in selected if r['parse_ok']]
        failed=[r for r in selected if not r['parse_ok']]
        targets=sum(r['verified_count'] for r in selected)
        legal_targets=sum(r['verified_count'] for r in legal)
        strict_correct=sum(r['identity_correct'] for r in selected)
        name_correct=sum(names[('C',r['sample_id'])]['correct'] for r in selected)
        legal_name_correct=sum(names[('C',r['sample_id'])]['correct'] for r in legal)
        scopes[key]={
            'images':len(selected),'targets':targets,'invalid_format_images':len(failed),
            'invalid_format_targets':targets-legal_targets,
            'strict':{'correct':strict_correct,'targets':targets,'rate':strict_correct/targets},
            'ignore_format':{'correct':name_correct,'targets':targets,'rate':name_correct/targets},
            'delete_invalid_images':{'images':len(legal),'correct':legal_name_correct,'targets':legal_targets,'rate':legal_name_correct/legal_targets},
            'correct_names_in_invalid_responses':sum(names[('C',r['sample_id'])]['correct'] for r in failed),
        }
    issues=Counter(issue.split(':')[0] for r in rows if not r['parse_ok'] for issue in r['issues'])
    boxes=[]
    for row in rows:
        if row['parse_ok']:
            continue
        raw=context['raw_records'][('C',row['sample_id'])]['response']['raw_text']
        for prediction_index,person in enumerate(raw_persons(raw)):
            box=person.get('bbox') if isinstance(person,dict) else None
            if bbox_valid(box):
                continue
            numeric=isinstance(box,list) and len(box)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in box)
            boxes.append({'sample_id':row['sample_id'],'prediction_index':prediction_index,'bbox':box,
                          'integer_valued':numeric and all(float(v).is_integer() for v in box),
                          'out_of_range':numeric and any(v<0 or v>1 for v in box),
                          'order_invalid':numeric and (box[0]>=box[2] or box[1]>=box[3])})
    return {'experiment':'C','scopes':scopes,'issues':dict(issues),
            'invalid_boxes':len(boxes),'integer_valued_boxes':sum(b['integer_valued'] for b in boxes),
            'noninteger_valued_boxes':sum(not b['integer_valued'] for b in boxes),
            'out_of_range_boxes':sum(b['out_of_range'] for b in boxes),
            'order_invalid_boxes':sum(b['order_invalid'] for b in boxes),'box_details':boxes}


def render_format_comparison(comparison: dict) -> str:
    rows=[]
    for key,title in [('all_four','全部四部'),('three_works','去掉功夫女足的三部')]:
        scope=comparison['scopes'][key]
        strict,ignored,deleted=(scope[k] for k in ('strict','ignore_format','delete_invalid_images'))
        rows.append([esc(title)+f'<small>原范围{scope["images"]}张 / {scope["targets"]}人</small>',
                     rate(strict['correct'],strict['targets']),rate(ignored['correct'],ignored['targets']),
                     rate(deleted['correct'],deleted['targets'])+f'<small>仅剩{deleted["images"]}张；移除{scope["invalid_format_images"]}张</small>'])
    failed=comparison['scopes']['all_four']['invalid_format_images']
    invalid_targets=comparison['scopes']['all_four']['invalid_format_targets']
    recovered=comparison['scopes']['all_four']['correct_names_in_invalid_responses']
    all_scope=comparison['scopes']['all_four']
    strict_correct=all_scope['strict']['correct']
    name_correct=all_scope['ignore_format']['correct']
    legal=all_scope['delete_invalid_images']
    examples=[]
    for sid,title in [('kuangbiao_0002','大整数坐标'),('gongfunvzu_0021','大数与小数混用'),('jianlai_0033','坐标次序也无效')]:
        item=next(b for b in comparison['box_details'] if b['sample_id']==sid)
        examples.append([esc(title),'<code>'+esc(json.dumps(item['bbox'],ensure_ascii=False))+'</code>',esc(sid)])
    return '<section class="panel comparison" id="format-comparison"><h2>C组：计入格式失败、忽略格式、删除失败图片</h2><p>前两列保留相同的全部目标，比较原协议与仅姓名口径；最后一列真正移除格式失败图片，分母随之改变。</p>'+table(['作品范围','按原协议：格式失败记失败','忽略格式：全部目标只看姓名','删除格式失败图片：仅合规子集姓名'],rows)+f'<div class="notice"><strong>这批C组的{failed}张格式失败全部涉及bbox。</strong>其中有{invalid_targets}个真值目标；原始回答中仍能读出{recovered}个正确角色姓名，姓名口径因此从{strict_correct}增加到{name_correct}。姓名答对并不证明这批非法坐标已经正确还原。</div><p class="muted">第一列是“按原协议定位且姓名正确的人数÷全部目标数”，属于端到端人物召回，不是F1或整图全对率。第二列忽略坐标和编号对应。第三列只看保留下来的合规图片，不能当作完整样本上的准确率，也不能与第二列混称为“去掉格式影响”。</p><h3>坐标具体出了什么问题</h3><p>这46条回答的JSON可读取，失败发生在bbox数值校验。提示要求归一化[x1,y1,x2,y2]：每个值都在0～1之间，且x1&lt;x2、y1&lt;y2。模型有时返回较大的整数坐标样式，有时将大数与0～1小数混在同一框内。</p>'+table(['常见问题','实际返回bbox','样本'],examples)+f'<p>共{comparison["invalid_boxes"]}个非法框，全部存在超出0～1范围的问题；{comparison["integer_valued_boxes"]}个框的四个坐标均为整数值（包含1.0这类写法），另{comparison["noninteger_valued_boxes"]}个混有小数。另有{comparison["order_invalid_boxes"]}个框同时存在左右坐标次序错误，属于重叠问题，不能把分类数量相加。</p><p>因此不能将所有非法框统一除以1000后宣布修复。需要先确认整框的坐标约定，再检验换算后的框是否真正对准人物；混合尺度、次序无效的输出应单独处理。</p><h3>对项目目标是否重要</h3><p>项目面向车载影视问答，包括“画面中的几个人是谁”“左边的人是谁”“左右两人是什么关系”。只需列角色名时，姓名命中率最直接；涉及左右指代和关系查询时，<strong>人物与位置的对应是关键</strong>，否则可能查到错误人物的关系。</p><p>0～1还是0～1000属于接口约定，可以通过明确协议和经过验证的转换来适配；框有没有对准人物、姓名有没有对应正确，则是任务能力。建议分别呈现姓名识别、人物定位与对应、输出可用率，保留原协议成绩作为可直接接入的指标。</p><p class="muted">本批格式合规率为{legal["images"]/all_scope["images"]:.2%}（{legal["images"]}/{all_scope["images"]}张）。合规C输出中，{legal["targets"]}个已标主要人物在IoU≥0.5下均有定位匹配；这不能证明{failed}张格式失败图片的框都可恢复。报告目前没有修改这些坐标，也没有重新调用模型。</p></section>'


EXTRA_CSS = """
.comparison th,.comparison td{text-align:center}.comparison td:first-child,.comparison th:first-child{text-align:left}.comparison strong{font-size:22px;color:#075d86}.mini-bars{display:flex;flex-direction:column;gap:15px;margin:20px 0}.bar-line{display:grid;grid-template-columns:115px 1fr 85px;gap:10px;align-items:center;font-size:14px}.track{background:#e4edf3;height:20px;border-radius:6px;overflow:hidden}.fill{height:100%;background:#126e93}.fill.three{background:#087f5b}.bar-legend{display:flex;gap:20px;font-size:14px}.blue{color:#126e93}.green{color:#087f5b}.correction-image{width:100%;max-height:390px;object-fit:contain;background:#14222d;border-radius:8px}.raw-names{background:#f0f6fa;padding:10px;border-radius:6px;overflow-wrap:anywhere}.small-table strong{font-size:16px}.diagnostic{border-left:4px solid #075d86;padding:12px 16px;background:#eef7fc}.case h2{font-size:19px}.case details summary{font-size:14px}.case pre{max-height:330px}.correction-panel summary{font-size:19px}.method li{margin:8px 0}@media(max-width:650px){.bar-line{grid-template-columns:90px 1fr 65px;font-size:12px}.comparison strong{font-size:18px}}
"""

JS = """
const cards=[...document.querySelectorAll('.case')];
const scope=document.getElementById('scope');
const work=document.getElementById('work');
const result=document.getElementById('result');
const search=document.getElementById('search');
function applyFilters(){
 let shown=0;
 for(const card of cards){
  const visible=(scope.value!=='three_works'||card.dataset.work!=='gongfunvzu')&&
   (!work.value||card.dataset.work===work.value)&&
   (!result.value||(result.value==='complete'&&card.dataset.complete==='true')||
    (result.value==='wrong'&&card.dataset.complete==='false')||
    (result.value==='format'&&card.dataset.format==='false'))&&
   (!search.value.trim()||card.dataset.search.includes(search.value.trim().toLocaleLowerCase()));
  card.hidden=!visible;shown+=Number(visible);
 }
 document.getElementById('visible-count').textContent=`显示 ${shown} / ${cards.length} 张 · Qwen Max C组`;
 document.getElementById('empty').hidden=shown!==0;
}
for(const elem of [scope,work,result])elem.addEventListener('change',applyFilters);
search.addEventListener('input',applyFilters);
document.getElementById('reset').addEventListener('click',()=>{scope.value='all_four';work.value='';result.value='';search.value='';applyFilters();});
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{
 input.closest('.case').querySelector('.image-wrap').classList.toggle(`hide-${input.dataset.layer}`,!input.checked);
}));
const dialog=document.getElementById('zoom-dialog');
document.querySelectorAll('.zoom').forEach(button=>button.addEventListener('click',()=>{
 const card=button.closest('.case');
 document.getElementById('zoom-title').textContent=card.querySelector('h2').textContent;
 document.getElementById('zoom-content').replaceChildren(card.querySelector('.image-wrap').cloneNode(true));
 dialog.showModal();
}));
document.getElementById('close-zoom').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
applyFilters();
"""


def render_comparison(metrics: dict) -> str:
    all_exp = metrics['scopes']['all_four']['experiments']
    three_exp = metrics['scopes']['three_works']['experiments']
    rows = []
    bars = []
    for exp, title in EXPERIMENTS.items():
        a, t = all_exp[exp], three_exp[exp]
        rows.append([f'<strong>{exp}</strong><small>{esc(title)}</small>', rate(a['correct'],a['target_count']),
                     rate(t['correct'],t['target_count']), f'{(t["accuracy"]-a["accuracy"])*100:.2f} 个百分点'])
        for data, suffix, cls in [(a,'全部四部',''),(t,'其余三部','three')]:
            bars.append(f'<div class="bar-line"><span>{exp} · {suffix}</span><div class="track"><div class="fill {cls}" style="width:{data["accuracy"]*100:.6f}%"></div></div><strong>{data["accuracy"]:.2%}</strong></div>')
    return '<section class="panel comparison"><h2>姓名识别：全部四部与其余三部</h2>' + table(['实验','全部四部 · 160张 / 207人','去掉功夫女足 · 120张 / 157人','差值'],rows) + '<div class="mini-bars">' + ''.join(bars) + '</div><p class="muted">同一套回答、相同规则，仅改变统计作品范围；没有重新调用模型。这里按目标人物数汇总，不对四部作品的百分比简单求平均。</p></section>'


def render_method(metrics: dict, context: dict) -> str:
    rows = []
    extra_rows = []
    for scope_key, scope in metrics['scopes'].items():
        for exp in EXPERIMENTS:
            st = scope['experiments'][exp]
            old = [r for r in context['original_rows'] if r['experiment']==exp and r['work_slug'] in scope['slugs']]
            rows.append([esc(scope['label']), exp,
                         rate(sum(r['identity_correct'] for r in old),sum(r['verified_count'] for r in old)),
                         rate(st['strict_correct'],st['strict_targets']),rate(st['correct'],st['target_count']),
                         str(st['strict_invalid_images'])+'张'])
            extra_rows.append([esc(scope['label']),exp,str(st['named_predictions']),str(st['extra_names']),
                               rate(st['correct'],st['named_predictions'],'条姓名'),f'{st["all_target_names_matched_images"]} / {st["images"]} 张'])
    return '<section class="panel"><h2>统计口径</h2><div class="diagnostic"><strong>姓名命中率＝匹配正确的角色目标数 ÷ 全部已确认角色目标数。</strong><br>忽略坐标、编号及其格式；同张图内按角色姓名和已确认别名一对一匹配。一条返回姓名最多匹配一个目标，重复回答不能反复得分。null、漏答、错名仍未命中，不从分母删除。</div><p>这个口径回答“画面中的角色名字有没有认出来”，不判断姓名是否对应到正确位置。多人图中交换了姓名与编号，姓名命中率可能正确，原协议下的人物对应仍可能错误。</p><details class="method"><summary>与原协议得分比较，以及额外答案</summary><p>以下“原协议正确人物/全部目标”保留坐标或编号约束。C组这一列是端到端人物召回口径，并非匹配后身份准确率，也不是F1。姓名命中率单独诊断角色命名能力，不替换原始正式结果。</p>' + table(['范围','实验','原标注 · 原协议','更正标注 · 原协议','更正标注 · 仅姓名','原格式失败图片'],rows) + '<p>多报姓名不会额外增加正确目标数，但姓名命中率本身不扣罚多报；因此同时列出“姓名精确率＝正确命中数÷返回的非空有效姓名数”。“未命中目标的姓名”包含回答错名，以及没有对应主要目标的背景或重复姓名，不全是额外检测到的人物，也不据此认定每条都是模型幻觉。</p>' + table(['范围','实验','返回有效姓名条数','未命中目标的姓名条数','姓名精确率','全部目标姓名均命中图片'],extra_rows) + '<p>“全部目标姓名均命中”只检查目标是否齐全，允许另有额外答案；它不是原协议整图完全正确率。</p></details></section>'


def render_correction(context: dict, gallery: dict) -> str:
    if not context['corrections']:
        return ''
    parts=['<section class="panel correction-panel"><h2>已更正的角色标注</h2><p>以下列出用户明确确认的标注更正；图片、框与原始模型回答保留，报告按更正后的身份重新统计。</p>']
    for correction in context['corrections']:
        sid,pid=correction['sample_id'],correction['person_id']
        old_name,new_name=correction['old_character'],correction['new_character']
        entry=gallery[sid]['experiments']['B']
        image_bytes=(ROOT/'results/multi/input_gallery'/entry['image']).read_bytes()
        expected=context['raw_records'][('B',sid)]['request']['image_meta']['encoded_sha256']
        if hashlib.sha256(image_bytes).hexdigest()!=expected:
            raise ValueError('Correction illustration differs from original input')
        parts.append('<details class="correction-item" data-sample="'+esc(sid)+'" data-person-id="'+str(pid)+'"><summary>'+esc(sid)+f' 人物{pid}：'+esc(old_name)+' → '+esc(new_name)+'</summary><img class="correction-image" alt="'+esc(sid)+' · 用户确认角色'+esc(new_name)+'" src="data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode('ascii')+'">')
        rows=[]
        for exp in EXPERIMENTS:
            original=next(r for r in context['original_rows'] if r['sample_id']==sid and r['experiment']==exp)
            named=next(r for r in context['name_rows'] if r['sample_id']==sid and r['experiment']==exp)
            person=next(p for p in named['person_details'] if p['person_id']==pid)
            names=[p.get('character') for p in raw_persons(context['raw_records'][(exp,sid)]['response']['raw_text'])]
            result='<span class="good">姓名正确</span>' if person['outcome']=='correct' else '<span class="bad">姓名未命中</span>'
            rows.append([exp,esc('、'.join('null' if n is None else str(n) for n in names)),
                         '原坐标格式失败' if not original['parse_ok'] else '原协议在旧标注上评分',result])
        parts.append(table(['实验','模型原始姓名列表','原评测状态','本目标更正后仅姓名'],rows)+'</details>')
    parts.append('<p class="muted">原始运行快照和历史发布结果保留。当前工作数据集已应用更正，本报告在冻结快照上叠加相同的明确更正统计。</p></section>')
    return ''.join(parts)


def render_name_analysis(sample: dict, named: dict, strict: dict, predictions: list[dict]) -> str:
    """Explain name coverage without inventing a physical association for wrong names."""
    tags={p['person_id']:p for p in strict['person_details'] if p.get('person_id') is not None}
    people={p['person_id']:p for p in sample['persons']}
    rows=[]
    for person in named['person_details']:
        correct=person['outcome']=='correct'
        gt=person['gt_character']
        returned=person.get('predicted')
        index=person.get('prediction_index')
        ident=f'真值{person["person_id"]}'
        if correct and index is not None:
            ident+=f'<small>命中返回条目{index+1}</small>'
        if correct:
            answer='null' if returned is None else str(returned)
            explanation=f'返回“{answer}”，按已确认的角色名或别名规则命中“{gt}”。'
            if not strict['parse_ok']:
                explanation+='原坐标格式失败，不影响本项姓名计为正确。'
        elif len(named['person_details'])==len(predictions)==1:
            actual=predictions[0].get('character')
            answer='null' if actual is None else str(actual)
            explanation=f'应为“{gt}”，返回'+('null，未确定姓名。' if actual is None else f'“{answer}”，没有命中角色名或已确认别名。')
        else:
            answer='未命中；实际姓名列表见上方'
            explanation=f'没有可供“{gt}”一对一命中的返回姓名；不按返回顺序强行对应人物。'
        role=esc(gt)
        aliases=people[person['person_id']].get('character_aliases',[])
        if aliases:
            role+='<small>可接受别名：'+esc('、'.join(aliases))+'</small>'
        tag=tags.get(person['person_id'],{})
        rows.append([ident,role,esc(answer),'<span class="good">姓名正确</span>' if correct else '<span class="bad">姓名未命中</span>',
                     esc(explanation),esc(POSE.get(tag.get('face_orientation'),'—'))+'<small>'+esc(VISIBILITY.get(tag.get('visibility'),'—'))+'</small>'])
    return '<div class="name-analysis">'+table(['人物参考','真值角色','返回姓名（仅姓名匹配）','姓名判定','具体说明','朝向 / 可见性'],rows)+'</div>'


def render_details(context: dict, gallery: dict) -> str:
    strict = {(r['experiment'],r['sample_id']):r for r in context['rows']}
    cards=[]
    c_rows=[r for r in context['name_rows'] if r['experiment']=='C']
    if len(c_rows)!=160 or len({r['sample_id'] for r in c_rows})!=160:
        raise ValueError('Expected 160 unique Qwen Max C-group images')
    for row in sorted(c_rows,key=lambda r:r['sample_id']):
        sid, exp = row['sample_id'], row['experiment']
        source = strict[(exp,sid)]
        sample = context['samples'][sid]
        raw = context['raw_records'][(exp,sid)]
        predictions = raw_persons(raw['response']['raw_text'])
        names = [p.get('character') for p in predictions if isinstance(p,dict)]
        names_text = '、'.join('null' if p is None else str(p) for p in names) or '（无姓名条目）'
        search = ' '.join([sid,sample['work_name'],names_text]+[str(p['character']) for p in sample['persons']]).lower()
        complete = row['correct']==row['target_count']
        entry=gallery[sid]['experiments']['C']
        image_bytes=(ROOT/'results/multi/input_gallery'/entry['image']).read_bytes()
        expected=raw['request']['image_meta']['encoded_sha256']
        if hashlib.sha256(image_bytes).hexdigest()!=expected or expected!=entry['encoded_sha256']:
            raise ValueError(f'C image differs from actual API input: {sid}')
        width,height=raw['request']['image_meta']['encoded_size']
        cards.append(f'<article class="case" id="{esc(sid)}_{exp}" data-work="{esc(row["work_slug"])}" data-experiment="{exp}" data-complete="{str(complete).lower()}" data-format="{str(source["parse_ok"]).lower()}" data-search="{esc(search)}"><div class="case-head"><h2>{esc(sid)} · Qwen Max C组 <small>{esc(sample["work_name"])}</small></h2><div class="tags"><span class="tag">姓名 {row["correct"]} / {row["target_count"]} 命中</span>' + (f'<span class="tag">另有{row["extra_names"]}条未命中目标的姓名</span>' if row['extra_names'] else '') + '</div></div>')
        cards.append('<div class="picture-controls"><button class="zoom" type="button">放大图片</button><label class="gt-legend"><input type="checkbox" data-layer="gt" checked>绿色实线：人工真值框</label>')
        cards.append('<label class="pred-legend"><input type="checkbox" data-layer="pred" checked'+('' if source['parse_ok'] else ' disabled')+'>橙色虚线：原合法预测框</label></div>')
        cards.append(f'<div class="image-wrap"><img class="input-image" loading="lazy" decoding="async" width="{width}" height="{height}" data-sha256="{expected}" alt="{esc(sid)} · Qwen Max C组实际无框输入" src="data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode('ascii')+'">'+overlay(sample,source,predictions,[width,height])+'</div>')
        cards.append('<p class="caption">底图是实际提交给C组的无框JPEG，叠加框仅供本报告人工复核，未发送给模型。上方框线不参与下方仅姓名评分。'+('原返回坐标格式无效，本图只显示人工真值框。' if not source['parse_ok'] else '灰框为原评测忽略的背景预测。')+'</p>')
        cards.append('<p class="raw-names">模型实际姓名列表：<strong>'+esc(names_text)+'</strong></p>')
        cards.append(render_name_analysis(sample,row,source,predictions))
        cards.append('<details class="protocol-details"><summary>原协议定位与格式分析（仅供参考）</summary><p>以下保留原定位与格式口径，不改变上方忽略坐标、编号的姓名判定。</p>')
        if not source['parse_ok']:
            cards.append('<div class="notice"><strong>原坐标格式失败：</strong>返回bbox不符合0–1归一化要求。原协议没有自动修正坐标；本报告仍读取原始姓名单独统计，格式失败不等同于姓名错误。<br>解析记录：<code>'+esc('；'.join(source['issues']))+'</code></div>')
        cards.append(detail_table(sample,source)+prediction_table(source,predictions)+'</details>')
        cards.append('<details class="raw-response"><summary>模型原始回答与当次输入条件</summary><p>' + esc(EXPERIMENTS[exp]) + '；该回答原协议格式' + ('合规' if source['parse_ok'] else '不合规') + '。</p><pre>' + esc(raw['response']['raw_text']) + '</pre><details><summary>当次提示词与图片指纹</summary><pre>' + esc(dump({'system':raw['request']['system'],'user':raw['request']['user'],'image_meta':raw['request']['image_meta']})) + '</pre></details></details></article>')
    works=dict.fromkeys((s['work_slug'],s['work_name']) for s in context['samples'].values())
    return '<section class="panel"><h2>Qwen Max · C组逐图分析</h2><p>以下展示C组全部160张图片及逐人物姓名判定，含识别正确与错误的样本。每张配有实际输入图片、人工真值框、模型姓名、具体说明和可展开的原始回答。</p><p class="muted">前面的A/B/C对比保持完整；逐图部分仅展示C组。姓名按一对一匹配，不使用坐标或条目顺序强行对应人物。</p></section><div class="toolbar" id="detail-top"><label>逐图作品范围<select id="scope"><option value="all_four">全部四部</option><option value="three_works">去掉功夫女足的三部</option></select></label><label>作品<select id="work"><option value="">全部作品</option>'+''.join(f'<option value="{esc(slug)}">{esc(title)}</option>' for slug,title in works)+'</select></label><label>结果<select id="result"><option value="">全部结果</option><option value="wrong">目标姓名未全命中</option><option value="complete">目标姓名全命中</option><option value="format">原坐标格式失败</option></select></label><label>搜索编号或姓名<input type="search" id="search" placeholder="例如 邓子越 或 qingyunian2_0029"></label><button id="reset" type="button">重置筛选</button><strong id="visible-count"></strong></div><p id="empty" hidden>没有符合条件的图片。</p>'+''.join(cards)


def export_identity(output_dir: Path=DEFAULT_OUTPUT) -> dict:
    context=build_identity_review(ROOT)
    metrics=make_metrics(context)
    format_comparison=make_format_comparison(context)
    gallery={r['sample_id']:r for r in json.loads((ROOT/'results/multi/input_gallery/manifest.json').read_text())['rows']}
    by_work=[]
    for slug,data in metrics['by_work'].items():
        row=[esc(data['label'])]
        for exp in EXPERIMENTS:
            st=data['experiments'][exp]
            row.append(rate(st['correct'],st['target_count']))
        by_work.append(row)
    parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Max · 仅姓名识别与三部作品对照</title><style>' + CSS + EXTRA_CSS + '</style></head><body><main>',
           '<header class="intro"><h1>Qwen Max · 仅姓名识别复核</h1><p>忽略坐标与编号格式，区分全部四部和去掉《功夫女足》的其余三部；已应用用户确认的角色标注更正。</p><p class="muted">模型：<code>' + MODEL_ID + '</code> · 对比统计使用既有回答480次 · 逐图展示C组160张 · 本次新增模型调用0次</p><div class="notice">这是姓名命中率诊断：不把格式失败图片直接剔除，也不判断姓名与位置或编号的对应。原始正式结果保留，逐项统计口径见下方。</div><p>本文件的图表、C组全部图片、逐图分析和原始回答均内嵌。分享完整HTML给他人，下载后用现代浏览器打开即可，不需要本项目、服务器、API Key或账号。</p></header>',
           render_format_comparison(format_comparison),render_comparison(metrics),render_correction(context,gallery),
           '<section class="panel comparison"><h2>四部作品分别表现</h2>' + table(['作品','A · 仅姓名','B · 仅姓名','C · 仅姓名'],by_work) + '<p class="muted">《功夫女足》仍为0 / 50，错误来自返回姓名不匹配或null；忽略坐标格式后没有恢复正确姓名。其余三部统计是另一个作品范围，未删除原四部数据。</p></section>',
           render_method(metrics,context),render_details(context,gallery),
           '<footer><details><summary>审计来源与规则</summary><pre>' + esc(dump({'run_id':RUN_ID,'model_id':MODEL_ID,'source_dataset_hash':context['source_dataset_hash'],'review_dataset_hash':context['review_dataset_hash'],'corrections':context['corrections'],'new_api_calls':0,'rule':'raw persons[].character only; one-to-one maximum name matching; coordinates/IDs ignored; null/missing targets retained; confirmed aliases accepted'})) + '</pre></details><p>原始调用与冻结快照保持不变；本报告离线计算，不包含API凭据。</p></footer></main><dialog id="zoom-dialog"><div class="dialog-head"><strong id="zoom-title"></strong><button id="close-zoom" type="button">关闭（Esc）</button></div><div id="zoom-content" class="dialog-content"></div></dialog><script>' + JS + '</script></body></html>']
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'qwenmax_identity_summary.html'
    path.write_text(''.join(parts),encoding='utf-8')
    manifest={'run_id':RUN_ID,'model_id':MODEL_ID,'generated_at':datetime.now(timezone.utc).isoformat(),
              'source_dataset_hash':context['source_dataset_hash'],'review_dataset_hash':context['review_dataset_hash'],
              'matcher_source':context['matcher_source'],'diagnostic_summary':context['diagnostic_summary'],
              'format_comparison':format_comparison,
              'corrections':context['corrections'],'new_api_calls':0,'metrics':metrics,'name_rows':context['name_rows'],
              'detail_view':{'model_id':MODEL_ID,'experiment':'C','image_count':160,'embedded_input_images':True,'image_sha256_verified_against_raw_requests':True},
              'html_file':path.name,'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (output_dir/'qwenmax_identity_summary.json').write_text(dump(manifest)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=export_identity(args.output)
    print(dump({'output':str(args.output/result['html_file']),'scopes':result['metrics']['scopes'],'new_api_calls':0}))
