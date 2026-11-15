"""Read saved search-pilot responses and export a standalone, offline comparison."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.export_gongfunvzu_html import SAMPLE_IDS
from scripts.export_qwenmax_review import CSS, esc, table, overlay, raw_persons
from vcrb.annotation.rosters import load_roster
from vcrb.config import load_works
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.identity_review import build_identity_review, score_name_only
from vcrb.multi.parsing import parse_response
from vcrb.multi.schema import load_dataset
from vcrb.multi.scoring import score_sample

DEFAULT_RUN='multi_v2_20261115_130614_7348e9'
DEFAULT_OUTPUT=ROOT/'reports/2026-11-15-qwen-search-pilot'
MODEL='qwen3.8-max'


def jdump(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)


def search_evidence(raw):
    """A requested tool or an error object is not proof of retrieved sources."""
    if raw is None:
        return {'sources':[],'queries':[],'domains':[],'verified':False,'reported_calls':0,'requested_max':None,'cap_exceeded':False,'errors':[]}
    response=raw['response'];metadata=response.get('search_metadata') or {}
    sources={};queries=[];errors=[];tool_ids=set();reported=[]
    usage_count=(response.get('usage') or {}).get('server_tool_use',{}).get('web_search_requests')
    if type(usage_count) is int and usage_count>=0:
        reported.append(usage_count)
    for message in metadata.get('messages',[]):
        info=message.get('search_info') or {}
        count=(info.get('server_tool_use') or {}).get('web_search_requests')
        if type(count) is int and count>=0:
            reported.append(count)
        for tool in info.get('server_tool_uses',[]):
            if tool.get('name')!='web_search':
                continue
            tool_ids.add(tool.get('id') or jdump(tool))
            arguments=tool.get('input') or {}
            values=arguments.get('queries',arguments.get('query',[]))
            if isinstance(values,str):values=[values]
            queries.extend(value for value in values if isinstance(value,str))
        for block in info.get('web_search_tool_results',[]):
            content=block.get('content')
            if not isinstance(content,list):
                if content:
                    errors.append(content)
                continue
            for item in content:
                if not isinstance(item,dict) or item.get('type')!='web_search_result':
                    continue
                url=item.get('url')
                try:
                    parsed=urlsplit(url) if isinstance(url,str) else None
                except ValueError:
                    parsed=None
                if parsed and parsed.scheme in ('http','https') and parsed.hostname:
                    sources[url]={'url':url,'title':item.get('title') or parsed.hostname,'domain':parsed.hostname}
    calls=max(reported+[len(tool_ids)])
    requested=raw['request'].get('search_max_uses')
    return {'sources':list(sources.values()),'queries':list(dict.fromkeys(queries)),
            'domains':sorted({s['domain'] for s in sources.values()}),'verified':bool(sources),
            'reported_calls':calls,'requested_max':requested,
            'cap_exceeded':type(requested) is int and calls>requested,'errors':errors}


def metrics(records,which):
    names=[r[which]['names'] for r in records]
    strict=[r[which]['strict'] for r in records]
    targets=sum(n['target_count'] for n in names)
    correct=sum(n['correct'] for n in names)
    returned=sum(n['named_predictions'] for n in names)
    loc={key:sum(s['localization_'+key] for s in strict) for key in ('tp','fp','fn')}
    precision=loc['tp']/(loc['tp']+loc['fp']) if loc['tp']+loc['fp'] else None
    recall=loc['tp']/(loc['tp']+loc['fn']) if loc['tp']+loc['fn'] else None
    return {'images':len(records),'targets':targets,'name_correct':correct,'name_accuracy':correct/targets if targets else None,
            'named_predictions':returned,'extra_names':returned-correct,'name_precision':correct/returned if returned else None,
            'strict_identity_correct':sum(s['identity_correct'] for s in strict),
            'strict_identity_rate_all_targets':sum(s['identity_correct'] for s in strict)/targets if targets else None,
            'matched_verified':sum(s['matched_verified'] for s in strict),
            'localization':{**loc,'precision':precision,'recall':recall,'f1':2*loc['tp']/(2*loc['tp']+loc['fp']+loc['fn']) if 2*loc['tp']+loc['fp']+loc['fn'] else None},
            'legal_format_images':sum(s['parse_ok'] and s['api_state']=='ok' for s in strict),
            'api_ok_images':sum(s['api_state']=='ok' for s in strict),
            'all_target_names_images':sum(n['image_names_complete'] for n in names),
            'strict_image_correct':sum(s['image_correct'] is True for s in strict)}


def ratio(n,d,unit='人'):
    return f'<strong>{n/d:.2%}</strong><small>{n} / {d} {esc(unit)}</small>' if d else '—'


def summary_table(old,new,search):
    def loc(m):
        value=m['localization']
        p='—' if value['precision'] is None else f'{value["precision"]:.2%}'
        r='—' if value['recall'] is None else f'{value["recall"]:.2%}'
        f='—' if value['f1'] is None else f'{value["f1"]:.4f}'
        return f'{p} / {r} / {f}<small>TP{value["tp"]} / FP{value["fp"]} / FN{value["fn"]}</small>'
    return table(['同一批15张 / 19个人物','旧组：无搜索','新组：联网工具方案'],[
        ['仅姓名命中率',ratio(old['name_correct'],old['targets']),ratio(new['name_correct'],new['targets'])],
        ['位置与姓名均正确 / 全部目标',ratio(old['strict_identity_correct'],old['targets']),ratio(new['strict_identity_correct'],new['targets'])],
        ['定位 Precision / Recall / F1',loc(old),loc(new)],
        ['格式合规图片',ratio(old['legal_format_images'],old['images'],'张'),ratio(new['legal_format_images'],new['images'],'张')],
        ['全部目标姓名均命中',f'{old["all_target_names_images"]} / {old["images"]} 张',f'{new["all_target_names_images"]} / {new["images"]} 张'],
        ['按原协议整图全对',f'{old["strict_image_correct"]} / {old["images"]} 张',f'{new["strict_image_correct"]} / {new["images"]} 张'],
        ['未命中目标的返回姓名条数',str(old['extra_names']),str(new['extra_names'])],
        ['实际返回有效搜索来源的图片','0（未启用）',f'{search["verified_images"]} / {new["images"]} 张'],
    ])


STYLE="""
.comparison strong{font-size:23px;color:#08668e}.metric-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.metric-panel{padding:16px;background:#f0f6fa;border-radius:9px}.metric-panel h3{margin-bottom:10px}.source-list{line-height:1.8;overflow-wrap:anywhere}.source-list a{font-size:13px}.search-queries{background:#f0f6fa;padding:12px;border-radius:7px;white-space:pre-wrap;overflow-wrap:anywhere}.badge{font-size:13px;padding:5px 11px;border-radius:18px;background:#e5f5eb;color:#21623f}.badge.warn{background:#fff1db;color:#8b4c09}.case h2{font-size:20px}.case[hidden]{display:none}.cost{font-size:14px}.raw-names{background:#f3f7fa;padding:12px;border-radius:8px}h3{margin:13px 0 9px}.fixed-note{font-size:14px;color:#5c7283}.legend{font-size:13px;color:#577086;margin:7px 0}.queries-label{font-size:13px;font-weight:600}#visible-count{font-size:14px}@media(max-width:700px){.metric-pair{grid-template-columns:1fr}}
"""
JS="""
const cards=[...document.querySelectorAll('.case')],query=document.getElementById('search'),result=document.getElementById('result');
function filter(){let count=0;for(const card of cards){const yes=(!query.value.trim()||card.dataset.search.includes(query.value.trim().toLowerCase()))&&(!result.value||card.dataset[result.value]==='true');card.hidden=!yes;count+=Number(yes);}document.getElementById('visible-count').textContent=`显示 ${count} / ${cards.length} 张`;}
query.addEventListener('input',filter);result.addEventListener('change',filter);document.getElementById('reset').addEventListener('click',()=>{query.value='';result.value='';filter();});
const dialog=document.getElementById('zoom-dialog');document.querySelectorAll('.zoom').forEach(button=>button.addEventListener('click',()=>{const card=button.closest('.case');document.getElementById('zoom-title').textContent=card.querySelector('h2').textContent;document.getElementById('zoom-content').replaceChildren(card.querySelector('.image-wrap').cloneNode(true));dialog.showModal();}));document.getElementById('close-zoom').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});filter();
"""


def render_card(record,image_bytes,size):
    sid=record['sample_id'];sample=record['sample'];old,new=record['old'],record['new'];search=record['search']
    improved=new['names']['correct']>old['names']['correct']
    wrong=new['names']['correct']<new['names']['target_count']
    terms=' '.join([sid]+[p['character'] for p in sample['persons']]).lower()
    parts=[f'<article class="case" id="{sid}" data-search="{esc(terms)}" data-improved="{str(improved).lower()}" data-wrong="{str(wrong).lower()}" data-cap="{str(search["cap_exceeded"]).lower()}"><div class="case-head"><h2>{sid} · 功夫女足</h2><span class="badge'+('' if search['verified'] else ' warn')+'">'+('已返回搜索来源' if search['verified'] else '未确认有效检索来源')+'</span></div><p>旧组姓名 '+str(old['names']['correct'])+' / '+str(old['names']['target_count'])+' → 新组姓名 '+str(new['names']['correct'])+' / '+str(new['names']['target_count'])+'</p><button class="zoom" type="button">放大同一输入图</button><div class="image-wrap" style="margin-top:12px"><img class="input-image" loading="lazy" width="'+str(size[0])+'" height="'+str(size[1])+'" alt="'+sid+' · 新旧组相同的C组输入" src="data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode('ascii')+'">'+overlay(sample,new['strict'],new['raw_predictions'],size)+'</div><p class="legend">同一无框输入；绿色实线为人工真值，橙色虚线为新组合法预测，灰色为原规则忽略的背景预测。叠加层未提供给模型。</p>']
    rows=[]
    by_old={p['person_id']:p for p in old['names']['person_details']};by_new={p['person_id']:p for p in new['names']['person_details']}
    for person in sample['persons']:
        values=[]
        for detail in (by_old[person['person_id']],by_new[person['person_id']]):
            values.append('<span class="good">命中：'+esc(detail['predicted'])+'</span>' if detail['outcome']=='correct' else '<span class="bad">未命中</span>')
        rows.append([str(person['person_id']),esc(person['character']),*values])
    parts.append(table(['真值人物编号','当前已确认角色','旧组仅姓名','新组仅姓名'],rows))
    for title,data in [('旧组',old),('新组',new)]:
        names='、'.join('null' if p.get('character') is None else str(p['character']) for p in data['raw_predictions']) or '（无姓名条目）'
        parts.append('<p class="raw-names">'+title+'实际姓名列表：'+esc(names)+'</p>')
    parts.append('<p class="fixed-note">姓名按一对一匹配；未命中不按返回顺序强行分配给某个人物。上方姓名判定不要求位置对应。</p>')
    if search['cap_exceeded']:
        parts.append(f'<div class="notice"><strong>工具次数超过请求约束：</strong>本次请求max_uses={search["requested_max"]}，服务端实际报告{search["reported_calls"]}次搜索。不能将配置上限理解为已强制执行的消费上限。</div>')
    parts.append('<h3>实际检索记录</h3><p>服务端报告搜索次数：'+str(search['reported_calls'])+'；返回有效来源'+str(len(search['sources']))+'条。来源域名：'+esc('、'.join(search['domains']) or '无')+'</p><div class="queries-label">模型实际提交的查询</div><div class="search-queries">'+esc('\n'.join(search['queries']) or '无可核对查询')+'</div>')
    if search['sources']:
        parts.append('<details class="sources"><summary>展开搜索来源（'+str(len(search['sources']))+'条）</summary><ul class="source-list">'+''.join('<li><a href="'+esc(s['url'])+'" target="_blank" rel="noopener noreferrer">'+esc(s['title'])+'</a> <small>'+esc(s['domain'])+'</small></li>' for s in search['sources'])+'</ul></details>')
    if search['errors']:
        parts.append('<details><summary>搜索工具错误内容</summary><pre>'+esc(jdump(search['errors']))+'</pre></details>')
    parts.append('<details class="raw-responses"><summary>新旧原始回答与定位判定</summary><div class="metric-pair">')
    for title,data in [('旧组',old),('新组',new)]:
        parts.append('<div class="metric-panel"><h3>'+title+'</h3><p>格式'+('合规' if data['strict']['parse_ok'] else '不合规 / 未完成')+'；按位置匹配且姓名正确 '+str(data['strict']['identity_correct'])+'人</p><pre>'+esc(data['raw_text'])+'</pre></div>')
    parts.append('</div></details></article>')
    return ''.join(parts)


def export_pilot(run_id=DEFAULT_RUN,output=DEFAULT_OUTPUT):
    context=build_identity_review(ROOT)
    with sqlite3.connect((ROOT/'results/multi/runs.db').as_uri()+'?mode=ro',uri=True) as conn:
        conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
        conn.row_factory=sqlite3.Row
        run=dict(conn.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone() or {})
        if not run:raise ValueError('Run does not exist')
        metadata=json.loads(run['fairness_meta'])
        if json.loads(run['models'])!=[MODEL] or json.loads(run['experiments'])!=['C'] or metadata.get('is_mock'):
            raise ValueError('Expected real Qwen Max C-only search pilot')
        stored={r['sample_id']:json.loads(r['raw_json']) for r in conn.execute('SELECT sample_id,raw_json FROM records WHERE run_id=?',(run_id,))}
        ledger=[dict(r) for r in conn.execute('SELECT * FROM multi_budget_calls WHERE run_id=?',(run_id,))]
    frozen={s.sample_id:s.to_dict() for s in load_dataset(Path(run['dataset_path']))}
    if set(frozen)!=set(SAMPLE_IDS) or set(stored)-set(SAMPLE_IDS):raise ValueError('Pilot sample scope differs from approved 15')
    gallery={r['sample_id']:r for r in json.loads((ROOT/'results/multi/input_gallery/manifest.json').read_text())['rows']}
    matcher=CharacterMatcher({w.slug:load_roster(w) for w in load_works()})
    old_names={(r['experiment'],r['sample_id']):r for r in context['name_rows']}
    old_strict={(r['experiment'],r['sample_id']):r for r in context['rows']}
    records=[];cards=[];completed=0
    for sid in SAMPLE_IDS:
        sample=frozen[sid];reference=context['samples'][sid]
        for person in sample['persons']:
            previous=next(p for p in reference['persons'] if p['person_id']==person['person_id'])
            for key in ('character','character_aliases','bbox','annotation_id','identity_status','review_status'):
                if person[key]!=previous[key]:raise ValueError(f'New frozen truth differs from corrected baseline: {sid}/{key}')
        old_raw=context['raw_records'][('C',sid)];raw=stored.get(sid)
        image_bytes=(ROOT/'results/multi/input_gallery'/gallery[sid]['experiments']['C']['image']).read_bytes()
        image_sha=hashlib.sha256(image_bytes).hexdigest()
        if image_sha!=old_raw['request']['image_meta']['encoded_sha256']:raise ValueError('Baseline input image mismatch')
        if raw:
            completed+=1
            request=raw['request']
            if request.get('api_mode')!='dashscope_anthropic' or not request.get('search_requested'):
                raise ValueError('Expected native search-enabled pilot requests')
            if image_sha!=request['image_meta']['encoded_sha256'] or sample['image_sha256']!=request['image_meta']['source_sha256']:
                raise ValueError('New request image differs from baseline')
            for key in ('system','user','temperature','max_tokens'):
                if request.get(key)!=old_raw['request'].get(key):raise ValueError(f'Business input changed: {sid}/{key}')
        text=raw['response']['raw_text'] if raw else ''
        parsed=parse_response(text,'C')
        api_state='ok' if raw and raw['response']['ok'] else 'api_failure' if raw else 'not_run'
        strict=score_sample(sample,'C',parsed,matcher,api_state=api_state)
        named=score_name_only(sample,'C',text,matcher)
        evidence=search_evidence(raw)
        record={'sample_id':sid,'sample':sample,'input_sha256':image_sha,'new_input_verified':bool(raw),
                'old':{'names':old_names[('C',sid)],'strict':old_strict[('C',sid)],'raw_text':old_raw['response']['raw_text'],
                       'raw_predictions':raw_persons(old_raw['response']['raw_text'])},
                'new':{'names':named,'strict':strict,'raw_text':text,'raw_predictions':[{'character':p.character,'bbox':p.bbox} for p in parsed.persons] if parsed.parse_ok else []},
                'search':evidence,'new_request':raw['request'] if raw else None}
        # Raw name lists survive an invalid bbox; parsed persons alone would hide them.
        record['new']['raw_predictions']=raw_persons(text)
        records.append(record)
        cards.append(render_card(record,image_bytes,old_raw['request']['image_meta']['encoded_size']))
    old,new=metrics(records,'old'),metrics(records,'new')
    evidence_summary={'verified_images':sum(r['search']['verified'] for r in records),
                      'reported_calls':sum(r['search']['reported_calls'] for r in records),
                      'cap_exceeded_images':[r['sample_id'] for r in records if r['search']['cap_exceeded']]}
    rate=metadata['pricing']['models'][MODEL]
    costs={'known_token_estimate_cny':sum(r['known_cny'] for r in ledger),'uncertain_cny':sum(r['uncertain_cny'] for r in ledger),
           'committed_cny':sum(r['committed_cny'] for r in ledger),'http_attempts':len(ledger),
           'configured_request_fee_upper_cny':rate.get('request_fee_upper_cny',0),
           'observed_search_fee_reference_upper_cny':evidence_summary['reported_calls']*rate.get('request_fee_upper_cny',0)}
    complete=completed==15 and run['status'] in ('done','completed')
    summary={'run_id':run_id,'baseline_run_id':context['summary']['run_id'],'model_id':MODEL,'generated_at':datetime.now(timezone.utc).isoformat(),
             'status':run['status'],'complete':complete,'completed_images':completed,'planned_images':15,'target_people':19,
             'comparison_boundary':'Tool-enhanced protocol comparison, not a strictly isolated search-toggle intervention: native Anthropic-compatible protocol adds mandatory system client marker.',
             'old':old,'new':new,'search':evidence_summary,'costs':costs,'records':records}
    status='已完成：15 / 15张' if complete else f'进行中：{completed} / 15张；非最终结果'
    cap_notice=''
    if evidence_summary['cap_exceeded_images']:
        exceeded='、'.join(r['sample_id']+'（'+str(r['search']['reported_calls'])+'次）' for r in records if r['search']['cap_exceeded'])
        cap_notice='<div class="notice"><strong>搜索次数约束未被完全遵守：</strong>'+esc(exceeded)+' 的实际搜索次数超过请求的max_uses=1。请按逐图记录与账本核对费用，不能假设每张只搜索一次。</div>'
    content='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Max · 功夫女足15张联网对照</title><style>'+CSS+STYLE+'</style></head><body><main><header class="intro"><h1>Qwen Max · 功夫女足15张联网对照</h1><span class="badge'+('' if complete else ' warn')+'">'+status+'</span><p>同15张图片、同19个人物真值，对照原无搜索C组与新联网工具方案；钰珑等标注已应用用户更正。全部图像、对照、原始回答与搜索来源清单均内嵌，可独立离线打开。</p><p class="fixed-note">这15张是清晰角色的定向诊断样本；结果不能代表《功夫女足》全部40张或四部作品160张的最终表现。</p><div class="notice">新组仍是Qwen Max，但改用百炼Anthropic兼容工具接口，并增加厂商要求的系统标记。业务提示词、图片、温度与输出预算核对一致；这比较的是完整联网工具方案，不能严格认定全部变化只来自一个搜索开关。</div>'+cap_notice+'</header><section class="panel comparison"><h2>同批样本对照</h2>'+summary_table(old,new,evidence_summary)+'<p class="fixed-note">仅姓名按角色名/已确认别名一对一匹配，忽略坐标；位置与姓名均正确采用原IoU≥0.5和背景忽略规则，以全部19个目标为分母。定位统计沿用原协议：格式失败也计入漏检和无效预测，因此F1仍受格式问题影响，不能与仅姓名命中率混作同一个准确率。未完成目标保留在计划分母中，进行中数字仅作过程记录。</p></section><section class="panel cost"><h2>本次调用与费用记录</h2><p>已保存'+str(completed)+'次图片回答，账本实际HTTP尝试'+str(costs['http_attempts'])+'次；实际有来源的检索记录'+str(evidence_summary['verified_images'])+'张，服务端报告搜索动作'+str(evidence_summary['reported_calls'])+'次。</p><p>本run已知token费用上界估算：<strong>¥'+f'{costs["known_token_estimate_cny"]:.6f}'+'</strong>；保留未知费用：<strong>¥'+f'{costs["uncertain_cny"]:.6f}'+'</strong>；本run当前保护占用：¥'+f'{costs["committed_cny"]:.6f}'+'。这些是本地估算与预留，不是供应商实际账单。</p><p class="fixed-note">按已观察的搜索动作与配置单价计算，工具费参考上界为¥'+f'{costs["observed_search_fee_reference_upper_cny"]:.6f}'+'；是否实际扣费需账单确认。此处只列本次run，不把历史累计费用当作新增支出。</p></section><div class="toolbar"><label>搜索编号或角色<input id="search" type="search" placeholder="例如 0007 或 钰珑"></label><label>筛选<select id="result"><option value="">全部15张</option><option value="improved">姓名命中数增加</option><option value="wrong">新组姓名未全命中</option><option value="cap">实际搜索超过请求约束</option></select></label><button id="reset" type="button">重置筛选</button><strong id="visible-count"></strong></div>'+''.join(cards)+'<footer><details><summary>调用条件与来源版本</summary><pre>'+esc(jdump({'run_id':run_id,'baseline_run_id':summary['baseline_run_id'],'source_dataset_hash':context['source_dataset_hash'],'corrected_review_dataset_hash':context['review_dataset_hash'],'new_frozen_dataset_hash':run['dataset_hash'],'protocol':'dashscope_anthropic','mandatory_system_marker':'x-anthropic-billing-header: cc_entrypoint=cli;','search_proof':'Only actual web_search_result entries with HTTP(S) URL count as retrieved sources; intent flags/counters alone do not prove successful retrieval.'}))+'</pre></details></footer></main><dialog id="zoom-dialog"><div class="dialog-head"><strong id="zoom-title"></strong><button id="close-zoom" type="button">关闭（Esc）</button></div><div id="zoom-content" class="dialog-content"></div></dialog><script>'+JS+'</script></body></html>'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'index.html';target.write_text(content,encoding='utf-8')
    summary['html_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (output/'summary.json').write_text(jdump(summary)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();result=export_pilot(args.run_id,args.output)
    print(jdump({k:result[k] for k in ('run_id','complete','completed_images','old','new','search','costs')}))
