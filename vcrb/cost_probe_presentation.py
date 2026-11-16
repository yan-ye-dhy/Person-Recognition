"""A compact, offline briefing from already reconciled costs and scored results."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, localcontext
from html import escape


LABELS = {
    'deepseek-flash': 'DeepSeek Flash', 'kimi-k3.1-pro': 'Kimi K3.1 Pro',
    'qwen3.8-max': 'Qwen Max', 'qwen3.7-plus': 'Qwen Plus',
    'qwen3.8-flash': 'Qwen Flash', 'glm-5.3-flash': 'GLM 5.3 Flash', 'glm-4.6v': 'GLM 4.6V',
}


def fixed(value, places=2):
    with localcontext() as context:
        context.prec = 100
        return format(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), f'.{places}f')


def unit_money(value):
    number = Decimal(value)
    return '<0.0001' if 0 < number < Decimal('0.00005') else fixed(number, 4)


def _divide(amount, count):
    with localcontext() as context:
        context.prec = 100
        return format(Decimal(amount) / count, 'f')


def _total(rows):
    with localcontext() as context:
        context.prec = 100
        return format(sum((Decimal(row['amount']) for row in rows), Decimal(0)), 'f')


def _cost_overview(summary, costs, cohort):
    daily = summary['historical_daily_bills']
    compared = {row['model_id'] for row in costs}
    overview = []
    for model in summary['models']:
        mid = model['model_id']
        current = [charge for charge in model['charges'] if charge['currency'] == 'CNY']
        if len(current) > 1:
            raise ValueError('Current model charges must have one total per currency')
        charge = current[0] if current else None
        date = (model.get('started_at') or model.get('finished_at') or summary['generated_at'])[:10]
        end_date = (model.get('finished_at') or date)[:10]
        if end_date != date:
            date += '–' + end_date
        tasks = model['submitted_count']
        source, posted, unit_cost, unit_kind = 'current', None, None, 'pending'
        note = '费用明细待核'
        if charge and model['billing_complete'] and charge['official_amount'] is not None and tasks:
            posted, unit_cost, unit_kind = charge['official_amount'], charge['average_per_submitted_image'], 'verified'
            note = '本批 · 推理+搜索已核'
        elif charge and charge.get('posted_amount') is not None and tasks:
            posted = charge['posted_amount']
            unit_cost = charge.get('posted_average_per_submitted_image')
            if unit_cost is None:
                unit_cost = _divide(posted, tasks)
            kinds = set(charge.get('kinds', charge.get('posted_by_kind', {})))
            if kinds == {'inference'}:
                unit_kind, note = 'inference_only', '本批 · 仅推理；搜索待核'
            elif kinds == {'search'}:
                unit_kind, note = 'search_only', '本批 · 仅搜索；推理待核'
            else:
                unit_kind, note = 'partial', '本批 · 部分费用；完整账单待核'
        else:
            historical = [r for r in summary['historical_batches'] if r['model_id'] == mid and r['currency'] == 'CNY']
            verified = max(historical, key=lambda r: (r['date'], r['image_tasks']), default=None)
            if verified:
                date, tasks, posted = verified['date'], verified['image_tasks'], verified['amount']
                unit_cost, unit_kind, source = _divide(posted, tasks), 'verified', 'historical_batch'
                note = '历史 · 推理+搜索已核'
            else:
                # Choose this model's own latest evidence, never a global day
                # selected by a different provider's newer export.
                model_daily = [r for r in daily if r['model_id'] == mid and r['currency'] == 'CNY' and r['amount'] is not None]
                references = [r for r in summary.get('daily_reference_tasks', []) if r['model_id'] == mid]
                reference = max(references, key=lambda r: r['date'], default=None)
                date = reference['date'] if reference else max((r['date'] for r in model_daily), default=date)
                rows = [r for r in model_daily if r['date'] == date]
                kinds = {r['kind'] for r in rows}
                if 'combined' in kinds and len(kinds) > 1:
                    raise ValueError('Daily totals and their components cannot be added together')
                if rows:
                    posted, source, tasks = _total(rows), 'historical_daily', None
                if reference:
                    posted, tasks = reference['amount'], reference['image_tasks']
                    unit_cost = _divide(posted, tasks)
                    unit_kind = 'inference_only' if reference['fee_scope'] == 'inference_only' else 'daily_reference'
                    source = 'daily_reference'
                    note = '仅推理；搜索待核' if unit_kind == 'inference_only' else '含搜索 · 日账参考'
                    if reference['contains_debugging']:
                        note = '旧接口调试均摊'
                    elif reference['cross_midnight']:
                        note += ' · 含跨日'
                    if any(r['note_code'] == 'resource_pack' for r in rows):
                        note += ' · 资源包抵扣'
                elif rows:
                    if any(r['note_code'] == 'resource_pack' for r in rows):
                        note = '推理资源包抵扣；搜索未分摊'
                    elif mid.startswith('glm-') and kinds == {'inference'}:
                        note = '仅推理；共用搜索未分摊'
                    elif kinds == {'inference'}:
                        note = '仅推理；搜索费用待核'
                    else:
                        note = '已出推理 + 搜索；单图待核'
        period_note = date + ' · ' + (str(tasks) + '任务 · ' if tasks is not None else '任务数待核 · ')
        in_comparison = bool(cohort and mid in compared and tasks == cohort['image_tasks']
                             and source == ('current' if cohort['kind'] == 'current' else 'historical_batch')
                             and (cohort['kind'] != 'historical' or date == cohort['date']))
        overview.append({'model_id': mid, 'label': LABELS.get(mid, mid), 'posted_amount': posted,
            'unit_cost': unit_cost, 'unit_kind': unit_kind, 'note': period_note + note,
            'date': date, 'image_tasks': tasks, 'source': source,
            'in_comparison': in_comparison})
    shared_rows = [r for r in daily if r['model_id'] == 'search-pro' and r['currency'] == 'CNY' and r['amount'] is not None]
    shared_date = max((r['date'] for r in shared_rows), default=None)
    shared = [r for r in shared_rows if r['date'] == shared_date]
    dates = {r['date'] for r in overview}
    return overview, next(iter(dates)) if len(dates) == 1 else None, _total(shared) if shared else None, shared_date


def build_presentation(summary, recognition=None):
    """Never rank costs from different batches or turn missing costs into zero."""
    groups = defaultdict(list)
    for row in summary['historical_batches']:
        groups[('historical', row['date'], row['image_tasks'])].append({
            'model_id': row['model_id'], 'amount': row['amount'], 'image_tasks': row['image_tasks'],
            'unit_cost': _divide(row['amount'], row['image_tasks']), 'currency': row['currency'],
        })
    for row in summary['models']:
        for charge in row['charges']:
            if row['billing_complete'] and charge['currency'] == 'CNY' and row['submitted_count']:
                groups[('current', summary['probe_id'], row['submitted_count'])].append({
                    'model_id': row['model_id'], 'amount': charge['official_amount'],
                    'image_tasks': row['submitted_count'], 'unit_cost': charge['average_per_submitted_image'],
                    'currency': 'CNY',
                })
    # The current identical image batch takes priority once at least two models
    # have complete fees. Partial inference evidence never enters this ranking.
    current = [k for k in groups if k[0] == 'current' and k[2] == len(summary['planned_sample_ids'])
               and len(groups[k]) >= 2]
    key = (max(current, key=lambda k: (len(groups[k]), k[1])) if current else
           max(groups, key=lambda k: (len(groups[k]) >= 2, k[2], len(groups[k]), k[1])) if groups else None)
    costs = sorted(groups[key], key=lambda row: Decimal(row['unit_cost'])) if key else []
    for row in costs:
        row['label'] = LABELS.get(row['model_id'], row['model_id'])
        with localcontext() as context:
            context.prec = 100
            row['per_1000'] = format(Decimal(row['unit_cost']) * 1000, 'f')
            row['per_10000'] = format(Decimal(row['unit_cost']) * 10000, 'f')
    quality = []
    if recognition is not None:
        if recognition.get('dataset_hash') != summary['dataset_hash']:
            raise ValueError('Recognition and cost presentation use different reviewed dataset versions')
        scopes = set()
        for mid in [row['model_id'] for row in summary['models']]:
            score = recognition.get('all_works', {}).get(mid, {}).get('new')
            if not isinstance(score, dict):
                continue
            fields = ['planned_images', 'completed_images', 'planned_targets', 'name_correct']
            if any(type(score.get(field)) is not int or score[field] < 0 for field in fields):
                raise ValueError('Invalid recognition sample counts')
            images, completed, targets, correct = (score[field] for field in fields)
            if not images or not targets or completed != images or correct > targets:
                raise ValueError('Presentation requires a completed recognition cohort')
            scopes.add((images, targets))
            quality.append({'model_id': mid, 'label': LABELS.get(mid, mid), 'images': images,
                'targets': targets, 'correct': correct, 'accuracy_percent': _divide(correct * 100, targets)})
        if len(scopes) > 1:
            raise ValueError('Recognition models must share the same sample and target denominator')
    quality.sort(key=lambda row: Decimal(row['accuracy_percent']), reverse=True)
    cost_map = {row['model_id']: row for row in costs}
    preferred = quality[0] if quality and costs and quality[0]['model_id'] == costs[0]['model_id'] else None
    ratio = None
    if 'deepseek-flash' in cost_map and 'kimi-k3.1-pro' in cost_map:
        ds = Decimal(cost_map['deepseek-flash']['unit_cost'])
        if ds > 0:
            ratio = _divide(Decimal(cost_map['kimi-k3.1-pro']['unit_cost']), ds)
    cohort = {'kind': key[0], 'date': key[1] if key[0] == 'historical' else None, 'image_tasks': key[2]} if key else None
    overview, overview_date, shared_search, shared_search_date = _cost_overview(summary, costs, cohort)
    if quality:
        ranks = {row['model_id']: index for index, row in enumerate(quality)}
        overview.sort(key=lambda row: ranks.get(row['model_id'], len(ranks)))
    return {'costs': costs, 'cost_cohort': cohort, 'recognition': quality,
            'cost_overview': overview, 'cost_overview_date': overview_date,
            'shared_glm_search_cost': shared_search, 'shared_glm_search_date': shared_search_date,
            'preferred': preferred, 'kimi_to_deepseek_cost_ratio': ratio,
            'pending_cost_models': [row['model_id'] for row in summary['models'] if row['model_id'] not in cost_map],
            'recognition_scope': {'images': quality[0]['images'], 'targets': quality[0]['targets']} if quality else None}


CSS = r'''
:root{--ink:#142d3a;--muted:#617580;--green:#078273;--purple:#6a63c7;--line:#dce7e9;--surface:#fff;--back:#f3f7f7}
*{box-sizing:border-box}body{margin:0;background:var(--back);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:38px 30px 28px}h1,h2,h3,p{margin:0}h1{font-size:38px;line-height:1.25;letter-spacing:-1px;margin:12px 0}h2{font-size:20px;line-height:1.4}h3{font-size:17px;line-height:1.5}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:700;color:var(--green)}.hero{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:25px}.intro{color:var(--muted);max-width:760px}.stamp{white-space:nowrap;color:var(--muted);font-size:12px;text-align:right;padding-top:6px}.stamp strong{display:block;color:var(--ink);font-size:14px;margin-bottom:5px}.recommend{display:inline-flex;border-radius:30px;padding:5px 12px;color:var(--green);background:#e7f5f0;font-size:12px;font-weight:600;margin-top:14px}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:22px}.kpi{padding:21px 23px;border:1px solid var(--line);border-radius:14px;background:white}.kpi:first-child{background:var(--green);border-color:var(--green);color:white}.kpi-label{font-size:13px;color:var(--muted);font-weight:600}.kpi:first-child .kpi-label,.kpi:first-child .kpi-note{color:#d7f1eb}.kpi-value{font-size:39px;line-height:1.25;font-weight:750;letter-spacing:-1px;margin:7px 0 6px;font-variant-numeric:tabular-nums}.kpi-value small{font-size:14px;font-weight:500;letter-spacing:0;margin-left:6px}.kpi-note{font-size:12px;color:var(--muted)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:stretch}.panel{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:24px}.sub{font-size:12px;color:var(--muted);margin-top:6px}.cost-bars{margin-top:28px}.cost-item{margin:0 0 25px}.row-top{display:flex;justify-content:space-between;gap:12px;font-size:14px;margin-bottom:9px;align-items:baseline}.row-top strong{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:16px}.track{height:14px;border-radius:4px;background:#edf2f3;overflow:hidden}.fill{height:100%;background:var(--green);border-radius:4px;min-width:0}.fill.purple{background:var(--purple)}.track-small{height:8px}.scale{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:-11px}.cost-callout{padding:15px 16px;background:#f1f8f5;border-radius:9px;margin-top:23px;font-size:14px}.cost-callout strong{color:var(--green)}.scenario{margin-top:18px;padding-top:17px;border-top:1px solid var(--line)}.scenario-row{display:flex;justify-content:space-between;font-size:13px;margin-top:6px}.scenario-row strong{font-variant-numeric:tabular-nums}.pending{font-size:12px;color:#8b6b36;background:#fbf6ec;padding:11px 13px;border-radius:8px;margin-top:17px}.accuracy-bars{margin-top:19px}.accuracy-item{margin-bottom:15px}.accuracy-item:last-child{margin-bottom:0}.accuracy-item .row-top{font-size:13px;margin-bottom:6px}.accuracy-item .row-top strong{font-size:14px}.accuracy-item .fill{background:#a9bbc6}.accuracy-item.highlight .fill{background:var(--green)}.accuracy-item.runner .fill{background:#526fa1}.accuracy-item.kimi .fill{background:var(--purple)}.findings{margin-top:26px}.findings-heading{display:flex;justify-content:space-between;align-items:baseline;gap:15px;margin-bottom:12px}.findings-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.finding{background:white;border:1px solid var(--line);border-radius:13px;padding:20px}.finding p{font-size:13px;color:var(--muted);margin-top:8px}.finding-index{font-size:11px;color:var(--green);font-weight:700;letter-spacing:1px;margin-bottom:8px}.foot{font-size:11px;color:var(--muted);border-top:1px solid var(--line);margin-top:23px;padding-top:15px}.foot p+p{margin-top:5px}.empty{color:var(--muted);padding:25px 0}.method{margin-top:10px}summary{cursor:pointer;color:var(--muted)}.method p{margin-top:5px}.no-data{font-size:15px}.kpi-value.no-data{letter-spacing:0}a{color:var(--green)}
.cost-overview{margin-top:15px}.overview-head,.overview-row{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(0,.85fr) minmax(0,.85fr);gap:12px;align-items:center}.overview-head{font-size:11px;color:var(--muted);padding-bottom:8px;border-bottom:1px solid var(--line)}.overview-head span:not(:first-child){text-align:right}.overview-row{padding:10px 0;border-bottom:1px solid #eef2f3;font-size:13px}.overview-row:last-child{border-bottom:0}.overview-row strong{font-weight:650}.overview-row small{display:block;font-size:10px;color:var(--muted);line-height:1.45;margin-top:3px}.posted-value,.known-unit,.pending-unit,.reference-unit{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.known-unit{color:var(--green)}.known-unit .track{height:5px;margin-top:5px}.pending-unit{color:#8b6b36}.reference-unit{color:#526fa1}.overview-row .reference-tag{font-size:9px;margin-top:2px;color:var(--muted)}.pending-track{height:5px;margin-top:5px;border-bottom:1px dashed #d2d9db}.compare-note{font-size:11px;color:var(--muted);margin-top:10px}.compare-note strong{color:var(--green)}
@media(max-width:760px){main{padding:24px 16px}h1{font-size:29px}.hero{gap:14px}.stamp{font-size:10px}.stamp strong{font-size:12px}.kpis{gap:10px}.kpi{padding:16px 13px}.kpi-value{font-size:28px}.kpi-value small{display:block;font-size:11px;margin-left:0}.charts{grid-template-columns:1fr}.findings-grid{grid-template-columns:1fr}.panel{padding:20px}.findings-heading{display:block}.kpi-note{font-size:10px}.overview-head,.overview-row{gap:7px;grid-template-columns:minmax(0,1.55fr) minmax(0,.9fr) minmax(0,.9fr)}.overview-row{font-size:11px}.overview-row small{font-size:9px}}
@media(max-width:420px){.hero{display:block}.stamp{text-align:left;padding-top:12px}.stamp strong{display:inline;margin-right:10px}.kpis{grid-template-columns:1fr}.kpi-value small{display:inline;margin-left:6px}.kpi-value{font-size:34px}}
@page{size:A4 landscape;margin:10mm}@media print{body{background:white;font-size:10px;line-height:1.35}main{padding:0;max-width:none;width:100%}h1{font-size:24px;line-height:1.1;margin:6px 0}h2{font-size:16px}.hero{display:flex;margin-bottom:10px}.intro{font-size:10px}.stamp,.stamp strong{font-size:9px}.eyebrow{font-size:9px}.recommend{margin-top:5px;padding:3px 9px;font-size:9px}.kpis{grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}.kpi{padding:10px 13px}.kpi-label{font-size:10px}.kpi-value{font-size:25px;line-height:1.1;margin:4px 0}.kpi-value small{display:inline;font-size:10px;margin-left:5px}.kpi-note{font-size:9px}.charts{grid-template-columns:1fr 1fr;gap:12px}.panel{padding:13px}.sub{font-size:9px;margin-top:3px}.cost-bars{margin-top:13px}.cost-item{margin-bottom:15px}.row-top{font-size:10px;margin-bottom:5px}.row-top strong{font-size:13px}.track{height:10px}.scale{font-size:8px;margin-top:-8px}.cost-callout{padding:8px 10px;margin-top:12px;font-size:10px}.scenario{margin-top:10px;padding-top:8px}.scenario-row{font-size:10px;margin-top:3px}.pending{padding:7px 10px;margin-top:10px;font-size:9px}.accuracy-bars{margin-top:12px}.accuracy-item{margin-bottom:8px}.accuracy-item .row-top{font-size:10px;margin-bottom:3px}.accuracy-item .row-top strong{font-size:11px}.track-small{height:5px}.findings{margin-top:12px}.findings-heading{margin-bottom:7px}.findings-grid{grid-template-columns:repeat(3,1fr);gap:12px}.finding{padding:10px}.finding-index{font-size:8px;margin-bottom:4px}.finding h3{font-size:12px}.finding p{font-size:9px;margin-top:4px}.foot{margin-top:10px;padding-top:8px;font-size:8px}.kpi,.panel,.finding{break-inside:avoid}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}.method{display:none}}
@media print{.cost-overview{margin-top:6px}.overview-head{font-size:8px;padding-bottom:4px}.overview-row{font-size:10px;padding:3px 0}.overview-row small{font-size:7.5px;line-height:1.15;margin-top:1px}.overview-row .reference-tag{font-size:7px;line-height:1.1;margin-top:1px}.overview-head,.overview-row{gap:8px}.compare-note{font-size:8px;margin-top:5px}.overview-row .track,.pending-track{height:3px;margin-top:3px}.panel .pending{font-size:8px;line-height:1.25;margin-top:6px;padding:5px 8px}.finding{padding:8px}.finding p{font-size:8.5px}.findings{margin-top:9px}}
'''


def render_presentation(summary, presentation):
    e = lambda value: escape(str(value), quote=True)
    costs, quality = presentation['costs'], presentation['recognition']
    leader = costs[0] if costs else None
    preferred = presentation['preferred']
    quality_best = quality[0] if quality else None
    scope = presentation['recognition_scope']
    head_name = leader['label'] if leader else '单图成本'
    kpi_cost = unit_money(leader['unit_cost']) if leader else '待核'
    kpi_accuracy = fixed(quality_best['accuracy_percent']) if quality_best else '待补'
    kpi_budget = fixed(leader['per_1000']) if leader else '待核'
    cohort = presentation['cost_cohort']
    cohort_label = (('本批' if cohort['kind'] == 'current' else cohort['date'] + '历史')
                    + str(cohort['image_tasks']) + '任务') if cohort else '费用待核'
    generated = summary['generated_at'][:10]
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>联网角色识别｜成本与效果简报</title><style>', CSS, '</style><main>']
    parts.append('<header class="hero"><div><p class="eyebrow">角色识别项目 · 联网测试</p><h1>联网识别的成本与效果</h1><p class="intro">聚焦单图费用、识别表现和选型建议，供项目汇报使用。</p>')
    if preferred:
        parts.append('<span class="recommend">优先试点：' + e(preferred['label']) + '</span>')
    parts.append('</div><div class="stamp"><strong>评测简报</strong>' + e(generated) + '<br>官方账单核对 · 已复核标注</div></header>')
    parts.append('<section class="kpis" aria-label="核心指标"><article class="kpi"><p class="kpi-label">' + e(head_name) + ' · 已核均价</p><p class="kpi-value">' + e(kpi_cost) + '<small>元 / 图</small></p><p class="kpi-note">' + e(cohort_label) + ' · 推理+搜索已核</p></article>')
    parts.append('<article class="kpi"><p class="kpi-label">' + e(quality_best['label'] if quality_best else '识别表现') + ' · 姓名命中率</p><p class="kpi-value">' + e(kpi_accuracy) + ('<small>%</small>' if quality_best else '') + '</p><p class="kpi-note">' + (e(f"{scope['images']}张正式样本 · {quality_best['correct']}/{scope['targets']}个必需目标") if scope else '尚未接入正式样本结果') + '</p></article>')
    parts.append('<article class="kpi"><p class="kpi-label">' + e(head_name) + ' · 千张成本参考</p><p class="kpi-value">' + e(kpi_budget) + '<small>元 / 千张</small></p><p class="kpi-note">按所示批次均价外推，非实际账单</p></article></section>')
    parts.append('<div class="charts"><section class="panel"><h2>七模型费用总览</h2>')
    parts.append('<p class="sub">优先本批费用，各行标明日期与任务数 · 单位：元；历史费用另列口径</p>')
    parts.append('<div class="cost-overview" role="table" aria-label="七个模型的已出账金额与单图均摊"><div class="overview-head" role="row"><span>模型 / 费用范围</span><span>已出账</span><span>单图均摊</span></div>')
    maximum = max((Decimal(row['unit_cost']) for row in presentation['cost_overview']
                   if row['unit_kind'] == 'verified' and row['unit_cost'] is not None), default=Decimal(0))
    for row in presentation['cost_overview']:
        posted = unit_money(row['posted_amount']) if row['posted_amount'] is not None else '待核'
        if row['posted_amount'] is not None and Decimal(row['posted_amount']) == 0 and row['unit_kind'] != 'verified':
            posted = '推理 ' + posted
        per_image = unit_money(row['unit_cost']) if row['unit_cost'] is not None else '待核'
        if row['unit_kind'] in ('daily_reference', 'inference_only') and row['source'] != 'current' and Decimal(row['unit_cost']) != 0:
            per_image = '≈ ' + per_image
        known_class = 'known-unit' if row['unit_kind'] == 'verified' else 'reference-unit' if row['unit_cost'] is not None else 'pending-unit'
        parts.append('<div class="overview-row" role="row" data-model="' + e(row['model_id']) + '"><div role="cell"><strong>' + e(row['label']) + '</strong><small>' + e(row['note']) + '</small></div><div class="posted-value" role="cell">' + e(posted) + '</div><div class="' + known_class + '" role="cell"><strong>' + e(per_image) + '</strong>')
        if row['unit_kind'] == 'verified':
            width = _divide(Decimal(row['unit_cost']) * 100, maximum) if maximum else '0'
            color = ' purple' if row['model_id'] == 'kimi-k3.1-pro' else ''
            parts.append('<div class="track track-small"><div class="fill' + color + '" style="width:' + fixed(width) + '%"></div></div>')
        elif row['unit_cost'] is not None:
            tag = {'inference_only': '仅推理', 'search_only': '仅搜索', 'partial': '部分费用'}.get(row['unit_kind'], '日账参考')
            parts.append('<small class="reference-tag">' + tag + '</small>')
        else:
            parts.append('<div class="pending-track" aria-hidden="true"></div>')
        parts.append('</div></div>')
    parts.append('</div>')
    if len(costs) >= 2 and Decimal(costs[0]['unit_cost']) > 0:
        ratio = _divide(costs[-1]['unit_cost'], Decimal(costs[0]['unit_cost']))
        parts.append('<p class="compare-note">' + e(cohort_label) + '完整费用比较：' + e(costs[-1]['label']) + '约为'
                     + e(costs[0]['label']) + '的 <strong>' + fixed(ratio) + '倍</strong>。</p>')
    if presentation['shared_glm_search_cost'] is not None:
        parts.append('<p class="pending">' + e(presentation['shared_glm_search_date']) + '智谱共用搜索费 <strong>' + unit_money(presentation['shared_glm_search_cost']) + '元</strong>未分摊到两款GLM；资源包抵扣0元不代表完整费用为0。</p>')
    else:
        parts.append('<p class="pending">七款模型均已列出；完整单图费用待核的模型不参与成本排名。</p>')
    parts.append('<p class="sub">成本排名仅含' + e(cohort_label) + '的' + e('、'.join(row['label'] for row in costs) or '待核模型')
                 + '；不属于所示比较批次的费用不参与排名。仅推理费用未含待核搜索。</p>')
    parts.append('</section><section class="panel"><h2>七模型识别表现</h2><p class="sub">' + (e(f"联网C组 · {scope['images']}张图片 · {scope['targets']}个必需角色目标") if scope else '正式评测结果待补充') + '</p><div class="accuracy-bars" role="img" aria-label="模型姓名命中率比较">')
    for index, row in enumerate(quality):
        style = ' highlight' if index == 0 else ' runner' if index == 1 else ' kimi' if row['model_id'] == 'kimi-k3.1-pro' else ''
        rate = fixed(row['accuracy_percent'])
        parts.append('<div class="accuracy-item' + style + '" title="' + e(f"{row['correct']}/{row['targets']}个必需目标命中") + '"><div class="row-top"><span>' + e(row['label']) + '</span><strong>' + rate + '%</strong></div><div class="track track-small"><div class="fill" style="width:' + rate + '%"></div></div></div>')
    if not quality:
        parts.append('<p class="empty">尚未接入正式识别结果；本页不据此判断性价比。</p>')
    parts.append('</div><p class="sub" style="margin-top:17px">无框原图 + 作品名称；此处只核对角色姓名，忽略坐标和输出格式。</p></section></div>')
    parts.append('<section class="findings"><div class="findings-heading"><h2>结论与建议</h2><span class="sub">结论限定于本次测试数据</span></div><div class="findings-grid">')
    findings = []
    if preferred:
        findings.append(('优先试用 ' + preferred['label'], '本测试姓名命中率' + fixed(preferred['accuracy_percent']) + '%，'
                         + cohort_label + '已核均价' + unit_money(leader['unit_cost']) + '元/图；在同批已核模型中兼顾效果与成本。'))
    else:
        findings.append(('先完成成本与效果核对', '待同范围账单和正式识别结果齐备后，再判断模型性价比。'))
    if len(quality) >= 2:
        second = quality[1]
        second_cost = next((row for row in presentation['cost_overview'] if row['model_id'] == second['model_id']), None)
        cost_note = '完整单图费用待核，补齐后再比较性价比。'
        if second_cost and second_cost['unit_kind'] == 'daily_reference':
            cost_note = '日账参考约' + unit_money(second_cost['unit_cost']) + '元/图，需注意跨日及批次差异。'
        elif second_cost and second_cost['unit_kind'] == 'verified' and second_cost['source'] == 'historical_batch':
            cost_note = second_cost['date'] + '历史已核' + unit_money(second_cost['unit_cost']) + '元/图，不加入本批成本排名。'
        elif second['model_id'] not in presentation['pending_cost_models']:
            cost_note = '可继续与首选模型对比实际业务表现。'
        findings.append(('保留 ' + second['label'] + ' 对照', '姓名命中率' + fixed(second['accuracy_percent']) + '%，本组排名第二。' + cost_note))
    else:
        findings.append(('补齐其他模型费用', '未核清的费用保留待核，避免把资源包抵扣或缺失金额当作免费。'))
    findings.append(('按业务量判断预算', ('按' + cohort_label + head_name + '均价外推，千张约' + fixed(leader['per_1000']) + '元。' if leader else '') + '是否可接受，还需结合业务预算与人工复核成本。'))
    for index, (title, body) in enumerate(findings, 1):
        parts.append('<article class="finding"><p class="finding-index">0' + str(index) + '</p><h3>' + e(title) + '</h3><p>' + e(body) + '</p></article>')
    parts.append('</div></section><footer class="foot"><p>金额来自官方账单；本批与历史各用自己的日期和任务数。当前仅推理金额不是完整费用，未知搜索不按0计算；“≈”为日账均摊参考，旧接口调试价不代表当前接口。千张费用仅为外推，未计人工复核。</p><details class="method"><summary>统计口径</summary><p>图指一次图片分析任务，同一原图在不同批次重复测试分别计数。优先比较同一本批图片且完整费用已核的模型；不足两款时沿用同范围历史批次。七行总览可展示不同日期，不能据此跨批排名。其他日账参考以对应模型官方已出金额除以本地当日结束的逻辑图片任务数，重试和内部多轮不重复计图片。跨午夜归属可能带来日界误差；共享搜索费不擅自拆到模型。</p><p>姓名命中率按已复核的必需角色目标统计，费用与效果不按逐条请求配对，也不据此计算“每次正确识别成本”。姓名指标不等同于人物定位、完整输出或生产环境成功率。</p></details></footer></main></html>')
    return ''.join(parts)
