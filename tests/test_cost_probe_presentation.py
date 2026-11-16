"""Presentation claims must keep comparable costs and scored samples separate."""
from copy import deepcopy
from decimal import Decimal

import pytest

from tests.test_cost_probe_report import manifest
from vcrb.cost_probe_report import render_report, summarize_probe
from vcrb.cost_probe_presentation import build_presentation


def briefing_data():
    data = manifest()
    data['historical_batches'] = [
        {'date': '2026-11-15', 'model_id': 'deepseek-flash', 'currency': 'CNY',
         'image_tasks': 168, 'amount': '5.79634', 'inference_and_search_confirmed': True},
        {'date': '2026-11-15', 'model_id': 'kimi-k3.1-pro', 'currency': 'CNY',
         'image_tasks': 168, 'amount': '14.36764', 'inference_and_search_confirmed': True},
    ]
    hits = [189, 154, 100, 21, 172, 139, 126]
    scores = {'dataset_hash': data['dataset_hash'], 'all_works': {
        row['model_id']: {'new': {'planned_images': 160, 'completed_images': 160,
            'planned_targets': 213, 'name_correct': hit,
            'strict_correct': 0, 'strict_planned_targets': 219}}
        for row, hit in zip(data['models'], hits)}}
    return data, scores


def test_costs_share_a_cohort_and_recognition_uses_names_not_strict_coordinates():
    data, scores = briefing_data()
    view = build_presentation(summarize_probe(data), scores)
    assert len(view['costs']) == 2 and len(view['pending_cost_models']) == 5
    assert view['cost_cohort'] == {'kind': 'historical', 'date': '2026-11-15', 'image_tasks': 168}
    assert view['recognition'][0]['correct'] == 189 and view['recognition'][0]['targets'] == 213
    assert Decimal(view['recognition'][0]['accuracy_percent']).quantize(Decimal('.01')) == Decimal('88.73')
    assert view['preferred']['model_id'] == 'deepseek-flash'
    assert Decimal(view['kimi_to_deepseek_cost_ratio']).quantize(Decimal('.01')) == Decimal('2.48')
    assert Decimal(view['costs'][0]['per_1000']).quantize(Decimal('.01')) == Decimal('34.50')


@pytest.mark.parametrize('changed', [{'date': '2026-11-14'}, {'image_tasks': 160}])
def test_different_dates_or_task_counts_are_never_paired_as_cost_comparison(changed):
    data, _ = briefing_data()
    data['historical_batches'][1].update(changed)
    view = build_presentation(summarize_probe(data))
    assert len(view['costs']) == 1
    assert view['kimi_to_deepseek_cost_ratio'] is None
    assert view['preferred'] is None


def test_same_day_large_cohort_takes_precedence_over_separate_five_image_probe():
    data, _ = briefing_data()
    summary = summarize_probe(data)
    summary['models'][0].update(billing_complete=True, charges=[{
        'currency': 'CNY', 'official_amount': '0.18568688',
        'average_per_submitted_image': '0.037137376'}])
    view = build_presentation(summary)
    assert view['cost_cohort']['image_tasks'] == 168
    assert view['costs'][0]['unit_cost'].startswith('0.03450202')


def test_all_seven_models_show_posted_costs_without_allocating_shared_search():
    data, scores = briefing_data()
    rows = [
        ('qwen3.8-max', 'inference', '20.01258', 'posted'),
        ('qwen3.8-max', 'search', '0.876', 'posted'),
        ('qwen3.8-flash', 'inference', '6.3276301', 'posted'),
        ('qwen3.8-flash', 'search', '4.432', 'posted'),
        ('qwen3.7-plus', 'inference', '1.5006592', 'inference_only'),
        ('glm-4.6v', 'inference', '1.8695358', 'posted'),
        ('glm-5.3-flash', 'inference', '0', 'resource_pack'),
        ('search-pro', 'search', '8.9782808', 'search_unallocated'),
    ]
    data['historical_daily_bills'] = [{'date': '2026-11-15', 'vendor': 'GLM' if mid.startswith('glm') or mid == 'search-pro' else 'Qwen',
        'model_id': mid, 'kind': kind, 'currency': 'CNY', 'amount': amount, 'note_code': note}
        for mid, kind, amount, note in rows]
    view = build_presentation(summarize_probe(data), scores)
    all_models = {row['model_id']: row for row in view['cost_overview']}
    assert len(all_models) == 7 and view['shared_glm_search_cost'] == '8.9782808'
    assert all_models['deepseek-flash']['posted_amount'] == '5.79634'
    assert all_models['qwen3.8-max']['posted_amount'] == '20.88858'
    assert all_models['qwen3.8-flash']['posted_amount'] == '10.7596301'
    assert all_models['glm-4.6v']['posted_amount'] == '1.8695358'
    assert all_models['glm-5.3-flash']['posted_amount'] == '0'
    for mid in ['qwen3.8-max', 'qwen3.8-flash', 'qwen3.7-plus', 'glm-4.6v', 'glm-5.3-flash']:
        assert all_models[mid]['unit_cost'] is None
    assert '资源包' in all_models['glm-5.3-flash']['note']
    assert len(view['costs']) == 2  # Only the comparable, complete pair is ranked.


def test_official_daily_fees_support_references_without_per_request_meter_matching():
    data, scores = briefing_data()
    rows = [('qwen3.8-max', 'inference', '20.01258'), ('qwen3.8-max', 'search', '0.876'),
            ('qwen3.8-flash', 'inference', '6.3276301'), ('qwen3.8-flash', 'search', '4.432'),
            ('qwen3.7-plus', 'inference', '1.5006592'), ('glm-4.6v', 'inference', '1.8695358'),
            ('glm-5.3-flash', 'inference', '0'), ('search-pro', 'search', '8.9782808')]
    data['historical_daily_bills'] = [{'date': '2026-11-15', 'model_id': mid,
        'vendor': 'GLM' if mid.startswith('glm') or mid == 'search-pro' else 'Qwen',
        'kind': kind, 'currency': 'CNY', 'amount': amount,
        'note_code': 'resource_pack' if mid == 'glm-5.3-flash' else 'posted'} for mid, kind, amount in rows]
    data['daily_reference_tasks'] = [
        {'date': '2026-11-15', 'model_id': mid, 'image_tasks': count, 'fee_scope': scope,
         'contains_debugging': mid == 'qwen3.8-flash', 'cross_midnight': mid == 'qwen3.8-max'}
        for mid, count, scope in [('qwen3.8-max', 173, 'inference_and_search'),
            ('qwen3.8-flash', 67, 'inference_and_search'), ('qwen3.7-plus', 169, 'inference_only'),
            ('glm-4.6v', 170, 'inference_only'), ('glm-5.3-flash', 170, 'inference_only')]]
    view = build_presentation(summarize_probe(data), scores)
    costs = {row['model_id']: row for row in view['cost_overview']}
    assert all(row['unit_cost'] is not None for row in costs.values())
    assert Decimal(costs['qwen3.8-max']['unit_cost']).quantize(Decimal('.00000001')) == Decimal('0.12074324')
    assert Decimal(costs['qwen3.8-flash']['unit_cost']).quantize(Decimal('.00000001')) == Decimal('0.16059149')
    assert Decimal(costs['qwen3.7-plus']['unit_cost']).quantize(Decimal('.00000001')) == Decimal('0.00887964')
    assert Decimal(costs['glm-4.6v']['unit_cost']).quantize(Decimal('.00000001')) == Decimal('0.01099727')
    assert costs['glm-5.3-flash']['unit_kind'] == 'inference_only'
    assert Decimal(costs['glm-5.3-flash']['unit_cost']) == 0
    assert costs['qwen3.8-max']['unit_kind'] == 'daily_reference'
    assert len(view['costs']) == 2 and view['shared_glm_search_cost'] == '8.9782808'
    data['daily_reference_tasks'][2]['fee_scope'] = 'inference_and_search'
    with pytest.raises(ValueError, match='requires official search'):
        summarize_probe(data)


def test_stale_annotations_or_different_recognition_denominators_are_rejected():
    data, scores = briefing_data()
    bad = deepcopy(scores); bad['dataset_hash'] = 'b' * 64
    with pytest.raises(ValueError, match='dataset versions'):
        build_presentation(summarize_probe(data), bad)
    bad = deepcopy(scores)
    bad['all_works']['kimi-k3.1-pro']['new']['planned_targets'] = 214
    with pytest.raises(ValueError, match='same sample'):
        build_presentation(summarize_probe(data), bad)


def test_explicit_zero_cost_does_not_invent_a_ratio_or_zero_out_unknown_models():
    data, scores = briefing_data()
    data['historical_batches'][0]['amount'] = '0'
    view = build_presentation(summarize_probe(data), scores)
    assert view['costs'][0]['unit_cost'] == '0'
    assert view['kimi_to_deepseek_cost_ratio'] is None
    assert len(view['pending_cost_models']) == 5


def test_report_removes_execution_and_daily_tables_but_labels_extrapolation(tmp_path):
    data, scores = briefing_data()
    data['historical_daily_bills'] = [{'date': '2026-11-15', 'model_id': 'qwen3.8-max',
        'vendor': 'Qwen', 'currency': 'CNY', 'kind': 'inference', 'amount': '99999.123456', 'note_code': 'posted'}]
    result = render_report(data, tmp_path / 'index.html', recognition=scores)
    html = (tmp_path / 'index.html').read_text()
    assert '99999.123456' not in html and '<table' not in html
    assert '推理HTTP' not in html and '历史项目费用：官方已出日账' not in html
    assert '非实际账单' in html and '七款模型均已列出' in html and '0.0345' in html
    assert '88.73%' in html and '80.75%' in html and '优先试用 DeepSeek Flash' in html
    assert '<script' not in html and '<img' not in html and 'https://' not in html
    assert result['historical_daily_bills'][0]['amount'] == '99999.123456'
