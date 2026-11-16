"""Official-charge import and offline privacy tests; no live provider calls."""
from copy import deepcopy
import csv
from decimal import Decimal
import json
import subprocess
import sys

import pytest

from vcrb.cost_probe_report import CSV_FIELDS, import_billing_csv, render_report, summarize_probe, validate_billing, validate_manifest, write_billing_template


def manifest():
    ids = ['deepseek-flash', 'kimi-k3.1-pro', 'glm-5.3-flash', 'glm-4.6v', 'qwen3.8-max', 'qwen3.7-plus', 'qwen3.8-flash']
    return {'probe_id': 'actual_cost_probe_01', 'dataset_hash': 'a' * 64,
            'planned_sample_ids': [f'sample_{n}' for n in range(5)],
            'models': [{'model_id': mid, 'vendor': 'testvendor', 'planned_count': 5, 'submitted_count': 5,
                        'success_count': 4, 'failure_count': 1, 'api_attempts': 6, 'search_attempts': None,
                        'status': 'completed', 'started_at': '2026-11-16T01:00:00Z', 'finished_at': '2026-11-16T01:01:00Z'} for mid in ids]}


def entry(**changes):
    return {'bill_id': 'official-detail-1', 'model_id': 'qwen3.8-max', 'currency': 'CNY', 'amount': '0.123456789',
            'kind': 'combined', 'cash_amount': '0.100000000', 'credit_amount': '0.023456789',
            'source': '/private/official-export.csv', 'period_start': '2026-11-16T01:00:00Z', 'period_end': '2026-11-16T01:01:00Z', **changes}


def bill(*entries, complete=True):
    return {'entries': list(entries), 'complete_models': ['qwen3.8-max'] if complete else []}


def test_pending_is_never_zero_and_all_seven_planned_counts_are_preserved(tmp_path):
    data = manifest()
    for row in data['models']:
        row.update(status='pending', submitted_count=0, success_count=0, failure_count=0, started_at=None, finished_at=None)
    path = tmp_path / 'index.html'
    summary = render_report(data, path)
    html = path.read_text()
    assert len(summary['models']) == 7
    assert sum(row['planned_count'] for row in summary['models']) == 35
    assert sum(row['submitted_count'] for row in summary['models']) == 0
    assert all(row['charges'] == [] and not row['billing_complete'] for row in summary['models'])
    assert all(row['billing_status'] == 'pending_official_billing' for row in summary['models'])
    assert '<script' not in html and '<img' not in html and '<iframe' not in html


def test_incomplete_official_exports_explain_pending_without_creating_fees(tmp_path):
    data = manifest()
    data['models'][1]['billing_evidence_status'] = 'request_log_only'
    data['models'][2].update(billing_evidence_status='date_not_covered', billing_latest_date='2026-11-15')
    data['models'][4]['billing_evidence_status'] = 'daily_aggregate_not_attributed'
    summary = render_report(data, tmp_path / 'index.html')
    assert summary['models'][1]['billing_evidence_status'] == 'request_log_only'
    assert summary['models'][2]['billing_evidence_status'] == 'date_not_covered'
    assert summary['models'][2]['billing_latest_date'] == '2026-11-15'
    assert summary['models'][4]['billing_evidence_status'] == 'daily_aggregate_not_attributed'
    assert all(not row['billing_complete'] and row['charges'] == [] for row in summary['models'])
    data['models'][1]['billing_evidence_status'] = 'unexpected raw account data'
    with pytest.raises(ValueError, match='billing evidence status'):
        validate_manifest(data)


def test_historical_daily_bills_do_not_enter_five_image_probe_cost(tmp_path):
    data = manifest()
    data['historical_daily_bills'] = [
        {'date': '2026-11-15', 'vendor': 'Qwen', 'model_id': 'qwen3.8-max',
         'kind': 'inference', 'currency': 'CNY', 'amount': '20.012580000000', 'note_code': 'posted',
         'private_account_id': 'must-not-render-account'},
        {'date': '2026-11-15', 'vendor': 'Kimi', 'model_id': 'kimi-k3.1-pro',
         'kind': 'combined', 'currency': 'CNY', 'amount': None, 'request_count': 168, 'note_code': 'no_amount'},
    ]
    result = render_report(data, tmp_path / 'index.html')
    html = (tmp_path / 'index.html').read_text()
    assert 'must-not-render-account' not in html
    assert all(row['charges'] == [] and not row['billing_complete'] for row in result['models'])
    assert result['historical_daily_bills'][0]['amount'] == '20.012580000000'
    assert result['historical_daily_bills'][1]['amount'] is None
    assert result['historical_daily_bills'][1]['request_count'] == 168
    assert result['historical_daily_bills'][1]['note_code'] == 'no_amount'
    assert 'private_account_id' not in json.dumps(result)
    data['historical_daily_bills'].append(deepcopy(data['historical_daily_bills'][0]))
    with pytest.raises(ValueError, match='Duplicate historical'):
        validate_manifest(data)


def test_historical_batch_average_uses_its_own_images_and_requires_search_fees(tmp_path):
    data = manifest()
    data['historical_batches'] = [{'date': '2026-11-15', 'model_id': 'deepseek-flash',
        'currency': 'CNY', 'image_tasks': 168, 'amount': '5.79634', 'inference_and_search_confirmed': True}]
    result = summarize_probe(data)
    assert result['historical_batches'][0]['average_per_image_task'].startswith('0.0345020238')
    assert all(not row['charges'] for row in result['models'])
    data['historical_batches'][0]['inference_and_search_confirmed'] = False
    with pytest.raises(ValueError, match='complete fees'):
        validate_manifest(data)


def test_exact_bill_average_uses_submitted_including_failure_and_retry_not_success(tmp_path):
    summary = render_report(manifest(), tmp_path / 'index.html', bill(entry()))
    row = next(row for row in summary['models'] if row['model_id'] == 'qwen3.8-max')
    charge = row['charges'][0]
    assert charge['official_amount'] == '0.123456789'
    assert charge['average_per_submitted_image'] == '0.0246913578'
    assert charge['cash_amount'] == '0.100000000' and charge['credit_amount'] == '0.023456789'
    assert row['started_at'] == '2026-11-16T09:00:00+08:00'
    assert row['api_attempts'] == 6 and row['submitted_count'] == 5 and row['success_count'] == 4


def test_partial_evidence_and_missing_payment_parts_remain_unknown():
    summary = summarize_probe(manifest(), bill(entry(amount='0', cash_amount=None, credit_amount=None), complete=False))
    row = next(row for row in summary['models'] if row['model_id'] == 'qwen3.8-max')
    assert row['charges'][0]['official_amount'] is None and row['charges'][0]['cash_amount'] is None
    assert row['charges'][0]['average_per_submitted_image'] is None
    summary = summarize_probe(manifest(), bill(entry(amount='0', cash_amount=None, credit_amount=None)))
    row = next(row for row in summary['models'] if row['model_id'] == 'qwen3.8-max')
    assert row['charges'][0]['official_amount'] == '0' and row['charges'][0]['average_per_submitted_image'] == '0'
    assert row['charges'][0]['cash_amount'] is None and row['charges'][0]['credit_amount'] is None


@pytest.mark.parametrize('amount', [0.1, True, '-1', 'NaN', 'Infinity', '1e3', '', None])
def test_invalid_or_binary_float_money_is_rejected(amount):
    with pytest.raises(ValueError):
        validate_billing(manifest(), bill(entry(amount=amount)))


def test_duplicate_detail_id_and_mixed_component_total_are_rejected():
    with pytest.raises(ValueError, match='Duplicate bill_id'):
        validate_billing(manifest(), bill(entry(), entry(bill_id=' official-detail-1 ')))
    with pytest.raises(ValueError, match='combined'):
        validate_billing(manifest(), bill(entry(), entry(bill_id='component', kind='search')))
    with pytest.raises(ValueError, match='ambiguous'):
        validate_billing(manifest(), bill(entry(model_id='qwen3.8-max,qwen3.7-plus')))


def test_inference_search_components_sum_exactly_but_currencies_never_mix():
    official = bill(entry(bill_id='i', kind='inference', amount='0.1'),
                    entry(bill_id='s', kind='search', amount='0.2', cash_amount=None),
                    entry(bill_id='u', kind='search', amount='1.25', currency='USD'))
    row = next(row for row in summarize_probe(manifest(), official)['models'] if row['model_id'] == 'qwen3.8-max')
    assert [charge['currency'] for charge in row['charges']] == ['CNY', 'USD']
    assert [charge['official_amount'] for charge in row['charges']] == ['0.3', '1.25']
    assert row['charges'][0]['cash_amount'] is None


def test_private_fields_bill_ids_sources_account_and_keys_do_not_leave_report(tmp_path):
    private = 'sk-do-not-expose-123456789'
    data = manifest()
    data.update(api_key=private, account_id='ACCOUNT_PRIVATE', balance='BALANCE_PRIVATE', inputs={'body': private})
    data['models'][0]['raw_response'] = {'account': 'ACCOUNT_PRIVATE', 'key': private}
    official = bill(entry(bill_id='BILL_PRIVATE', source=private))
    path = tmp_path / 'index.html'
    summary = render_report(data, path, official)
    emitted = path.read_text() + json.dumps(summary)
    for text in [private, 'ACCOUNT_PRIVATE', 'BALANCE_PRIVATE', 'BILL_PRIVATE', '/private/official-export.csv']:
        assert text not in emitted
    charge = next(row for row in summary['models'] if row['model_id'] == 'qwen3.8-max')['charges'][0]
    assert charge['cash_amount'] == '0.100000000'
    assert charge['credit_amount'] == '0.023456789'


def test_manifest_counts_timestamps_and_unfinished_bill_validation():
    for changes in [dict(submitted_count=6), dict(success_count=6), dict(planned_count=4), dict(started_at='2026-11-16T01:00:00'), dict(status=['invalid'])]:
        data = manifest(); data['models'][0].update(changes)
        with pytest.raises(ValueError):
            validate_manifest(data)
    data = manifest(); data['models'][4]['status'] = 'running'
    with pytest.raises(ValueError, match='finished batch'):
        validate_billing(data, bill(entry()))
    with pytest.raises(ValueError, match='evidence'):
        validate_billing(manifest(), bill())


def test_csv_explicit_template_import_and_no_model_inference(tmp_path):
    path = tmp_path / 'bills.csv'
    write_billing_template(path)
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writerow(entry())
    result = import_billing_csv(manifest(), path, complete_models=['qwen3.8-max'])
    assert result['entries'][0]['amount'] == '0.123456789'
    with path.open('a', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=CSV_FIELDS).writerow(entry())
    with pytest.raises(ValueError, match='Duplicate'):
        import_billing_csv(manifest(), path)


def test_cli_offline_json_import(tmp_path):
    man = tmp_path / 'private.json'; bills = tmp_path / 'bills.json'; output = tmp_path / 'index.html'; saved = tmp_path / 'private' / 'billing.json'
    man.write_text(json.dumps(manifest())); bills.write_text(json.dumps(bill(entry())))
    result = subprocess.run([sys.executable, 'scripts/import_cost_probe_bill.py', '--manifest', str(man), '--billing-json', str(bills), '--output', str(output), '--save-billing', str(saved)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['officially_confirmed'] == 1 and output.exists()
    assert saved.stat().st_mode & 0o777 == 0o600


def test_independent_search_http_zero_never_implies_native_search_zero(tmp_path):
    data = manifest()
    data['models'][4].update(search_attempts=0, native_tool_units={'search': 12, 'fetch': 3}, native_tool_usage_known_attempts=5)
    path = tmp_path / 'index.html'
    summary = render_report(data, path)
    row = summary['models'][4]
    assert row['search_attempts'] == 0
    assert row['native_tool_units'] == {'search': 12, 'fetch': 3}
    assert row['native_tool_usage_known_attempts'] == 5
    assert summary['models'][0]['native_tool_units'] == {}
    assert summary['models'][0]['native_tool_usage_known_attempts'] is None
    assert all(not model['billing_complete'] and model['charges'] == [] for model in summary['models'])
    bad = deepcopy(data); bad['models'][4]['native_tool_units']['search'] = 1.5
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_concise_render_preserves_exact_summary_and_excludes_admin_links(tmp_path):
    data = manifest()
    data['models'][4].update(submitted_count=3, success_count=2, failure_count=1, status='stopped')
    path = tmp_path / 'index.html'
    summary = render_report(data, path, bill(entry(amount='1')))
    row = summary['models'][4]['charges'][0]
    assert row['official_amount'] == '1'
    assert row['average_per_submitted_image'].startswith('0.' + '3' * 50)
    html = path.read_text()
    assert row['average_per_submitted_image'] not in html
    for admin_url in ('open.bigmodel.cn/finance/expensebill/list?active=detail',
                      'platform.kimi.com/console', 'platform.deepseek.com/usage'):
        assert admin_url not in html
    summary = render_report(data, path, bill(entry(amount='0.00000000001')))
    tiny_charge = summary['models'][4]['charges'][0]
    assert tiny_charge['official_amount'] == '0.00000000001'
    assert Decimal(tiny_charge['average_per_submitted_image']) > 0
    assert tiny_charge['average_per_submitted_image'].startswith('0.000000000003333333')
