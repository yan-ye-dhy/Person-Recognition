"""Selective offline failure retries retain the original model/sample cohort."""
from __future__ import annotations

import sqlite3

import pytest

from tests.test_multi_runner import demo, offline_only, manual_test_fixture, prices, read_run
from tests.test_multi_budget import UsageProbe, ledger_rows
from tests.test_multi_search_runner import tool_prices
from vcrb.multi import runner
from vcrb.multi.schema import load_dataset
from vcrb.providers.base import ProviderResult


QWEN = 'qwen3.8-max'
GLM = 'glm-5.3-flash'


class FailFirst(UsageProbe):
    def recognize(self, *args):
        result = super().recognize(*args)
        if len(self.calls) == 1:
            result.ok = False
            result.error_type = 'bad_request'
            result.error_message = 'offline original failure; never a real API call'
        return result


def start_partial(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    probes = {mid: FailFirst() for mid in [QWEN, GLM]}
    selected = [sample.sample_id for sample in load_dataset(dataset)[:2]]
    kwargs = {**options, 'mock':False, 'yes':True, 'model_ids':[QWEN, GLM], 'experiments':('C',),
              'sample_ids':selected, 'max_cost_cny':350, 'pricing':prices([QWEN, GLM]),
              'approval_note':'OFFLINE TEST ONLY', 'provider_overrides':probes}
    initial = runner.run_multi(dataset, **kwargs, max_requests=2)
    assert initial['executed'] == 2 and initial['progress'] == {'total':2,'ok':0,'failed':2}
    return dataset, kwargs, probes, initial


def test_only_selected_model_failures_retry_while_new_tasks_for_all_models_continue(demo):
    dataset, kwargs, probes, initial = start_partial(demo)
    _, original = read_run(kwargs, initial['run_id'])
    old_glm = next(row for row in original if row['model_id'] == GLM)
    result = runner.run_multi(dataset, **kwargs, run_id=initial['run_id'], retry_failed=True,
                              retry_failed_models=[QWEN])
    assert result['planned'] == 4 and result['executed'] == 3 and result['skipped'] == 1
    assert result['progress'] == {'total':4,'ok':3,'failed':1}
    assert len(probes[QWEN].calls) == 3 and len(probes[GLM].calls) == 2
    run, records = read_run(kwargs, initial['run_id'])
    assert run['models'] == [QWEN, GLM] and run['experiments'] == ['C']
    assert next(row for row in records if row['model_id'] == GLM and row['sample_id'] == old_glm['sample_id']) == old_glm
    again = runner.run_multi(dataset, **kwargs, run_id=initial['run_id'], retry_failed=True,
                             retry_failed_models={QWEN})
    assert again['executed'] == 0 and again['skipped'] == 4
    assert len(probes[QWEN].calls) == 3 and len(probes[GLM].calls) == 2


@pytest.mark.parametrize('selector,retry_failed,expected_new,expected_ok', [
    (None, True, 4, 4), (set(), True, 2, 2), ([QWEN], False, 2, 2),
])
def test_none_keeps_legacy_all_empty_means_none_and_flag_false_does_not_retry(
        demo, selector, retry_failed, expected_new, expected_ok):
    dataset, kwargs, _probes, initial = start_partial(demo)
    result = runner.run_multi(dataset, **kwargs, run_id=initial['run_id'], retry_failed=retry_failed,
                              retry_failed_models=selector)
    assert result['executed'] == expected_new
    assert result['progress'] == {'total':4,'ok':expected_ok,'failed':4-expected_ok}


@pytest.mark.parametrize('selector', ['qwen3.8-max', ['missing'], ['kimi-k3.1-pro'], [True], 1])
def test_invalid_retry_selector_is_rejected_before_provider_or_run_creation(demo, selector):
    dataset, options = demo
    manual_test_fixture(dataset)
    probes = {mid: UsageProbe() for mid in [QWEN, GLM]}
    with pytest.raises(ValueError, match='retry_failed_models'):
        runner.run_multi(dataset, **options, mock=False, yes=True, model_ids=[QWEN, GLM],
                         experiments=('C',), pricing=prices([QWEN, GLM]), max_cost_cny=350,
                         approval_note='OFFLINE TEST ONLY', provider_overrides=probes,
                         retry_failed=True, retry_failed_models=selector)
    assert all(not probe.calls for probe in probes.values())
    with sqlite3.connect(options['db_path']) as conn:
        assert conn.execute('select count(*) from runs').fetchone()[0] == 0


def test_http_200_balance_error_blocks_sibling_vendor_before_second_recognize(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    plus = 'qwen3.7-plus'
    flash = 'qwen3.8-flash'
    pricing = prices([QWEN, plus, flash])
    pricing['models'][QWEN] = tool_prices()['models'][QWEN]
    class BalanceBody:
        supports_attempt_observer = True
        name = 'offline-http200-balance-error'
        calls = 0
        def recognize(self, model, image, system, prompt, gen, attempt_observer):
            receipt = attempt_observer.before_attempt(model, image, prompt, 1)
            self.calls += 1
            # A gateway can return HTTP200 while the SSE body rejects access.
            # Even usage-looking fields must not release this failed attempt.
            result = ProviderResult(ok=False, http_status=200, attempts=1,
                error_type='insufficient_balance', error_message='offline Arrearage body',
                usage={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120})
            attempt_observer.after_attempt(receipt, model, result=result, error=None)
            return result
    first, sibling, other = BalanceBody(), UsageProbe(), UsageProbe()
    result = runner.run_multi(dataset, **options, mock=False, yes=True, model_ids=[QWEN, plus, flash],
                              experiments=('C',), pricing=pricing, max_cost_cny=350, limit=1,
                              approval_note='OFFLINE TEST ONLY', provider_overrides={QWEN:first, plus:sibling, flash:other})
    assert first.calls == 1 and not sibling.calls and not other.calls
    assert result['executed'] == 1 and result['budget']['blocked_providers'] == ['dashscope']
    rows = ledger_rows(options, result['run_id'])
    assert len(rows) == 1 and rows[0]['model_id'] == QWEN
    assert rows[0]['status'] == 'attempt_uncertain' and rows[0]['known_cny'] == 0
    expected = runner.task_upper_cost(QWEN, runner.load_settings(), pricing) / runner.load_settings()['retry']['max_attempts']
    assert rows[0]['uncertain_cny'] == rows[0]['committed_cny'] == pytest.approx(expected)


class MixedFailures(UsageProbe):
    def recognize(self, *args):
        result = super().recognize(*args)
        if len(self.calls) <= 2:
            result.ok = False
            result.error_type = 'insufficient_balance' if len(self.calls) == 1 else 'bad_request'
            result.error_message = 'offline balance error' if len(self.calls) == 1 else 'offline content filter or repetitive tool calls'
        return result


def start_mixed_failures(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    probes = {mid:MixedFailures() for mid in [QWEN, GLM]}
    kwargs = {**options,'mock':False,'yes':True,'model_ids':[QWEN,GLM],'experiments':('C',),
              'sample_ids':[s.sample_id for s in load_dataset(dataset)[:3]],'max_cost_cny':350,
              'pricing':prices([QWEN,GLM]),'approval_note':'OFFLINE TEST ONLY','provider_overrides':probes}
    first = runner.run_multi(dataset, **kwargs, max_requests=4)
    assert first['progress'] == {'total':4,'ok':0,'failed':4}
    return dataset, kwargs, probes, first


def test_error_and_model_filters_intersect_and_preserve_other_failures_with_one_snapshot(demo, monkeypatch):
    dataset, kwargs, probes, first = start_mixed_failures(demo)
    _, original = read_run(kwargs, first['run_id'])
    preserved = {(row['model_id'],row['sample_id']):row for row in original
                 if row['model_id'] != GLM or row['raw']['response']['error_type'] != 'insufficient_balance'}
    records_calls = []
    original_records = runner.RunStore.records
    def snapshot(store, rid):
        records_calls.append(rid)
        return original_records(store, rid)
    monkeypatch.setattr(runner.RunStore, 'records', snapshot)
    execute = runner.execute_run
    iterations = []
    def one_per_scheduling_pass(*args, **options):
        iterations.append(1)
        options['max_new_tasks'] = 1
        return execute(*args, **options)
    monkeypatch.setattr(runner, 'execute_run', one_per_scheduling_pass)
    result = runner.run_multi(dataset, **kwargs, run_id=first['run_id'], retry_failed=True,
                              retry_failed_models={GLM}, retry_failed_error_types={'insufficient_balance'})
    assert result['executed'] == 3 and result['skipped'] == 3
    assert result['progress'] == {'total':6,'ok':3,'failed':3}
    assert len(probes[GLM].calls) == 4 and len(probes[QWEN].calls) == 3
    assert len(iterations) == 3 and records_calls == [first['run_id']]
    _, final = read_run(kwargs, first['run_id'])
    for row in final:
        key = row['model_id'],row['sample_id']
        if key in preserved:
            assert row == preserved[key]


@pytest.mark.parametrize('types,retry,expected_new,expected_ok', [
    (None,True,6,6),({'insufficient_balance'},True,4,4),
    (set(),True,2,2),({'unknown-error'},True,2,2),({'insufficient_balance'},False,2,2),
])
def test_error_filter_none_empty_unknown_and_disabled_semantics(demo, types, retry, expected_new, expected_ok):
    dataset, kwargs, _probes, first = start_mixed_failures(demo)
    result = runner.run_multi(dataset, **kwargs, run_id=first['run_id'], retry_failed=retry,
                              retry_failed_error_types=types)
    assert result['executed'] == expected_new
    assert result['progress'] == {'total':6,'ok':expected_ok,'failed':6-expected_ok}


@pytest.mark.parametrize('selector', ['insufficient_balance',[None],[''],[' '],True,{}])
def test_invalid_retry_error_filter_is_rejected_before_calls_and_new_run(demo, selector):
    dataset, options = demo
    manual_test_fixture(dataset)
    probe = UsageProbe()
    with pytest.raises(ValueError, match='retry_failed_error_types'):
        runner.run_multi(dataset, **options, mock=False, yes=True, model_ids=[GLM],
                         experiments=('C',),pricing=prices([GLM]),max_cost_cny=350,
                         approval_note='OFFLINE TEST ONLY',provider_overrides={GLM:probe},
                         retry_failed=True,retry_failed_error_types=selector)
    assert not probe.calls
    with sqlite3.connect(options['db_path']) as conn:
        assert conn.execute('select count(*) from runs').fetchone()[0] == 0
