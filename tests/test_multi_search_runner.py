"""Offline search-pilot scope and fee accounting; never accesses real credentials."""
from copy import deepcopy
import json
import sqlite3
from pathlib import Path

import pytest

from tests.test_multi_runner import (
    demo, offline_only, manual_test_fixture, prices, paid_kwargs, read_run,
)
from tests.test_multi_budget import UsageProbe, ledger_rows, attempt_adapter
from tests.test_providers import _resp, _StatusErr, _baseline_qwen
from vcrb.multi import runner
from vcrb.multi.schema import load_dataset
from vcrb.providers.base import ProviderResult

MID = 'qwen3.8-max'
FEE = .003


@pytest.fixture
def baseline_chat_model(monkeypatch):
    """Token/fee tests use a non-stream fake independently of live search config."""
    resolve = runner.resolve_models
    baseline = _baseline_qwen()
    monkeypatch.setattr(runner, 'resolve_models', lambda ids=None: [
        baseline if model.id == MID else model for model in resolve(ids)
    ])


def search_prices():
    pricing = prices([MID])
    pricing['models'][MID].update(
        request_fee_upper_cny=FEE,
        request_fee_source_url='https://example.org/official-offline-search-price',
    )
    return pricing


def pilot_kwargs(options, pricing=None):
    kwargs = paid_kwargs(options, MID)
    kwargs.update(limit=0, experiments=('C',), pricing=pricing or search_prices())
    return kwargs


@pytest.mark.parametrize('fee', [-1, float('nan'), float('inf'), True, None])
def test_invalid_request_fee_rejects_pricing_before_paid_calls(fee):
    pricing = search_prices()
    pricing['models'][MID]['request_fee_upper_cny'] = fee
    assert runner.task_upper_cost(MID, runner.load_settings(), pricing) is None


@pytest.mark.parametrize('source', ['', 'http://example.org/pricing', 'https://', 'not-a-url'])
def test_nonzero_request_fee_requires_https_billing_source(source):
    pricing = search_prices()
    pricing['models'][MID]['request_fee_source_url'] = source
    assert runner.task_upper_cost(MID, runner.load_settings(), pricing) is None


def test_request_fee_is_reserved_for_every_possible_http_attempt():
    settings = runner.load_settings()
    base = runner.task_upper_cost(MID, settings, prices([MID]))
    priced = runner.task_upper_cost(MID, settings, search_prices())
    assert priced == pytest.approx(base + FEE * settings['retry']['max_attempts'])
    legacy = prices([MID])
    legacy['models'][MID]['request_fee_upper_cny'] = 0
    assert runner.task_upper_cost(MID, settings, legacy) == base


def test_http_success_keeps_request_fee_uncertain_after_token_settlement(demo, baseline_chat_model):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings, pricing = runner.load_settings(), search_prices()
    upper = runner.task_upper_cost(MID, settings, pricing) / settings['retry']['max_attempts']
    provider, client = attempt_adapter(MID, settings, [])

    def actual_http(**_kwargs):
        # Reservation must be durable, including the fee, before transport starts.
        import sqlite3
        with sqlite3.connect(options['db_path']) as conn:
            reserved = conn.execute("SELECT reserved_cny FROM multi_budget_calls WHERE status='in_flight'").fetchone()[0]
        assert reserved == pytest.approx(upper)
        return _resp('{"persons":[]}')

    client.chat.completions.create = actual_http
    result = runner.run_multi(dataset, **pilot_kwargs(options, pricing), max_cost_cny=upper * 1.1,
                              max_requests=1, provider_overrides={MID: provider})
    row = ledger_rows(options, result['run_id'])[0]
    token_charge = (10 * 2 + 5 * 3) / 1_000_000
    assert row['accounting_unit'] == 'http_attempt'
    assert row['known_cny'] == pytest.approx(token_charge)
    assert row['uncertain_cny'] == pytest.approx(FEE)
    assert row['committed_cny'] == pytest.approx(token_charge + FEE)
    assert result['budget']['uncertain_cny'] == pytest.approx(FEE)


def test_failed_retry_retains_full_attempt_bound_including_fee(demo, baseline_chat_model):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings, pricing = runner.load_settings(), search_prices()
    upper = runner.task_upper_cost(MID, settings, pricing) / settings['retry']['max_attempts']
    provider, client = attempt_adapter(MID, settings, [_StatusErr(503), _resp('{"persons":[]}')])
    result = runner.run_multi(dataset, **pilot_kwargs(options, pricing), max_cost_cny=upper * 2.1,
                              max_requests=1, provider_overrides={MID: provider})
    rows = ledger_rows(options, result['run_id'])
    assert len(client.chat.completions.calls) == len(rows) == 2
    assert rows[0]['known_cny'] == 0 and rows[0]['uncertain_cny'] == pytest.approx(upper)
    assert rows[1]['uncertain_cny'] == pytest.approx(FEE)
    assert result['budget']['uncertain_cny'] == pytest.approx(upper + FEE)


def test_fallback_success_keeps_previous_attempts_and_final_unbilled_fee(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings, pricing = runner.load_settings(), search_prices()
    upper = runner.task_upper_cost(MID, settings, pricing)
    usage = {'prompt_tokens':100, 'completion_tokens':20, 'total_tokens':120,
             'intentx_tools':[{'type':'web_search'}], 'plugins':{'search':{'count':1}}}
    probe = UsageProbe(usage=usage, attempts=2)
    result = runner.run_multi(dataset, **pilot_kwargs(options, pricing), max_cost_cny=upper * 1.1,
                              max_requests=1, provider_overrides={MID: probe})
    row = ledger_rows(options, result['run_id'])[0]
    token_charge = (100 * 2 + 20 * 3) / 1_000_000
    assert row['accounting_unit'] == 'task'
    assert row['known_cny'] == pytest.approx(token_charge)
    assert row['uncertain_cny'] == pytest.approx(upper / settings['retry']['max_attempts'] + FEE)
    assert row['committed_cny'] == pytest.approx(token_charge + row['uncertain_cny'])


@pytest.mark.parametrize('kind', ['unknown', 'duplicate', 'empty', 'mixed_limit', 'mixed_per_work'])
def test_invalid_sample_selector_never_reaches_paid_provider(demo, kind):
    dataset, options = demo
    manual_test_fixture(dataset)
    sid = load_dataset(dataset)[0].sample_id
    ids = ['missing-sample'] if kind == 'unknown' else [sid,sid] if kind == 'duplicate' else [] if kind == 'empty' else [sid]
    kwargs = pilot_kwargs(options)
    if kind == 'mixed_limit':
        kwargs['limit'] = 1
    if kind == 'mixed_per_work':
        kwargs['per_work'] = 1
    probe = UsageProbe()
    with pytest.raises(ValueError, match='sample_ids'):
        runner.run_multi(dataset, **kwargs, sample_ids=ids, max_cost_cny=100,
                         provider_overrides={MID:probe})
    assert not probe.calls


def test_subset_still_requires_confirmation_of_the_full_dataset(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    sid = load_dataset(dataset)[0].sample_id
    changed = json.loads(dataset.read_text())
    changed['samples'][-1]['note'] += ' unconfirmed change outside selected scope'
    dataset.write_text(json.dumps(changed), encoding='utf-8')
    probe = UsageProbe()
    with pytest.raises(ValueError, match='标注已变化'):
        runner.run_multi(dataset, **pilot_kwargs(options), sample_ids=[sid], max_cost_cny=100,
                         provider_overrides={MID:probe})
    assert not probe.calls


def test_explicit_scope_is_frozen_and_reused_on_resume_without_duplicate_calls(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    source = load_dataset(dataset)
    selected = [source[3].sample_id, source[0].sample_id]
    probe = UsageProbe()
    kwargs = pilot_kwargs(options)
    first = runner.run_multi(dataset, **kwargs, sample_ids=selected, max_cost_cny=100,
                             max_requests=1, provider_overrides={MID:probe})
    run, records = read_run(options, first['run_id'])
    assert run['fairness_meta']['sample_ids'] == selected
    assert run['fairness_meta']['sample_id_selection'] == selected
    assert [s.sample_id for s in load_dataset(run['dataset_path'])] == selected
    resumed = runner.run_multi(dataset, **kwargs, run_id=first['run_id'],
                               provider_overrides={MID:probe})
    assert resumed['planned'] == 2 and resumed['executed'] == 1
    assert len(probe.calls) == 2
    _, final = read_run(options, first['run_id'])
    assert {r['sample_id'] for r in final} == set(selected)
    assert len(final) == 2


def test_resume_rejects_a_different_explicit_subset_before_transport(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    source = load_dataset(dataset)
    selected = [source[0].sample_id, source[1].sample_id]
    probe = UsageProbe()
    kwargs = pilot_kwargs(options)
    first = runner.run_multi(dataset, **kwargs, sample_ids=selected, max_cost_cny=100,
                             max_requests=0, provider_overrides={MID:probe})
    with pytest.raises(ValueError, match='标注已变化|旧输入'):
        runner.run_multi(dataset, **kwargs, run_id=first['run_id'],
                         sample_ids=[source[0].sample_id,source[2].sample_id],
                         provider_overrides={MID:probe})
    assert not probe.calls


def tool_prices(unit=.1):
    pricing = prices([MID])
    pricing['models'][MID].update(tool_unit_cny=unit, tool_allowance_units=8,
                                  tool_usage_source_url='https://example.org/vendor-tool-pricing')
    return pricing


def tool_ledger(demo, *, maximum=350, pricing=None):
    dataset, options = demo
    manual_test_fixture(dataset)
    pricing = pricing or tool_prices()
    result = runner.run_multi(dataset, **pilot_kwargs(options, pricing), max_cost_cny=maximum,
                              max_requests=0, provider_overrides={MID: UsageProbe()})
    ledger = runner._RollingBudget(options['db_path'], result['run_id'], runner.load_settings(), pricing, maximum)
    return ledger, runner.resolve_models([MID])[0], options


def counted_result(count=None, *, usage=True, known=True):
    billing = {} if count is None else {'billing': {'units': {'search': count}, 'known_units': known}}
    return ProviderResult(ok=True, raw_text='{"persons":[]}', attempts=1, http_status=200,
                          usage={'prompt_tokens':100, 'completion_tokens':20, 'total_tokens':120} if usage else {},
                          search_metadata=billing)


@pytest.mark.parametrize('field,value', [
    ('tool_unit_cny', -1), ('tool_unit_cny', True), ('tool_unit_cny', float('nan')),
    ('tool_allowance_units', 0), ('tool_allowance_units', True),
    ('tool_usage_source_url', 'http://example.org/price'), ('tool_usage_source_url', ''),
])
def test_invalid_tool_prices_block_before_transport(field, value):
    pricing = tool_prices()
    pricing['models'][MID][field] = value
    assert runner.task_upper_cost(MID, runner.load_settings(), pricing) is None


def test_actual_tool_count_above_allowance_is_debited_without_per_call_breach(demo):
    ledger, model, options = tool_ledger(demo)
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    ledger.after_attempt(receipt, model, result=counted_result(40))
    row = ledger_rows(options, ledger.run_id)[0]
    assert row['known_cny'] == pytest.approx(.00026)
    assert row['uncertain_cny'] == pytest.approx(4.)
    assert row['committed_cny'] > row['reserved_cny']
    assert not ledger.breached
    assert json.loads(row['outcome_json'])['billing']['retained_units'] == 40
    # Another model round remains admissible; the advisory eight was not a cap.
    next_receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p2', 1)
    assert next_receipt != receipt
    ledger.after_attempt(next_receipt, model, result=counted_result(0))


@pytest.mark.parametrize('result,retained_units', [
    (counted_result(), 8), (counted_result(2, known=False), 8),
    (counted_result(40, usage=False), 40), (counted_result(0), 0),
])
def test_missing_tool_counts_retain_allowance_and_missing_tokens_keep_their_bound(demo, result, retained_units):
    ledger, model, options = tool_ledger(demo)
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    ledger.after_attempt(receipt, model, result=result)
    row = ledger_rows(options, ledger.run_id)[0]
    expected = retained_units * .1
    if not result.usage:
        expected += 1_048_576 * 2 / 1_000_000 + 100 * 3 / 1_000_000
    assert row['uncertain_cny'] == pytest.approx(expected)
    assert not ledger.breached


def test_model_rounds_and_external_search_each_have_individual_receipts(demo):
    ledger, model, options = tool_ledger(demo)
    image = 'data:image/jpeg;base64,YWJj'
    intermediate = ledger.before_attempt(model, image, 'tool-call-round', 1)
    ledger.after_attempt(intermediate, model, result=counted_result(0))
    search = ledger.before_search(model, image, 'vendor query', attempt_no=1)
    ledger.after_search(search, model, units=2)
    final = ledger.before_attempt(model, image, 'final-round', 1)
    ledger.after_attempt(final, model, result=counted_result(0))
    rows = ledger_rows(options, ledger.run_id)
    assert len({r['reservation_id'] for r in rows}) == 3
    assert [r['accounting_unit'] for r in rows] == ['http_attempt', 'search_http_attempt', 'http_attempt']
    assert sum(r['known_cny'] for r in rows) == pytest.approx(.00052)
    assert sum(r['uncertain_cny'] for r in rows) == pytest.approx(.2)
    assert rows[0]['status'] == 'settled_attempt_usage'  # Intermediate usage also settles.
    assert ledger.available() == pytest.approx(350 - .20052)
    run, _ = read_run(options, ledger.run_id)
    assert run['fairness_meta']['cost_reservation']['reserved_http_attempts'] == 3


def test_failed_external_search_keeps_fee_and_restart_keeps_inflight_search(demo):
    ledger, model, options = tool_ledger(demo)
    image = 'data:image/jpeg;base64,YWJj'
    failed = ledger.before_search(model, image, 'q')
    ledger.after_search(failed, model, units=0, error=TimeoutError('offline'))
    pending = ledger.before_search(model, image, 'q2')
    restarted = runner._RollingBudget(options['db_path'], ledger.run_id, ledger.settings, ledger.pricing, 350)
    rows = ledger_rows(options, ledger.run_id)
    assert rows[0]['uncertain_cny'] == pytest.approx(.1)
    assert rows[1]['reservation_id'] == pending and rows[1]['status'] == 'uncertain_after_restart'
    assert restarted.available() == pytest.approx(349.8)


def test_shared_350_scope_blocks_before_another_http_can_be_sent(demo):
    ledger, model, options = tool_ledger(demo)
    image = 'data:image/jpeg;base64,YWJj'
    receipt = ledger.before_attempt(model, image, 'p', 1)
    ledger.after_attempt(receipt, model, result=counted_result(3498))
    dataset, _ = demo
    child = runner.run_multi(dataset, **pilot_kwargs(options, ledger.pricing), max_cost_cny=350,
                             budget_parent_run_id=ledger.run_id, max_requests=0,
                             provider_overrides={MID: UsageProbe()})
    inherited = runner._RollingBudget(options['db_path'], child['run_id'], ledger.settings, ledger.pricing, 350)
    with pytest.raises(runner._BudgetPause):
        inherited.before_attempt(model, image, 'must-not-send', 1)
    assert not ledger_rows(options, child['run_id'])
    assert inherited.available() == pytest.approx(.19974)


def test_actual_tool_cost_above_shared_ceiling_is_retained_and_blocks_future_calls(demo):
    ledger, model, options = tool_ledger(demo)
    image = 'data:image/jpeg;base64,YWJj'
    receipt = ledger.before_attempt(model, image, 'p', 1)
    ledger.after_attempt(receipt, model, result=counted_result(4000))
    assert ledger.breached and ledger.available() == 0
    assert ledger_rows(options, ledger.run_id)[0]['uncertain_cny'] == pytest.approx(400)
    with pytest.raises(runner._BudgetPause):
        ledger.before_search(model, image, 'must-not-send')


def test_search_instruction_is_in_actual_request_and_frozen_for_resume(demo, monkeypatch):
    from vcrb.multi.prompts import SYSTEM_PROMPT
    from vcrb.providers.search_evidence import SEARCH_INSTRUCTION
    dataset, options = demo
    manual_test_fixture(dataset)
    resolve = runner.resolve_models
    search_model = deepcopy(resolve([MID])[0])
    search_model.params['search_enabled'] = True
    monkeypatch.setattr(runner, 'resolve_models', lambda ids=None: [
        search_model if model.id == MID else model for model in resolve(ids)])
    probe = UsageProbe()
    result = runner.run_multi(dataset, **pilot_kwargs(options, tool_prices()), max_cost_cny=350,
                              max_requests=1, provider_overrides={MID: probe})
    run, records = read_run(options, result['run_id'])
    assert records[0]['raw']['request']['system'] == SYSTEM_PROMPT + SEARCH_INSTRUCTION
    assert run['fairness_meta']['search_instruction_version'] == 'vendor-search-1'
    monkeypatch.setattr(runner, 'SEARCH_INSTRUCTION', SEARCH_INSTRUCTION + ' changed')
    with pytest.raises(ValueError, match='联网提示词已变化'):
        runner.run_multi(dataset, **pilot_kwargs(options, tool_prices()), run_id=result['run_id'],
                         provider_overrides={MID: probe})
    assert len(probe.calls) == 1


def output_allowance_prices():
    pricing = tool_prices()
    pricing['models'][MID].update(output_token_upper_bound=131_072, output_bound_evidence={
        'kind': 'provider_output_limit', 'tokens': 131_072,
        'source_url': 'https://help.aliyun.com/zh/model-studio/qwen3-8-max',
        'verified_on': '2026-11-15'})
    return pricing


def request_4096(monkeypatch):
    settings = runner.load_settings()
    settings['generation']['max_tokens'] = 4096
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))


class TruncatedNativeProbe:
    supports_attempt_observer = True
    name = 'offline-native-agent'

    def recognize(self, model, image, system, prompt, gen, attempt_observer):
        self.request_limit = gen.max_tokens
        receipt = attempt_observer.before_attempt(model, image, prompt, 1)
        result = counted_result(3)
        result.usage = {'prompt_tokens':100, 'completion_tokens':4295, 'total_tokens':4395}
        result.raw_text = '{"persons":['
        result.finish_reason = 'length'
        result.output_issue = 'truncated_output'
        attempt_observer.after_attempt(receipt, model, result=result)
        return result


def test_native_output_accounting_allowance_keeps_wire_limit_and_truncation_failure(demo, monkeypatch):
    request_4096(monkeypatch)
    dataset, options = demo
    manual_test_fixture(dataset)
    probe = TruncatedNativeProbe()
    result = runner.run_multi(dataset, **pilot_kwargs(options, output_allowance_prices()), max_cost_cny=350,
                              max_requests=1, provider_overrides={MID: probe})
    run, records = read_run(options, result['run_id'])
    assert probe.request_limit == 4096
    assert records[0]['raw']['response']['output_issue'] == 'truncated_output'
    assert records[0]['raw']['response']['raw_text'] == '{"persons":['
    assert not result['budget']['budget_breached']
    allowance = run['fairness_meta']['output_accounting_allowances'][MID]
    assert allowance['request_max_tokens'] == 4096 and allowance['accounting_allowance_tokens'] == 131_072
    assert ledger_rows(options, result['run_id'])[0]['known_cny'] == pytest.approx(.013085)


@pytest.mark.parametrize('change', ['too_small', 'bool', 'missing_source', 'http_source', 'unverified', 'wrong_kind', 'mismatched_tokens'])
def test_invalid_output_accounting_evidence_blocks_admission(change, monkeypatch):
    request_4096(monkeypatch)
    pricing = output_allowance_prices()
    rate = pricing['models'][MID]
    if change == 'too_small': rate['output_token_upper_bound'] = 4095
    elif change == 'bool': rate['output_token_upper_bound'] = True
    elif change == 'missing_source': rate['output_bound_evidence'].pop('source_url')
    elif change == 'http_source': rate['output_bound_evidence']['source_url'] = 'http://example.org'
    elif change == 'unverified': rate['output_bound_evidence'].pop('verified_on')
    elif change == 'wrong_kind': rate['output_bound_evidence']['kind'] = 'observed_average'
    else: rate['output_bound_evidence']['tokens'] = 4096
    assert runner.task_upper_cost(MID, runner.load_settings(), pricing) is None


def archived_output_mismatch(demo, monkeypatch):
    request_4096(monkeypatch)
    dataset, options = demo
    manual_test_fixture(dataset)
    result = runner.run_multi(dataset, **pilot_kwargs(options, tool_prices()), max_cost_cny=350,
                              max_requests=1, provider_overrides={MID: TruncatedNativeProbe()})
    # Recreate the historical policy's false-positive artifact in this temporary
    # fixture. The current policy must neither reproduce nor erase that history.
    with sqlite3.connect(options['db_path']) as conn:
        meta = json.loads(conn.execute('select fairness_meta from runs where run_id=?',(result['run_id'],)).fetchone()[0])
        meta['cost_reservation']['budget_breached'] = True
        meta['cost_reservation']['policy_version'] = 4
        conn.execute('update runs set fairness_meta=? where run_id=?',(json.dumps(meta),result['run_id']))
        conn.execute("update multi_budget_calls set status='bound_breached',note='Offline historical reply-limit accounting mismatch fixture' where run_id=?",(result['run_id'],))
    row = ledger_rows(options, result['run_id'])[0]
    assert row['status'] == 'bound_breached'
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    review = {'state':'validated_accounting_mismatch', 'reservation_id':row['reservation_id'],
              'previous_output_limit':4096, 'accounting_output_allowance':131_072,
              'source_url':'https://help.aliyun.com/zh/model-studio/qwen3-8-max',
              'reason':'Offline fixture proves reply-limit versus billed-output mismatch',
              'reviewed_at':'2026-11-15T14:00:00+00:00'}
    gate['budget_accounting_reviews'] = {result['run_id']:review}
    return result['run_id'], row, gate


def test_exact_review_allows_new_cohort_without_erasing_old_breach_or_charges(demo, monkeypatch):
    archived, old_row, gate = archived_output_mismatch(demo, monkeypatch)
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    dataset, options = demo
    new = runner.run_multi(dataset, **pilot_kwargs(options, output_allowance_prices()), max_cost_cny=350,
                           max_requests=1, budget_parent_run_id=archived,
                           provider_overrides={MID: TruncatedNativeProbe()})
    assert not new['budget']['budget_breached'] and new['executed'] == 1
    old, _ = read_run(options, archived)
    assert old['fairness_meta']['cost_reservation']['budget_breached'] is True
    assert ledger_rows(options, archived)[0] == old_row
    new_run, _ = read_run(options, new['run_id'])
    assert new_run['fairness_meta']['budget_accounting_reviews'] == gate['budget_accounting_reviews']
    assert new['budget']['reserved_cny'] == pytest.approx(old_row['committed_cny'] * 2)


@pytest.mark.parametrize('tamper', ['reservation', 'source', 'second_breach', 'monetary_breach'])
def test_review_cannot_suppress_other_or_unproven_historical_breaches(demo, monkeypatch, tamper):
    archived, old_row, gate = archived_output_mismatch(demo, monkeypatch)
    review = gate['budget_accounting_reviews'][archived]
    dataset, options = demo
    if tamper == 'reservation': review['reservation_id'] = 'wrong-id'
    elif tamper == 'source': review['source_url'] = 'http://help.aliyun.com/zh/model-studio/qwen3-8-max'
    elif tamper == 'second_breach':
        with sqlite3.connect(options['db_path']) as conn:
            cols = [r[1] for r in conn.execute('pragma table_info(multi_budget_calls)')]
            values = [old_row[col] for col in cols]
            values[cols.index('reservation_id')] = 'another-breached-http'
            conn.execute('insert into multi_budget_calls ('+','.join(cols)+') values ('+','.join('?' for _ in cols)+')', values)
    else:
        with sqlite3.connect(options['db_path']) as conn:
            meta = json.loads(conn.execute('select fairness_meta from runs where run_id=?',(archived,)).fetchone()[0])
            meta['cost_reservation']['reserved_cny'] = 351
            conn.execute('update runs set fairness_meta=? where run_id=?',(json.dumps(meta),archived))
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    probe = TruncatedNativeProbe()
    child = runner.run_multi(dataset, **pilot_kwargs(options, output_allowance_prices()), max_cost_cny=350,
                             max_requests=1, budget_parent_run_id=archived, provider_overrides={MID:probe})
    assert child['executed'] == 0 and child['budget']['budget_breached']
    assert not hasattr(probe, 'request_limit')


def test_run_ready_callback_sees_durable_run_before_any_provider_attempt(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    probe, seen = UsageProbe(), []
    def ready(rid):
        run, records = read_run(options, rid)
        assert run['run_id'] == rid and not records and not probe.calls
        assert Path(run['dataset_path']).is_file()
        seen.append(rid)
    result = runner.run_multi(dataset, **pilot_kwargs(options, tool_prices()), max_cost_cny=350,
                              max_requests=1, provider_overrides={MID: probe}, on_run_ready=ready)
    assert seen == [result['run_id']] and len(probe.calls) == 1


def test_failed_run_ready_callback_prevents_first_provider_attempt(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    probe = UsageProbe()
    def cannot_save_state(_rid):
        raise OSError('offline simulated state-write failure')
    with pytest.raises(OSError, match='state-write failure'):
        runner.run_multi(dataset, **pilot_kwargs(options, tool_prices()), max_cost_cny=350,
                         max_requests=1, provider_overrides={MID: probe}, on_run_ready=cannot_save_state)
    assert not probe.calls


def test_provider_hold_blocks_vendor_live_and_survives_resume_then_explicit_clear(demo):
    ledger, model, options = tool_ledger(demo)
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    gate['provider_holds'] = {'dashscope': {'reason':'Arrearage', 'recorded_at':'2026-11-15T14:00:00Z'}}
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    assert ledger.is_blocked(MID) and ledger.is_blocked('qwen3.8-flash')
    assert not ledger.is_blocked('deepseek-flash')
    with pytest.raises(runner._BudgetPause):
        ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'held', 1)
    assert not ledger_rows(options, ledger.run_id)
    resumed = runner._RollingBudget(options['db_path'], ledger.run_id, ledger.settings, ledger.pricing, 350)
    assert resumed.is_blocked(MID)
    run, _ = read_run(options, ledger.run_id)
    assert run['fairness_meta']['provider_holds'] == gate['provider_holds']
    gate.pop('provider_holds')  # Operator records the confirmed top-up separately.
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    cleared = runner._RollingBudget(options['db_path'], ledger.run_id, ledger.settings, ledger.pricing, 350)
    assert not cleared.is_blocked(MID)
    run, _ = read_run(options, ledger.run_id)
    assert run['fairness_meta']['provider_holds'] == {}


def test_held_vendor_never_reaches_provider_while_other_vendor_continues(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    gate['provider_holds'] = {'dashscope': {'reason':'Arrearage'}}
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    mids = [MID, 'deepseek-flash']
    probes = {mid: UsageProbe() for mid in mids}
    result = runner.run_multi(dataset, **options, mock=False, yes=True, model_ids=mids,
                              experiments=('C',), pricing=prices(mids), max_cost_cny=350,
                              approval_note='OFFLINE TEST ONLY', limit=1, provider_overrides=probes)
    assert not probes[MID].calls and len(probes['deepseek-flash'].calls) == 1
    assert result['budget']['blocked_providers'] == ['dashscope']


def test_zero_request_recovery_settles_orphan_reservation_without_provider_call(demo):
    ledger, model, options = tool_ledger(demo)
    reservation = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'orphan', 1)
    probe = UsageProbe()
    dataset, _ = demo
    recovered = runner.run_multi(dataset, **pilot_kwargs(options, ledger.pricing),
                                 run_id=ledger.run_id, max_requests=0, provider_overrides={MID:probe})
    assert not probe.calls and recovered['executed'] == 0
    row = ledger_rows(options, ledger.run_id)[0]
    assert row['reservation_id'] == reservation and row['status'] == 'uncertain_after_restart'
    assert row['uncertain_cny'] == row['committed_cny'] == row['reserved_cny']
    assert recovered['budget']['inflight_cny'] == 0


@pytest.mark.parametrize('prompt_tokens,completion_tokens', [(1_100_000,20),(100,140_000),(1_100_000,140_000)])
def test_proven_native_agent_aggregate_over_allowance_debits_actual_and_continues(
        demo, prompt_tokens, completion_tokens):
    ledger, model, options = tool_ledger(demo, pricing=output_allowance_prices())
    model = deepcopy(model)
    model.params['api_mode'] = 'dashscope_anthropic'
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    result = counted_result(3)
    result.usage = {'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,
                    'total_tokens':prompt_tokens+completion_tokens}
    ledger.after_attempt(receipt, model, result=result)
    row = ledger_rows(options, ledger.run_id)[0]
    assert row['status'] == 'native_metering_over_allowance' and not ledger.breached
    assert row['known_cny'] == pytest.approx((prompt_tokens*2+completion_tokens*3)/1_000_000)
    assert row['uncertain_cny'] == pytest.approx(.3)
    warning = json.loads(row['outcome_json'])['metering_warning']
    assert warning['prompt_tokens'] == prompt_tokens and warning['completion_tokens'] == completion_tokens
    assert warning['input_accounting_allowance'] == 1_048_576
    again = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'next', 1)
    ledger.after_attempt(again, model, result=counted_result(0))


@pytest.mark.parametrize('mode,proof', [
    ('chat_completions','count'),('dashscope_multimodal_search','count'),
    ('deepseek_native_search','count'),('dashscope_anthropic','zero'),
    ('dashscope_anthropic','sources_only'),('dashscope_anthropic','unmatched_blocks'),
])
def test_non_agent_or_unproven_native_usage_still_enforces_token_bounds(demo, mode, proof):
    ledger, model, options = tool_ledger(demo)
    model = deepcopy(model)
    model.params['api_mode'] = mode
    result = counted_result(3 if proof == 'count' else 0)
    result.usage = {'prompt_tokens':1_100_000,'completion_tokens':20,'total_tokens':1_100_020}
    if proof == 'sources_only':
        result.search_metadata = {'requested':True,'retrieved':True,'sources':[{'url':'https://example.org'}]}
    elif proof == 'unmatched_blocks':
        result.search_metadata = {'native':{'content':[
            {'type':'server_tool_use','id':'actual-call','name':'web_search'},
            {'type':'web_search_tool_result','tool_use_id':'wrong-call','content':[]}]}}
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    ledger.after_attempt(receipt, model, result=result)
    assert ledger.breached and ledger_rows(options, ledger.run_id)[0]['status'] == 'bound_breached'


def test_matched_native_server_blocks_prove_aggregate_but_do_not_invent_billing_units(demo):
    ledger, model, options = tool_ledger(demo)
    model = deepcopy(model)
    model.params['api_mode'] = 'dashscope_anthropic'
    result = counted_result()
    result.usage = {'prompt_tokens':1_100_000,'completion_tokens':20,'total_tokens':1_100_020}
    result.search_metadata['native'] = {'content':[
        {'type':'server_tool_use','id':'actual-call','name':'web_search'},
        {'type':'web_search_tool_result','tool_use_id':'actual-call','content':[]}]}
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    ledger.after_attempt(receipt, model, result=result)
    row = ledger_rows(options, ledger.run_id)[0]
    assert not ledger.breached and row['status'] == 'native_metering_over_allowance'
    assert row['uncertain_cny'] == pytest.approx(.8)  # Unknown native billable count keeps eight.
    assert json.loads(row['outcome_json'])['billing']['known_units'] is False


def test_native_aggregate_cost_above_shared_ceiling_still_stops_calls(demo):
    ledger, model, options = tool_ledger(demo)
    model = deepcopy(model)
    model.params['api_mode'] = 'dashscope_anthropic'
    result = counted_result(3)
    result.usage = {'prompt_tokens':200_000_000,'completion_tokens':20,'total_tokens':200_000_020}
    receipt = ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'p', 1)
    ledger.after_attempt(receipt, model, result=result)
    row = ledger_rows(options, ledger.run_id)[0]
    assert row['known_cny'] == pytest.approx(400.00006)
    assert ledger.breached and ledger.available() == 0
    with pytest.raises(runner._BudgetPause):
        ledger.before_attempt(model, 'data:image/jpeg;base64,YWJj', 'must-not-send', 1)
