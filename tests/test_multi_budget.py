"""Rolling budget settlement tests using injected providers and temporary DBs."""
from copy import deepcopy
import json
import sqlite3
import threading
import time

import pytest

from tests.test_multi_runner import (
    demo, offline_only, manual_test_fixture, prices, paid_kwargs, read_run,
    inject_provider, OfflineProbeProvider, approve_offline_fixture,
)
from vcrb.multi import runner
from vcrb.providers.base import ProviderResult
from vcrb.providers.openai_compat import OpenAICompatProvider
from vcrb.providers.errors import classify_exception
from vcrb.models import load_registry
from tests.test_providers import FakeClient, _resp, _StatusErr


class UsageProbe(OfflineProbeProvider):
    def __init__(self, usage=None, attempts=1, hook=None):
        super().__init__(hook)
        self.usage = usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        self.attempts = attempts

    def recognize(self, model, image_data_url, system_prompt, user_prompt, gen):
        super().recognize(model, image_data_url, system_prompt, user_prompt, gen)
        return ProviderResult(ok=True, raw_text='{"persons":[]}', served_model=model.id + '#OFFLINE-TEST',
                              usage=deepcopy(self.usage), attempts=self.attempts, http_status=200)


def ledger_rows(options, run_id):
    with sqlite3.connect(options["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute('select * from multi_budget_calls where run_id=? order by rowid',(run_id,))]


def test_success_usage_releases_retry_and_context_padding_for_rolling_calls(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(mid, runner.load_settings(), prices([mid]))
    probe = UsageProbe()
    inject_provider(monkeypatch, probe)
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 1.1)
    assert result['planned'] == result['executed'] == 6  # Six calls, budget fits only one worst-case call.
    run, raw = read_run(options, result['run_id'])
    cost = run['fairness_meta']['cost_reservation']
    actual = (100 * 2 + 20 * 3) / 1_000_000
    assert cost['reserved_tasks'] == 6 and cost['known_usage_upper_cny'] == pytest.approx(actual * 6)
    assert cost['reserved_cny'] == pytest.approx(actual * 6) and cost['uncertain_cny'] == 0
    assert cost['inflight_cny'] == 0 and cost['released_cny'] == pytest.approx(unit * 6 - actual * 6)
    assert all(r['raw']['response']['usage'] == probe.usage for r in raw)
    assert all(r['status'] == 'settled_usage' for r in ledger_rows(options, result['run_id']))


def test_parallel_models_reserve_their_own_prices_and_drain_then_resume(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings = runner.load_settings()
    settings['runner'] = {'concurrency': 8}
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))
    mids = runner.MODEL_IDS[:2]
    pricing = prices(mids)
    pricing['models'][mids[1]].update(input_cny_per_million=8, output_cny_per_million=12)
    units = {m: runner.task_upper_cost(m, settings, pricing) for m in mids}
    maximum = max(units.values()) * 1.1  # Cannot reserve one task from both models together.
    def check_budget(*_):
        with sqlite3.connect(options['db_path']) as conn:
            total = conn.execute('select sum(committed_cny) from multi_budget_calls').fetchone()[0]
        assert total <= maximum + 1e-9
    probes = {mid: UsageProbe(hook=check_budget) for mid in mids}
    result = runner.run_multi(dataset, **options, model_ids=mids, mock=False, yes=True,
                             approval_note='OFFLINE TEST ONLY', pricing=pricing, max_cost_cny=maximum,
                             limit=2, provider_overrides=probes)
    assert result['executed'] == 12 and result['progress'] == {'total': 12, 'ok': 12, 'failed': 0}
    rows = ledger_rows(options, result['run_id'])
    assert len(rows) == 12
    for row in rows:
        assert row['reserved_cny'] == pytest.approx(units[row['model_id']])
        assert row['status'] == 'settled_usage'
    run, _ = read_run(options, result['run_id'])
    assert run['status'] == 'done' and run['fairness_meta']['cost_reservation']['inflight_cny'] == 0


def test_successful_retry_keeps_prior_attempt_cost_but_releases_unused_attempts(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings = runner.load_settings()
    settings['retry']['max_attempts'] = 4
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))
    mid = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(mid, settings, prices([mid]))
    probe = UsageProbe(attempts=2)
    inject_provider(monkeypatch, probe)
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 2, max_requests=1)
    row = ledger_rows(options, result['run_id'])[0]
    final = (100 * 2 + 20 * 3) / 1_000_000
    assert row['known_cny'] == pytest.approx(final)
    assert row['uncertain_cny'] == pytest.approx(unit / 4)
    assert row['committed_cny'] == pytest.approx(final + unit / 4)
    assert row['attempts'] == 2 and row['status'] == 'settled_usage'


@pytest.mark.parametrize('usage', [
    {}, {'prompt_tokens': 100}, {'prompt_tokens': True, 'completion_tokens': 20},
    {'prompt_tokens': 100, 'completion_tokens': -1},
    {'prompt_tokens': 0, 'completion_tokens': 20},
    {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 200},
])
def test_missing_or_unreliable_usage_never_frees_budget(demo, monkeypatch, usage):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(mid, runner.load_settings(), prices([mid]))
    probe = UsageProbe(usage)
    inject_provider(monkeypatch, probe)
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 1.1)
    assert result['executed'] == 1
    row = ledger_rows(options, result['run_id'])[0]
    assert row['committed_cny'] == row['uncertain_cny'] == pytest.approx(unit)
    assert row['known_cny'] == 0


def test_bound_violation_is_accounted_and_remains_blocked_on_resume(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(mid, runner.load_settings(), prices([mid]))
    usage = {'prompt_tokens': 1_048_700, 'completion_tokens': 20, 'total_tokens': 1_048_720}
    probe = UsageProbe(usage)
    inject_provider(monkeypatch, probe)
    kwargs = paid_kwargs(options, mid)
    result = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 20)
    assert result['executed'] == 1
    run, _ = read_run(options, result['run_id'])
    assert run['fairness_meta']['cost_reservation']['budget_breached'] is True
    assert ledger_rows(options, result['run_id'])[0]['status'] == 'bound_breached'
    resumed = runner.run_multi(dataset, **kwargs, run_id=result['run_id'])
    assert resumed['executed'] == 0 and len(probe.calls) == 1
    run, _ = read_run(options, result['run_id'])
    assert run['fairness_meta']['cost_reservation']['budget_breached'] is True


def test_restart_keeps_inflight_reservation_uncertain(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    # Initialize an authorized zero-call run with no provider construction/network.
    result = runner.run_multi(dataset, **paid_kwargs(options, runner.MODEL_IDS[0]), max_cost_cny=100,
                             max_requests=0, provider_overrides={runner.MODEL_IDS[0]: UsageProbe()})
    settings = runner.load_settings()
    ledger = runner._RollingBudget(options['db_path'], result['run_id'], settings, prices(), 100)
    reservation = ledger.reserve(runner.MODEL_IDS[0], 'data:image/jpeg;base64,YWJj', 'p')
    before = ledger_rows(options, result['run_id'])[0]
    assert before['status'] == 'in_flight'
    restarted = runner._RollingBudget(options['db_path'], result['run_id'], settings, prices(), 100)
    row = ledger_rows(options, result['run_id'])[0]
    assert row['reservation_id'] == reservation and row['status'] == 'uncertain_after_restart'
    assert row['uncertain_cny'] == row['committed_cny'] == before['reserved_cny']
    assert restarted.available() == pytest.approx(100 - row['reserved_cny'])


def test_smaller_context_bound_requires_verified_provider_evidence():
    mid = 'glm-4.6v'
    pricing = prices([mid])
    rate = pricing['models'][mid]
    rate['input_token_upper_bound'] = 131_072
    assert runner.task_upper_cost(mid, runner.load_settings(), pricing) is None
    rate['input_bound_evidence'] = {'kind':'historical_p95', 'context_tokens':131_072,
        'source_url':'https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v', 'verified_on':'2026-11-13'}
    assert runner.task_upper_cost(mid, runner.load_settings(), pricing) is None
    rate['input_bound_evidence']['kind'] = 'provider_context_limit'
    assert runner.task_upper_cost(mid, runner.load_settings(), pricing) is not None
    rate['input_bound_evidence']['source_url'] = 'https://example.org/not-provider-evidence'
    assert runner.task_upper_cost(mid, runner.load_settings(), pricing) is None
    rate['input_bound_evidence']['source_url'] = 'https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v'
    rate['input_token_upper_bound'] = 4096
    assert runner.task_upper_cost(mid, runner.load_settings(), pricing) is None


def test_retry_failed_is_not_repeated_again_during_internal_budget_pauses(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    settings = runner.load_settings()
    settings['runner'] = {'concurrency':8}
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))
    mids = runner.MODEL_IDS[:2]
    unit = runner.task_upper_cost(mids[0], settings, prices(mids))
    class FailOnce(UsageProbe):
        def recognize(self, *args):
            result = super().recognize(*args)
            if len(self.calls) == 1:
                return ProviderResult(ok=False, attempts=2, error_type='timeout', error_message='offline-only timeout')
            return result
    probe = FailOnce()
    overrides = {mid:probe for mid in mids}
    kwargs = {**options, 'mock':False, 'yes':True, 'approval_note':'OFFLINE TEST ONLY',
              'model_ids':mids, 'pricing':prices(mids), 'max_cost_cny':unit * 2.1,
              'limit':2, 'provider_overrides':overrides, 'retry_failed':True}
    result = runner.run_multi(dataset, **kwargs)
    assert result['executed'] == 12 and len(probe.calls) == 12
    assert result['progress'] == {'total':12, 'ok':11, 'failed':1}
    retried = runner.run_multi(dataset, **kwargs, run_id=result['run_id'])
    assert retried['executed'] == 1 and len(probe.calls) == 13
    assert retried['progress'] == {'total':12, 'ok':12, 'failed':0}


def test_first_168_requests_cover_balanced_pilot_despite_budget_pauses(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    payload = json.loads(dataset.read_text())
    extended = []
    for sample in payload['samples']:
        duplicate = deepcopy(sample)
        duplicate['sample_id'] += '_extra'
        extended.extend([sample, duplicate])
    payload['samples'] = extended
    dataset.write_text(json.dumps(payload, ensure_ascii=False))
    approve_offline_fixture(dataset)
    samples = runner.load_dataset(dataset)
    ordered = runner.pilot_first_samples(samples)
    pilot = {s.sample_id for s in ordered[:8]}
    assert len(pilot) == 8 and pilot != {s.sample_id for s in samples[:8]}
    settings = runner.load_settings()
    settings['runner'] = {'concurrency':8}
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))
    pricing = prices(runner.MODEL_IDS)
    unit = runner.task_upper_cost(runner.MODEL_IDS[0], settings, pricing)
    probe = UsageProbe()
    overrides = {mid:probe for mid in runner.MODEL_IDS}
    result = runner.run_multi(dataset, **options, mock=False, yes=True,
        approval_note='OFFLINE TEST ONLY', pricing=pricing, max_cost_cny=unit * 1.1,
        max_requests=168, provider_overrides=overrides)
    assert result['planned'] == 336 and result['executed'] == 168
    run, records = read_run(options, result['run_id'])
    assert {r['sample_id'] for r in records} == pilot
    assert {(r['model_id'], r['experiment'], r['sample_id']) for r in records} == {
        (mid, exp, sid) for mid in runner.MODEL_IDS for exp in ('A','B','C') for sid in pilot}
    frozen = runner.load_dataset(__import__('pathlib').Path(run['dataset_path']))
    assert [s.sample_id for s in frozen] == [s.sample_id for s in samples]
    assert runner.dataset_hash(frozen) == runner.dataset_hash(samples)
    resumed = runner.run_multi(dataset, **options, run_id=result['run_id'], mock=False, yes=True,
        provider_overrides=overrides, max_requests=168)
    assert resumed['executed'] == 168 and resumed['progress']['total'] == 336
    assert len(probe.calls) == 336  # Pilot responses are retained, never paid twice.


def test_provider_cannot_hide_attempts_above_reserved_retry_limit(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(mid, runner.load_settings(), prices([mid]))
    probe = UsageProbe(attempts=3)  # Configuration permits only two attempts.
    inject_provider(monkeypatch, probe)
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 20)
    assert result['executed'] == 1
    row = ledger_rows(options, result['run_id'])[0]
    assert row['status'] == 'bound_breached'
    assert row['committed_cny'] == row['uncertain_cny'] == pytest.approx(unit * 1.5)


def attempt_adapter(mid, settings, behaviors):
    client = FakeClient(behaviors)
    registry = load_registry()
    provider = OpenAICompatProvider(registry.providers[registry.get(mid).provider],
        settings=settings, client_factory=lambda: client, sleeper=lambda _seconds: None)
    return provider, client


def test_actual_http_attempt_reservation_can_run_below_full_retry_bound(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid])) / settings['retry']['max_attempts']
    provider, client = attempt_adapter(mid, settings, [_resp('{"persons":[]}') for _ in range(6)])
    def assert_durable_before_http(**_kwargs):
        with sqlite3.connect(options['db_path']) as conn:
            active, total = conn.execute("select count(*),sum(committed_cny) from multi_budget_calls where status='in_flight'").fetchone()
        assert active == 1 and total == pytest.approx(unit)
        return _resp('{"persons":[]}')
    client.chat.completions.create = assert_durable_before_http
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 1.1,
                             provider_overrides={mid:provider})
    assert result['executed'] == 6 and result['progress']['ok'] == 6
    rows = ledger_rows(options, result['run_id'])
    assert len(rows) == 6
    assert all(r['accounting_unit'] == 'http_attempt' and r['attempt_no'] == 1 for r in rows)
    assert all(r['reserved_cny'] == pytest.approx(unit) for r in rows)
    assert all(r['status'] == 'settled_attempt_usage' and r['uncertain_cny'] == 0 for r in rows)
    assert result['budget']['reserved_http_attempts'] == 6


def test_http_retries_reserve_only_attempts_actually_sent_and_audit_failure(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    settings['retry']['max_attempts'] = 4
    # Freeze the same retry cap for runner and adapter.
    from unittest.mock import patch
    unit = runner.task_upper_cost(mid, settings, prices([mid])) / 4
    provider, client = attempt_adapter(mid, settings, [_StatusErr(503), _resp('{"persons":[]}')])
    with patch.object(runner, 'load_settings', lambda: deepcopy(settings)):
        result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 2.1,
                                 max_requests=1, provider_overrides={mid:provider})
    assert result['executed'] == 1 and len(client.chat.completions.calls) == 2
    rows = ledger_rows(options, result['run_id'])
    assert [r['attempt_no'] for r in rows] == [1,2]
    assert rows[0]['committed_cny'] == rows[0]['uncertain_cny'] == pytest.approx(unit)
    assert json.loads(rows[0]['outcome_json']) == {'ok':False,'http_status':503,'error_type':'server_error','interrupted':False}
    assert rows[1]['known_cny'] > 0 and rows[1]['uncertain_cny'] == 0
    assert result['budget']['reserved_tasks'] == 1
    assert result['budget']['reserved_total_cny'] == pytest.approx(2 * unit)


def test_no_budget_for_next_retry_sends_no_http_and_keeps_first_attempt(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid])) / settings['retry']['max_attempts']
    provider, client = attempt_adapter(mid, settings, [_StatusErr(503), _resp('must not be sent')])
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 1.1,
                             max_requests=1, provider_overrides={mid:provider})
    assert len(client.chat.completions.calls) == 1
    assert result['executed'] == 0 and result['progress']['total'] == 0
    rows = ledger_rows(options, result['run_id'])
    assert len(rows) == 1 and rows[0]['uncertain_cny'] == pytest.approx(unit)
    assert result['budget']['reserved_total_cny'] == pytest.approx(unit)
    assert not list((options['output_root'] / 'raw').rglob('*.jsonl'))


@pytest.mark.parametrize('code,message,expected', [
    ('1113', 'official balance error', ('insufficient_balance', False)),
    ('1302', 'rate limit', ('rate_limit', True)),
    ('1305', 'model capacity', ('rate_limit', True)),
    (None, '余额不足，请充值后重试', ('insufficient_balance', False)),
])
def test_balance_429_is_distinct_from_rate_limit_without_assuming_free(code, message, expected):
    error = _StatusErr(429)
    error.args = (message,)
    error.body = {'error': {'code':code, 'message':message}}
    assert classify_exception(error) == expected


def test_balance_error_stops_retry_and_other_tasks_for_same_provider(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = 'glm-4.6v', runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid])) / settings['retry']['max_attempts']
    error = _StatusErr(429)
    error.body = {'error':{'code':'1113','message':'余额不足'}}
    provider, client = attempt_adapter(mid, settings, [error, _resp('must not be sent')])
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 20,
                             provider_overrides={mid:provider})
    assert len(client.chat.completions.calls) == result['executed'] == 1
    assert result['progress'] == {'total':1,'ok':0,'failed':1}
    assert result['budget']['blocked_providers'] == ['zhipu']
    assert result['budget']['uncertain_cny'] == pytest.approx(unit)
    row = ledger_rows(options, result['run_id'])[0]
    assert row['attempt_no'] == 1 and json.loads(row['outcome_json'])['error_type'] == 'insufficient_balance'


def test_observer_failures_propagate_and_never_trigger_api_retry():
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    provider, client = attempt_adapter(mid, settings, [_resp('unused')])
    class RefuseAdmission:
        def before_attempt(self, *_args):
            raise RuntimeError('offline reservation refused')
    from vcrb.providers.base import GenerationParams
    with pytest.raises(RuntimeError, match='reservation refused'):
        provider.recognize(load_registry().get(mid), 'data:image/jpeg;base64,YWJj', 's', 'u',
                           GenerationParams(), attempt_observer=RefuseAdmission())
    assert not client.chat.completions.calls
    class RefuseSettlement:
        def before_attempt(self, *_args):
            return 'receipt'
        def after_attempt(self, *_args, **_kwargs):
            raise RuntimeError('offline ledger settlement failed')
    with pytest.raises(RuntimeError, match='settlement failed'):
        provider.recognize(load_registry().get(mid), 'data:image/jpeg;base64,YWJj', 's', 'u',
                           GenerationParams(), attempt_observer=RefuseSettlement())
    assert len(client.chat.completions.calls) == 1


def test_revision_inherits_prior_unknown_cost_and_zero_call_resume_does_not_duplicate_it(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid]))
    provider = UsageProbe(usage={})
    kwargs = paid_kwargs(options, mid)
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 1.5, max_requests=1,
                            provider_overrides={mid:provider})
    old_run, old_records = read_run(options, first['run_id'])
    payload = json.loads(dataset.read_text())
    payload['samples'][0]['note'] += '; revised offline annotation'
    dataset.write_text(json.dumps(payload, ensure_ascii=False))
    approve_offline_fixture(dataset)
    with pytest.raises(ValueError, match='继承预算|归零'):
        runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 1.5, max_requests=0,
                         provider_overrides={mid:provider})
    with pytest.raises(ValueError, match='原预算上限'):
        runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 20, max_requests=0,
                         budget_parent_run_id=first['run_id'], provider_overrides={mid:provider})
    child = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 1.5,
        budget_parent_run_id=first['run_id'], provider_overrides={mid:provider})
    assert child['executed'] == 0 and len(provider.calls) == 1
    assert child['budget']['reserved_cny'] == pytest.approx(unit)
    assert child['budget']['own_run_reserved_cny'] == 0
    assert set(child['budget']['scope_run_ids']) == {first['run_id'], child['run_id']}
    resumed = runner.run_multi(dataset, **kwargs, run_id=child['run_id'],
                               provider_overrides={mid:provider})
    assert resumed['executed'] == 0 and resumed['budget']['reserved_cny'] == pytest.approx(unit)
    assert not ledger_rows(options, child['run_id'])
    assert read_run(options, first['run_id'])[1] == old_records


def test_sibling_revisions_share_the_same_budget_scope(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid]))
    kwargs = paid_kwargs(options, mid)
    provider = UsageProbe(usage={})
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 2.1, max_requests=1,
                            provider_overrides={mid:provider})
    child = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 2.1, max_requests=1,
                            budget_parent_run_id=first['run_id'], provider_overrides={mid:provider})
    sibling = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 2.1, max_requests=1,
                              budget_parent_run_id=first['run_id'], provider_overrides={mid:provider})
    assert first['executed'] == child['executed'] == 1 and sibling['executed'] == 0
    assert sibling['budget']['reserved_cny'] == pytest.approx(unit * 2)
    assert len(sibling['budget']['scope_run_ids']) == 3 and len(provider.calls) == 2


@pytest.mark.parametrize('change', [
    {'state': 'pending'}, {'root_run_id': 'another-root'},
    {'previous_max_cost_cny': 99}, {'max_cost_cny': 200},
    {'user_instruction': ''}, {'confirmed_at': None},
    {'previous_max_cost_cny': True}, {'max_cost_cny': True},
])
def test_cumulative_cap_extension_rejects_missing_or_mismatched_user_decision(change):
    approval = {'state': 'confirmed_by_user', 'root_run_id': 'root',
                'previous_max_cost_cny': 100, 'max_cost_cny': 150,
                'user_instruction': 'OFFLINE TEST: cumulative cap 150',
                'confirmed_at': '2026-11-13T00:00:00Z'}
    approval.update(change)
    with pytest.raises(ValueError, match='单独明确确认'):
        runner._explicit_budget_extension({'budget_extension': approval}, 'root', 100, 150)


def test_explicit_cumulative_extension_retains_parent_unknown_cost_and_history(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid]))
    kwargs, provider = paid_kwargs(options, mid), UsageProbe(usage={})
    original_cap, extended_cap = unit * 1.5, unit * 4
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=original_cap,
                             max_requests=1, provider_overrides={mid: provider})
    parent_before = read_run(options, first['run_id'])
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    gate['budget_extension'] = {'state': 'confirmed_by_user',
        'root_run_id': first['run_id'], 'previous_max_cost_cny': original_cap,
        'max_cost_cny': extended_cap, 'user_instruction': 'OFFLINE TEST ONLY: explicitly raise cumulative cap',
        'confirmed_at': '2026-11-13T00:00:00Z'}
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    child = runner.run_multi(dataset, **kwargs, max_cost_cny=extended_cap,
        max_requests=1, budget_parent_run_id=first['run_id'], provider_overrides={mid: provider})
    assert child['executed'] == 1
    assert child['budget']['reserved_cny'] == pytest.approx(unit * 2)
    assert child['budget']['uncertain_cny'] == pytest.approx(unit * 2)
    assert read_run(options, first['run_id']) == parent_before
    child_run, _ = read_run(options, child['run_id'])
    scope = child_run['fairness_meta']['budget_scope']
    assert scope['root_run_id'] == first['run_id']
    assert scope['max_cost_cny'] == extended_cap
    assert scope['user_confirmed_extension'] == gate['budget_extension']


def test_insufficient_balance_does_not_hold_healthy_model_at_pilot_barrier(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    payload = json.loads(dataset.read_text())
    for index in range(3):
        extra = deepcopy(payload['samples'][0])
        extra['sample_id'] += f'_extra_{index}'
        payload['samples'].append(extra)
    dataset.write_text(json.dumps(payload))
    approve_offline_fixture(dataset)
    bad_mid, good_mid = 'glm-4.6v', 'deepseek-flash'
    settings = runner.load_settings()
    balance_error = _StatusErr(429)
    balance_error.body = {'error': {'code': '1113', 'message': 'insufficient balance'}}
    bad, bad_client = attempt_adapter(bad_mid, settings, [balance_error])
    expected_good = len(payload['samples']) * 3
    good, good_client = attempt_adapter(good_mid, settings,
        [_resp('{"persons":[]}') for _ in range(expected_good)])
    result = runner.run_multi(dataset, **options, mock=False, yes=True,
        model_ids=[bad_mid, good_mid], max_cost_cny=100,
        approval_note='OFFLINE TEST ONLY', pricing=prices([bad_mid, good_mid]),
        provider_overrides={bad_mid: bad, good_mid: good})
    assert len(bad_client.chat.completions.calls) == 1
    assert len(good_client.chat.completions.calls) == expected_good
    assert result['progress'] == {'total': expected_good + 1, 'ok': expected_good, 'failed': 1}
    assert result['budget']['blocked_providers'] == ['zhipu']


def test_explicit_cap_extension_resumes_current_revision_without_duplicate_requests(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, provider = runner.MODEL_IDS[0], UsageProbe(usage={})
    unit = runner.task_upper_cost(mid, runner.load_settings(), prices([mid]))
    kwargs = paid_kwargs(options, mid)
    old_cap, new_cap = unit * 1.5, unit * 4
    parent = runner.run_multi(dataset, **kwargs, max_cost_cny=old_cap,
        max_requests=1, provider_overrides={mid: provider})
    parent_before = read_run(options, parent['run_id'])
    child = runner.run_multi(dataset, **kwargs, max_cost_cny=old_cap,
        max_requests=0, budget_parent_run_id=parent['run_id'], provider_overrides={mid: provider})
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    gate['budget_extension'] = {'state': 'confirmed_by_user', 'root_run_id': parent['run_id'],
        'previous_max_cost_cny': old_cap, 'max_cost_cny': new_cap,
        'user_instruction': 'OFFLINE TEST ONLY: cumulative extension', 'confirmed_at': '2026-11-13T00:00:00Z'}
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))
    resumed = runner.run_multi(dataset, **kwargs, run_id=child['run_id'],
        max_cost_cny=new_cap, max_requests=1, provider_overrides={mid: provider})
    assert resumed['run_id'] == child['run_id'] and resumed['executed'] == 1
    assert resumed['budget']['reserved_cny'] == pytest.approx(unit * 2)
    assert read_run(options, parent['run_id']) == parent_before
    amended, records = read_run(options, child['run_id'])
    assert amended['fairness_meta']['max_cost_cny'] == new_cap
    assert len(amended['fairness_meta']['budget_amendments']) == 1 and len(records) == 1
    # Repeating the same authorized cap keeps the same history; successful
    # task keys are skipped rather than called again.
    again = runner.run_multi(dataset, **kwargs, run_id=child['run_id'],
        max_cost_cny=new_cap, max_requests=1, provider_overrides={mid: provider})
    assert again['executed'] == 1 and len(read_run(options, child['run_id'])[1]) == 2
    assert len(read_run(options, child['run_id'])[0]['fairness_meta']['budget_amendments']) == 1


def test_user_revocation_between_http_attempts_stops_without_another_request(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid]))
    provider, client = attempt_adapter(mid, settings, [])
    calls = []
    def revoke_after_http(**kwargs):
        calls.append(kwargs)
        runner.REAL_CALL_GATE.write_text(json.dumps({'state':'awaiting_user_confirmation'}))
        raise _StatusErr(503)
    client.chat.completions.create = revoke_after_http
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 10,
                             provider_overrides={mid:provider})
    assert len(calls) == 1 and result['executed'] == 0
    assert result['budget']['real_call_gate_blocked'] is True
    assert result['budget']['uncertain_cny'] == pytest.approx(unit / settings['retry']['max_attempts'])


def test_parallel_http_admission_never_exceeds_shared_cap(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    mids, settings = runner.MODEL_IDS[:2], runner.load_settings()
    settings['runner'] = {'concurrency':8}
    monkeypatch.setattr(runner, 'load_settings', lambda: deepcopy(settings))
    pricing = prices(mids)
    pricing['models'][mids[1]].update(input_cny_per_million=8, output_cny_per_million=12)
    units = {mid:runner.task_upper_cost(mid, settings, pricing) / settings['retry']['max_attempts'] for mid in mids}
    maximum = max(units.values()) * 1.1
    count, lock, providers = [], threading.Lock(), {}
    def http(**kwargs):
        with sqlite3.connect(options['db_path']) as conn:
            committed = conn.execute('select sum(committed_cny) from multi_budget_calls').fetchone()[0]
        assert committed <= maximum + 1e-9
        with lock:
            count.append(kwargs['model'])
        time.sleep(.003)
        return _resp('{"persons":[]}')
    for mid in mids:
        provider, client = attempt_adapter(mid, settings, [])
        client.chat.completions.create = http
        providers[mid] = provider
    result = runner.run_multi(dataset, **options, mock=False, yes=True,
        approval_note='OFFLINE TEST ONLY', model_ids=mids, pricing=pricing, max_cost_cny=maximum,
        limit=2, provider_overrides=providers)
    assert result['executed'] == len(count) == 12
    assert result['budget']['uncertain_cny'] == result['budget']['inflight_cny'] == 0
    assert all(row['reserved_cny'] == pytest.approx(units[row['model_id']])
               for row in ledger_rows(options, result['run_id']))


def test_interrupt_during_http_retains_exactly_one_attempt(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid, settings = runner.MODEL_IDS[0], runner.load_settings()
    unit = runner.task_upper_cost(mid, settings, prices([mid])) / settings['retry']['max_attempts']
    provider, client = attempt_adapter(mid, settings, [])
    calls = []
    def interrupt(**_kwargs):
        calls.append(True)
        raise KeyboardInterrupt()
    client.chat.completions.create = interrupt
    result = runner.run_multi(dataset, **paid_kwargs(options, mid), max_cost_cny=unit * 10,
                             provider_overrides={mid:provider})
    rows = ledger_rows(options, result['run_id'])
    assert len(calls) == len(rows) == 1 and result['executed'] == 0
    assert rows[0]['uncertain_cny'] == pytest.approx(unit)
    assert json.loads(rows[0]['outcome_json'])['interrupted'] is True
