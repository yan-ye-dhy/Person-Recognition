"""The opt-in audit composes with the durable rolling budget, offline."""
import json
import sqlite3

import pytest

from tests.test_multi_runner import demo, offline_only, manual_test_fixture, paid_kwargs, read_run
from tests.test_multi_budget import attempt_adapter, ledger_rows, UsageProbe
from tests.test_providers import _baseline_qwen, _resp
from vcrb.multi import runner


def test_run_multi_audit_keeps_budget_and_ledger_link_before_transport(demo, monkeypatch, tmp_path):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = _baseline_qwen()
    monkeypatch.setattr(runner, 'resolve_models', lambda ids=None: [model])
    provider, client = attempt_adapter(model.id, runner.load_settings(), [])
    audit_path = tmp_path / 'attempts.jsonl'
    def actual_http(**kwargs):
        events = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(events) == 1 and events[0]['event'] == 'http_attempt_start'
        with sqlite3.connect(options['db_path']) as db:
            state = db.execute('select status from multi_budget_calls where reservation_id=?',
                               (events[0]['budget_reservation_id'],)).fetchone()
        assert state == ('in_flight',)
        response = _resp('{"persons":[]}')
        response._request_id, response.id = 'sdk-header-123', 'response-456'
        return response
    client.chat.completions.create = actual_http
    result = runner.run_multi(dataset, **paid_kwargs(options, model.id), experiments=('C',),
                              provider_overrides={model.id: provider}, max_cost_cny=100,
                              max_requests=1, attempt_audit_path=audit_path)
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(events) == 2 and events[-1]['ok'] is True
    assert events[-1]['provider_request_id'] == 'sdk-header-123'
    assert events[-1]['provider_request_id_source'] == 'sdk._request_id'
    assert events[-1]['response_id'] == 'response-456'
    assert events[-1]['run_id'] == result['run_id']
    ledger = ledger_rows(options, result['run_id'])
    assert len(ledger) == 1 and ledger[0]['status'] == 'settled_attempt_usage'
    assert ledger[0]['reservation_id'] == events[-1]['budget_reservation_id']
    _, records = read_run(options, result['run_id'])
    assert records[0]['raw']['request']['image_meta']['encoded_sha256'] == events[0]['image_encoded_sha256']
    assert result['budget']['inflight_cny'] == 0


def test_requested_per_http_audit_rejects_unobservable_provider_without_dispatch(demo, tmp_path):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    provider = UsageProbe()
    with pytest.raises(ValueError, match='requires attempt-observer'):
        runner.run_multi(dataset, **paid_kwargs(options, model), provider_overrides={model: provider},
                          max_cost_cny=100, max_requests=1, attempt_audit_path=tmp_path / 'not-sent.jsonl')
    assert provider.calls == []
    assert not (tmp_path / 'not-sent.jsonl').exists()
