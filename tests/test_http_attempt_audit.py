"""Offline per-HTTP evidence: durable starts, retries and vendor-tool rounds."""
from datetime import datetime
import json
import stat
from types import SimpleNamespace

import httpx
import pytest

from tests.test_native_search_providers import make_provider, call, response_for
from tests.test_zhipu_search import provider as zhipu_provider, call as zhipu_call, answer, tool, Observer
from vcrb.providers.http_audit import AttemptAuditObserver, response_identity


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('vendor', ['deepseek', 'moonshot'])
def test_each_native_http_retry_has_durable_pair_and_distinct_request_response_ids(vendor, monkeypatch, tmp_path):
    path = tmp_path / 'private' / 'http.jsonl'
    class Delegate:
        def __init__(self): self.before, self.after = [], []
        def before_attempt(self, *args):
            receipt = 'reservation-' + str(len(self.before) + 1)
            self.before.append(receipt)
            return receipt
        def after_attempt(self, receipt, model, *, result, error): self.after.append(receipt)
    delegate = Delegate()
    observer = AttemptAuditObserver(path, 'offline-run', delegate)
    calls = []
    def handler(request):
        # No transport can start without its complete, fsynced audit start.
        current = rows(path)
        assert current[-1]['event'] == 'http_attempt_start'
        assert current[-1]['budget_reservation_id'] == delegate.before[-1]
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, headers={'x-request-id': 'request-failed'},
                                  json={'request_id': 'body-failed', 'message': 'offline failure'})
        body = response_for(vendor)
        body['id'] = 'response-success'
        return httpx.Response(200, headers={'x-request-id': 'request-success', 'authorization': 'secret-not-audit'}, json=body)
    p = make_provider(vendor, monkeypatch, handler, attempts=2)
    result, _, _ = call(p, vendor, observer=observer)
    events = rows(path)
    assert result.ok and result.attempts == 2 and len(calls) == 2
    assert [e['event'] for e in events] == ['http_attempt_start', 'http_attempt_end'] * 2
    assert delegate.before == delegate.after == ['reservation-1', 'reservation-2']
    assert events[0]['audit_attempt_id'] == events[1]['audit_attempt_id']
    assert events[2]['audit_attempt_id'] == events[3]['audit_attempt_id'] != events[0]['audit_attempt_id']
    assert events[1]['ok'] is False and events[1]['http_status'] == 503
    assert events[1]['provider_request_id'] == 'request-failed'
    assert events[3]['provider_request_id'] == 'request-success'
    assert events[3]['provider_request_id_source'] == 'response_header.x-request-id'
    assert events[3]['response_id'] == 'response-success'
    assert events[3]['usage']['prompt_tokens'] == 30
    assert events[3]['search_usage']['known_units'] is True
    assert datetime.fromisoformat(events[3]['ended_at']) >= datetime.fromisoformat(events[3]['started_at'])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    assert 'sk-unit-' not in text and 'secret-not-audit' not in text and 'base64' not in text


def test_zhipu_two_model_rounds_and_one_external_search_have_separate_http_evidence(monkeypatch, tmp_path):
    path = tmp_path / 'glm.jsonl'
    first, last = answer(calls=[tool()], content=None), answer()
    first.id, first.request_id = 'response-round-1', 'request-round-1'
    last.id, last.request_id = 'response-round-2', 'request-round-2'
    def search(request):
        assert rows(path)[-1]['kind'] == 'search'
        return httpx.Response(200, headers={'request-id': 'search-header-id'}, json={
            'id': 'search-response-id', 'request_id': 'search-body-id',
            'search_result': [{'link': 'https://example.test/role', 'title': 'role'}]})
    p, _ = zhipu_provider(monkeypatch, [first, last], search)
    delegate = Observer()
    result = zhipu_call(p, observer=AttemptAuditObserver(path, 'offline-glm', delegate))
    ends = [e for e in rows(path) if e['event'] == 'http_attempt_end']
    assert result.ok and [e['kind'] for e in ends] == ['inference', 'search', 'inference']
    assert [e['provider_request_id'] for e in ends] == ['request-round-1', 'search-header-id', 'request-round-2']
    assert ends[1]['response_id'] == 'search-response-id'
    assert ends[1]['search_usage'] == {'units': {'search': 1}, 'known_units': True}
    assert ends[0]['search_usage']['units'] == ends[2]['search_usage']['units'] == {'search': 0}
    assert sum(e['usage'].get('prompt_tokens', 0) for e in ends) == result.usage['prompt_tokens'] == 40
    assert delegate.events == ['before_model', 'after_model', 'before_search', 'after_search', 'before_model', 'after_model']


def test_response_id_never_becomes_request_id_and_sse_duplicate_ids_are_not_counts():
    assert response_identity({'id': 'response-123'}) == {
        'provider_request_id': None, 'provider_request_id_source': None,
        'response_id': 'response-123', 'response_id_source': 'response_body.id'}
    identity = response_identity({'events': [{'request_id': 'sse-1'}] * 20})
    assert identity['provider_request_id'] == 'sse-1'
    assert identity['sse_request_ids'] == ['sse-1']
    assert response_identity({'request_id': 'sk-do-not-log-this-secret'})['provider_request_id'] is None


def test_budget_rejection_has_no_audit_start_and_no_transport(tmp_path):
    class Reject:
        def before_attempt(self, *args): raise RuntimeError('offline admission rejected')
    path = tmp_path / 'never-started.jsonl'
    observer = AttemptAuditObserver(path, 'offline', Reject())
    with pytest.raises(RuntimeError, match='admission rejected'):
        observer.before_attempt(SimpleNamespace(id='model', provider='vendor'), 'data:image/jpeg;base64,AAAA', 'prompt', 1)
    assert not path.exists()


def test_interruption_is_audited_even_if_delegate_stops_and_metering_discards_text(tmp_path):
    from vcrb.providers.base import ProviderResult
    class StopAfter:
        def before_attempt(self, *args): return 'reserved'
        def after_attempt(self, *args, **kwargs): raise KeyboardInterrupt()
    path = tmp_path / 'interruption.jsonl'
    observer = AttemptAuditObserver(path, 'offline', StopAfter())
    model = SimpleNamespace(id='model', provider='vendor')
    receipt = observer.before_attempt(model, 'data:image/jpeg;base64,AAAA', 'secret prompt', 1)
    result = ProviderResult(ok=True, usage={'prompt_tokens': 12, 'bad': 'sk-secret', 'api_key': 'do-not-log'})
    with pytest.raises(KeyboardInterrupt): observer.after_attempt(receipt, model, result=result)
    assert rows(path)[-1]['usage'] == {'prompt_tokens': 12, 'bad': None}
    assert 'secret prompt' not in path.read_text() and 'sk-secret' not in path.read_text()
