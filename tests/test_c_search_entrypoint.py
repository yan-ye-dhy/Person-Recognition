"""Offline guards for the paid C-search command; all state is temporary."""
from __future__ import annotations

from copy import deepcopy
import json
import socket
import sys
from types import SimpleNamespace

import pytest

from scripts import run_c_search as entry


@pytest.fixture
def command(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Entrypoint tests must never access the network')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    state = tmp_path / 'state'
    monkeypatch.setattr(entry, 'STATE', state)
    monkeypatch.setattr(entry, 'MULTI_DB', tmp_path / 'offline.sqlite')
    ids = ['gongfunvzu_0003'] + [f'offline_sample_{i:03d}' for i in range(1, 160)]
    samples = [SimpleNamespace(sample_id=sid) for sid in ids]
    models = {mid: SimpleNamespace(id=mid, params={'search_enabled': True, 'api_mode': 'offline'})
              for mid in entry.MODEL_IDS}
    pricing = {'models': {mid: {'offline_fixture': True} for mid in entry.MODEL_IDS}}
    calls, reads, records, store_calls = [], [], [], []

    def load_yaml(path):
        reads.append(path)
        return deepcopy(pricing)

    def plan(selected, chosen, experiments, settings, rates):
        assert experiments == ['C']
        assert rates == pricing and settings == {'offline_fixture': True}
        return {'planned_requests': len(selected) * len(chosen),
                'cost_upper_bound_per_task_cny': {m.id: 1. for m in chosen}}

    def run_multi(dataset, **kwargs):
        calls.append((dataset, kwargs))
        rid = kwargs.get('run_id') or 'new-offline-run'
        return {'run_id': rid, 'status': 'done', 'planned': 1120, 'executed': kwargs['max_requests'],
                'skipped': 0, 'progress': {'total': 1120, 'ok': 0, 'failed': 0}}

    class Store:
        def __init__(self, path):
            store_calls.append(('open', path))
        def records(self, run_id):
            store_calls.append(('records', run_id))
            return deepcopy(records)
        def close(self):
            store_calls.append(('close',))

    monkeypatch.setattr(entry, 'load_yaml', load_yaml)
    monkeypatch.setattr(entry, 'load_settings', lambda: {'offline_fixture': True})
    monkeypatch.setattr(entry, 'load_dataset', lambda _path: list(samples))
    monkeypatch.setattr(entry, 'resolve_models', lambda ids: [models[mid] for mid in ids])
    monkeypatch.setattr(entry, 'pilot_first_samples', lambda selected: list(selected))
    monkeypatch.setattr(entry, 'plan', plan)
    monkeypatch.setattr(entry, 'run_multi', run_multi)
    monkeypatch.setattr(entry, 'RunStore', Store)

    def invoke(*args):
        monkeypatch.setattr(sys, 'argv', ['run_c_search.py', *args])
        entry.main()

    def freeze(*, latest=None):
        state.mkdir(exist_ok=True)
        (state / 'profile_selection.json').write_text(json.dumps({'profile_sha256': entry.profile_hash()}))
        if latest:
            (state / 'latest.json').write_text(json.dumps({'run_id': latest}))

    return SimpleNamespace(invoke=invoke, freeze=freeze, calls=calls, reads=reads, records=records,
                           store_calls=store_calls, state=state, root=tmp_path, samples=samples, pricing=pricing)


def test_default_is_read_only_seven_model_c_plan(command, capsys):
    command.invoke()
    output = json.loads(capsys.readouterr().out)
    assert output['mode'] == 'plan_only' and output['phase'] == 'preflight'
    assert output['models'] == list(entry.MODEL_IDS)
    assert output['samples'] == 160 and output['tasks'] == 1120
    assert output['cumulative_cap_cny'] == 500
    assert command.reads == [command.root / 'config/c_search_pricing.yaml']
    assert not command.calls and not command.store_calls and not command.state.exists()


def test_probe_plan_selects_only_two_glm_models_and_one_image(command, capsys):
    command.invoke('--phase', 'probe')
    output = json.loads(capsys.readouterr().out)
    assert output['models'] == ['glm-5.3-flash', 'glm-4.6v']
    assert output['samples'] == 1 and output['tasks'] == 2
    assert not command.calls and not command.state.exists()


@pytest.mark.parametrize('mismatch', [False, True])
def test_paid_formal_command_requires_matching_frozen_profile(command, mismatch):
    if mismatch:
        command.freeze()
        (command.state / 'profile_selection.json').write_text('{"profile_sha256":"old-profile"}')
    with pytest.raises(ValueError, match='Freeze the validated'):
        command.invoke('--execute', '--phase', 'full')
    assert not command.calls and not command.store_calls


def test_probe_has_separate_scope_and_does_not_replace_formal_latest(command):
    command.freeze(latest='formal-existing')
    command.invoke('--execute', '--phase', 'probe')
    dataset, kwargs = command.calls[0]
    assert dataset == command.root / 'data/v2/dataset.json'
    assert kwargs['model_ids'] == ['glm-5.3-flash', 'glm-4.6v']
    assert kwargs['sample_ids'] == ['gongfunvzu_0003']
    assert kwargs['experiments'] == ('C',) and kwargs['max_requests'] == 2
    assert kwargs['budget_parent_run_id'] == entry.PARENT and kwargs['max_cost_cny'] == 500
    assert kwargs['pricing'] == command.pricing and kwargs['mock'] is False and kwargs['yes'] is True
    assert json.loads((command.state / 'latest.json').read_text())['run_id'] == 'formal-existing'
    assert (command.state / 'probe.json').exists()


def test_full_automatically_resumes_latest_instead_of_duplicating_preflight(command):
    command.freeze(latest='formal-existing')
    command.invoke('--execute', '--phase', 'full')
    _, kwargs = command.calls[0]
    assert kwargs['run_id'] == 'formal-existing'
    assert kwargs['budget_parent_run_id'] is None and kwargs['max_requests'] is None
    assert kwargs['model_ids'] == list(entry.MODEL_IDS) and kwargs['experiments'] == ('C',)


def test_completed_preflight_is_noop_and_never_advances_formal_images(command, capsys):
    command.freeze(latest='formal-existing')
    command.records.extend({'sample_id': sample.sample_id, 'model_id': model,
                            'experiment': 'C', 'ok': True}
                           for sample in command.samples[:8] for model in entry.MODEL_IDS)
    before = (command.state / 'latest.json').read_bytes()
    command.invoke('--execute', '--phase', 'preflight')
    output = json.loads(capsys.readouterr().out)
    assert output == {'run_id': 'formal-existing', 'status': 'preflight_already_attempted', 'new_requests': 0}
    assert not command.calls
    assert (command.state / 'latest.json').read_bytes() == before
    assert command.store_calls[-1] == ('close',)


@pytest.mark.parametrize('retry_failed,expected_new', [(False, 7), (True, 14)])
def test_partial_preflight_only_fills_remaining_first_eight_tasks(command, retry_failed, expected_new):
    command.freeze(latest='formal-existing')
    command.records.extend({'sample_id': sample.sample_id, 'model_id': model,
                            'experiment': 'C', 'ok': index < 6}
                           for index, sample in enumerate(command.samples[:7]) for model in entry.MODEL_IDS)
    # Neither another experiment nor later formal images occupy preflight slots.
    command.records.extend([{'sample_id': command.samples[0].sample_id, 'model_id': entry.MODEL_IDS[0],
                             'experiment': 'B', 'ok': True},
                            {'sample_id': command.samples[9].sample_id, 'model_id': entry.MODEL_IDS[0],
                             'experiment': 'C', 'ok': True}])
    args = ['--execute', '--phase', 'preflight'] + (['--retry-failed'] if retry_failed else [])
    command.invoke(*args)
    _, kwargs = command.calls[0]
    assert kwargs['run_id'] == 'formal-existing' and kwargs['max_requests'] == expected_new
    assert kwargs['retry_failed'] is retry_failed and kwargs['budget_parent_run_id'] is None


def test_explicit_new_run_starts_fresh_but_inherits_original_budget(command):
    command.freeze(latest='formal-existing')
    command.invoke('--execute', '--phase', 'preflight', '--new-run')
    _, kwargs = command.calls[0]
    assert kwargs['run_id'] is None and kwargs['max_requests'] == 56
    assert kwargs['budget_parent_run_id'] == entry.PARENT and kwargs['max_cost_cny'] == 500
    assert not command.store_calls
    assert json.loads((command.state / 'latest.json').read_text())['run_id'] == 'new-offline-run'


def test_explicit_run_id_takes_precedence_over_latest(command):
    command.freeze(latest='formal-existing')
    command.invoke('--execute', '--phase', 'full', '--run-id', 'requested-existing')
    assert command.calls[0][1]['run_id'] == 'requested-existing'


def test_ready_callback_persists_new_run_before_an_interrupted_execution(command, monkeypatch):
    command.freeze()
    def interrupted_run(_dataset, **kwargs):
        kwargs['on_run_ready']('persisted-before-first-http')
        saved = json.loads((command.state / 'latest.json').read_text())
        assert saved['run_id'] == 'persisted-before-first-http' and saved['status'] == 'running'
        assert saved['phase'] == 'preflight' and saved['profile_sha256'] == entry.profile_hash()
        raise KeyboardInterrupt('offline simulated process interruption')
    monkeypatch.setattr(entry, 'run_multi', interrupted_run)
    with pytest.raises(KeyboardInterrupt):
        command.invoke('--execute', '--phase', 'preflight')
    assert json.loads((command.state / 'latest.json').read_text())['run_id'] == 'persisted-before-first-http'
