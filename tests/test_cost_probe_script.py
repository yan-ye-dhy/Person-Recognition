"""The cost experiment must never silently add retries or change paid scope."""
import json

import pytest

from scripts import run_cost_probe as script
from vcrb.config import load_yaml


def test_offline_default_verifies_fixed_five_inputs_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(script, 'STATE', tmp_path / 'private')
    monkeypatch.setattr(script, 'run_multi', lambda *a, **kw: pytest.fail('offline plan cannot call APIs'))
    result = script.main([])
    assert result['tasks'] == 35 and len(result['sample_ids']) == 5
    assert len(result['models']) == 7
    assert result['concurrency'] == result['http_attempts_per_request'] == 1
    assert result['cumulative_protection_cap_cny'] == 500
    assert result['official_costs'] == 'pending_official_bills'
    assert not (script.STATE / 'latest.json').exists()


def test_final_profiles_match_selected_search_runs_and_do_not_mutate_global_config(tmp_path):
    before = {name: (script.ROOT / 'config' / name).read_bytes() for name in script.CONFIG_FILES}
    directory = tmp_path / 'config'
    digest = script.prepare_profile(directory)
    assert script.prepare_profile(directory) == digest
    settings = load_yaml(directory / 'settings.yaml')
    assert settings['retry']['max_attempts'] == settings['runner']['concurrency'] == 1
    assert set(settings['runner']['per_model_concurrency'].values()) == {1}
    sources = json.loads(script.SOURCE.read_text())['model_sources']
    models = load_yaml(directory / 'models.yaml')['models']
    for model in models:
        if model['id'] in script.MODEL_IDS:
            assert model['params'] == sources[model['id']]['request_params']
    assert before == {name: (script.ROOT / 'config' / name).read_bytes() for name in script.CONFIG_FILES}
    assert not (directory / '.env').exists()
    (directory / 'settings.yaml').write_text('tampered: true\n')
    with pytest.raises(ValueError, match='modified'):
        script.prepare_profile(directory)


def probe_manifest():
    return {'models': [{'model_id': mid, 'status': 'pending', 'submitted_count': 0,
                       'success_count': 0, 'failure_count': 0} for mid in script.MODEL_IDS]}


def mock_execution(monkeypatch):
    import vcrb.cost_probe_balance as balance
    monkeypatch.setattr(balance, 'snapshot_balance', lambda mid: {'status': 'unavailable', 'reason': 'mock'})
    monkeypatch.setattr(script, 'export', lambda *a: None)
    monkeypatch.setattr(script, 'load_yaml', lambda *a: {})
    calls = []
    def fake(dataset, **kwargs):
        calls.append(kwargs)
        assert kwargs['experiments'] == ('C',)
        assert kwargs['sample_ids'] == script.SAMPLE_IDS
        assert kwargs['max_requests'] == 5 and not kwargs['retry_failed']
        assert kwargs['budget_parent_run_id'] == script.PARENT
        assert kwargs['max_cost_cny'] == 500
        assert kwargs['attempt_audit_path'].suffix == '.jsonl'
        kwargs['on_run_ready']('new-' + kwargs['model_ids'][0])
        return {'executed': 5}
    monkeypatch.setattr(script, 'run_multi', fake)
    monkeypatch.setattr(script, 'refresh_counts', lambda row, _: row.update(
        submitted_count=5, success_count=4, failure_count=1, api_attempts=7, search_attempts=2))
    return calls


def test_serial_seven_batches_never_replay_completed_or_failed_batches(tmp_path, monkeypatch):
    calls = mock_execution(monkeypatch)
    manifest = probe_manifest()
    script.execute(manifest, tmp_path)
    assert [k['model_ids'][0] for k in calls] == list(script.MODEL_IDS)
    assert len(calls) == 7 and manifest['status'] == 'completed'
    assert all(row['submitted_count'] == 5 for row in manifest['models'])
    script.execute(manifest, tmp_path)
    assert len(calls) == 7


def test_interrupted_batch_is_stopped_without_automatic_paid_replay(tmp_path, monkeypatch):
    calls = mock_execution(monkeypatch)
    manifest = probe_manifest()
    manifest['models'][0]['status'] = 'running'
    script.execute(manifest, tmp_path)
    assert len(calls) == 6
    assert manifest['models'][0]['status'] == 'stopped'
    assert 'no automatic replay' in manifest['models'][0]['stop_reason']


def test_receipt_counts_real_inference_and_search_not_intent_or_bill_estimates(tmp_path):
    mid = 'glm-4.6v'
    path = tmp_path / 'attempts' / f'{mid}.jsonl'
    path.parent.mkdir()
    events = [
        {'event': 'http_attempt_start', 'kind': 'inference', 'image_encoded_sha256': 'image1'},
        {'event': 'http_attempt_end', 'kind': 'inference', 'image_encoded_sha256': 'image1'},
        {'event': 'http_attempt_start', 'kind': 'search', 'image_encoded_sha256': 'image1'},
        {'event': 'http_attempt_start', 'kind': 'inference', 'image_encoded_sha256': 'image1'},
        {'event': 'http_attempt_start', 'kind': 'inference', 'image_encoded_sha256': 'image2'},
    ]
    path.write_text('\n'.join(json.dumps(x) for x in events))
    row = {'model_id': mid}
    script.refresh_counts(row, tmp_path)
    assert row['submitted_count'] == 2
    assert row['api_attempts'] == 3 and row['search_attempts'] == 1
    assert not any('cny' in key for key in row)


def test_local_exception_with_legacy_attempts_one_is_not_a_submitted_http(tmp_path, monkeypatch):
    class Store:
        def __init__(self, _): pass
        def records(self, _):
            return [{'sample_id': script.SAMPLE_IDS[0], 'ok': False,
                     'raw': {'response': {'attempts': 1, 'error_type': 'internal_error'}}}]
        def close(self): pass
    monkeypatch.setattr(script, 'RunStore', Store)
    row = {'model_id': 'deepseek-flash', 'run_id': 'mock-no-transport'}
    script.refresh_counts(row, tmp_path)
    assert row['submitted_count'] == row['success_count'] == row['failure_count'] == 0
    assert row['records_without_http_receipt'] == 1


def test_report_only_uses_frozen_receipts_even_after_config_or_label_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(script, 'STATE', tmp_path)
    directory = tmp_path / 'frozen'
    directory.mkdir()
    script.write_json(tmp_path / 'latest.json', {'directory': str(directory)})
    script.write_json(directory / 'manifest.json', probe_manifest())
    monkeypatch.setattr(script, 'prepare_profile', lambda *a: pytest.fail('report must not inspect new parameters'))
    monkeypatch.setattr(script, 'verify_inputs', lambda *a: pytest.fail('report must not inspect edited labels'))
    monkeypatch.setattr(script, 'refresh_counts', lambda *a: None)
    exports = []
    monkeypatch.setattr(script, 'export', lambda *a: exports.append(a))
    script.main(['--report-only'])
    assert len(exports) == 1
