"""Isolated Flash revision planning, profile freezing and shared budget wiring."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest

from scripts import run_flash_search_revision as script
from vcrb.config import load_settings, load_yaml
from vcrb.models import load_registry


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    real = script.ROOT
    root = tmp_path / 'project'
    (root / 'config').mkdir(parents=True)
    (root / 'data/v2').mkdir(parents=True)
    for name in script.CONFIG_FILES:
        shutil.copyfile(real / 'config' / name, root / 'config' / name)
    shutil.copyfile(real / 'data/v2/dataset.json', root / 'data/v2/dataset.json')
    (root / '.env').write_text('DASHSCOPE_API_KEY=sk-do-not-copy-secret12345\n')
    state = root / 'results/multi/c_search/flash_revision'
    monkeypatch.setattr(script, 'ROOT', root)
    monkeypatch.setattr(script, 'STATE', state)
    return root, state


def flash(document):
    return next(x for x in document['models'] if x['id'] == script.MODEL)


def test_default_plan_is_one_known_loop_frame_with_zero_api_calls_and_environment_restored(workspace, monkeypatch):
    root, state = workspace
    monkeypatch.setenv('VCRB_CONFIG_DIR', '/previous/config')
    monkeypatch.setattr(script, 'run_multi', lambda *args, **kwargs: pytest.fail('Plan cannot call models'))
    before = {p.name: p.read_bytes() for p in (root / 'config').iterdir()}
    result = script.main([])
    assert result['mode'] == 'plan_only' and result['phase'] == 'probe' and result['profile'] == 'auto'
    assert result['sample_ids'] == ['qingyunian2_0004'] and result['tasks'] == result['samples'] == 1
    assert result['http_attempts_per_task'] == 1
    assert result['budget_parent_run_id'] == script.PARENT and result['cumulative_cap_cny'] == 500
    assert os.environ['VCRB_CONFIG_DIR'] == '/previous/config'
    assert before == {p.name: p.read_bytes() for p in (root / 'config').iterdir()}
    directory = state / 'profiles/auto'
    assert set(p.name for p in directory.iterdir()) == set(script.CONFIG_FILES) | {'manifest.json'}
    assert all('sk-do-not-copy' not in p.read_text() for p in directory.iterdir())
    assert not (state / 'latest.json').exists()


def test_auto_changes_only_forced_search_model_parameter_and_overrides_global_300_timeout(workspace):
    root, _ = workspace
    original = flash(load_yaml(root / 'config/models.yaml'))
    expected = deepcopy(original['params'])
    expected['extra_body']['search_options']['forced_search'] = False
    directory, digest = script.prepare_profile('auto', 'probe')
    assert flash(load_yaml(directory / 'models.yaml'))['params'] == expected
    with script.use_profile(directory):
        settings = load_settings()
        assert settings['retry']['max_attempts'] == 1 and settings['retry']['timeout_seconds'] == 120
        assert settings['runner']['per_model_http_timeout_seconds'][script.MODEL] == 120
        assert settings['generation']['temperature'] == .1 and settings['generation']['max_tokens'] == 4096
    _, full_digest = script.prepare_profile('auto', 'full')
    assert full_digest == digest
    with script.use_profile(directory):
        assert load_settings()['retry']['max_attempts'] == 4
        assert load_settings()['runner']['per_model_http_timeout_seconds'][script.MODEL] == 300


def test_multimodal_backup_uses_same_original_model_and_disabled_thinking(workspace):
    directory, _ = script.prepare_profile('multimodal', 'probe')
    with script.use_profile(directory):
        model = load_registry().get(script.MODEL)
        assert model.provider == 'dashscope' and model.params['api_mode'] == 'dashscope_multimodal_search'
        assert model.extra_body['enable_thinking'] is False
        assert model.extra_body['search_options'] == {'forced_search': True, 'search_strategy': 'max', 'enable_source': True}


@pytest.mark.parametrize('phase,expected', [('probe', script.PROBE_IDS), ('preflight', script.PREFLIGHT_IDS)])
def test_paid_diagnostic_scope_is_exactly_one_or_four_and_budget_inherited(workspace, monkeypatch, phase, expected):
    _, state = workspace
    calls = []
    def fake(dataset, **kwargs):
        calls.append(kwargs)
        assert load_settings()['retry']['max_attempts'] == 1
        assert load_settings()['runner']['per_model_http_timeout_seconds'][script.MODEL] == 120
        kwargs['on_run_ready']('new-diagnostic')
        return {'run_id': 'new-diagnostic', 'status': 'done', 'executed': len(expected)}
    monkeypatch.setattr(script, 'run_multi', fake)
    result = script.main(['--execute', '--phase', phase])
    assert len(calls) == 1 and calls[0]['sample_ids'] == expected
    assert calls[0]['max_requests'] == len(expected)
    assert calls[0]['model_ids'] == [script.MODEL] and calls[0]['experiments'] == ('C',)
    assert calls[0]['budget_parent_run_id'] == script.PARENT and calls[0]['max_cost_cny'] == 500
    assert result['report_source_override'] == {}
    assert json.loads((state / f'auto_{phase}_latest.json').read_text())['run_id'] == 'new-diagnostic'


def verified(workspace):
    _, state = workspace
    _, digest = script.prepare_profile('auto', 'preflight')
    document = {'state': 'verified', 'profile': 'auto', 'profile_sha256': digest,
                'confirmed_dataset_hash': script.CONFIRMED_HASH, 'preflight_run_id': 'verified-four-frame-run'}
    (state / 'profile_selection.json').write_text(json.dumps(document))
    return document


def test_full_requires_root_verification_before_paid_call(workspace, monkeypatch):
    monkeypatch.setattr(script, 'run_multi', lambda *args, **kwargs: pytest.fail('Unverified profile must not call APIs'))
    with pytest.raises(ValueError, match='verified Flash'):
        script.main(['--execute', '--phase', 'full'])


def test_full_new_160_cohort_and_repeat_resumes_it_instead_of_duplicate_run(workspace, monkeypatch):
    verified(workspace)
    calls = []
    def fake(dataset, **kwargs):
        calls.append(kwargs)
        settings = load_settings()
        assert settings['retry']['max_attempts'] == 4 and settings['runner']['per_model_http_timeout_seconds'][script.MODEL] == 300
        assert kwargs['sample_ids'] is None and kwargs['max_requests'] is None
        kwargs['on_run_ready']('flash-full160')
        return {'run_id': 'flash-full160', 'status': 'limited', 'executed': 5}
    monkeypatch.setattr(script, 'run_multi', fake)
    result = script.main(['--execute', '--phase', 'full'])
    script.main(['--execute', '--phase', 'full'])
    assert result['samples'] == result['tasks'] == 160
    assert result['report_source_override'] == {script.MODEL: 'flash-full160'}
    assert calls[0]['run_id'] is None and calls[0]['budget_parent_run_id'] == script.PARENT
    assert calls[1]['run_id'] == 'flash-full160' and calls[1]['budget_parent_run_id'] is None


def test_tampered_verification_or_profile_rejected(workspace, monkeypatch):
    _, state = workspace
    value = verified(workspace)
    value['profile_sha256'] = 'bad'
    (state / 'profile_selection.json').write_text(json.dumps(value))
    monkeypatch.setattr(script, 'run_multi', lambda *args, **kwargs: pytest.fail('Manifest mismatch must not call models'))
    with pytest.raises(ValueError, match='does not match'):
        script.main(['--execute', '--phase', 'full'])


def test_environment_restored_even_when_run_is_interrupted(workspace, monkeypatch):
    monkeypatch.delenv('VCRB_CONFIG_DIR', raising=False)
    def interrupted(*args, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(script, 'run_multi', interrupted)
    with pytest.raises(KeyboardInterrupt):
        script.main(['--execute'])
    assert 'VCRB_CONFIG_DIR' not in os.environ
