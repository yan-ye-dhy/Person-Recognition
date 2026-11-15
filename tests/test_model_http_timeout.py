"""Operational HTTP timeout overrides do not mutate benchmark/model settings."""
from copy import deepcopy
from dataclasses import replace

import pytest

from tests.conftest import build_tiny_dataset
from vcrb.cli import _build_provider
from vcrb.models import load_registry
from vcrb.providers import MockProvider
from vcrb.runner.executor import _effective_http_timeout_seconds, _fairness_meta, _request_tool_metadata, execute_run
from vcrb.runner.store import RunStore


def settings():
    return {'retry': {'timeout_seconds': 120, 'max_attempts': 2},
            'generation': {'temperature': .1, 'max_tokens': 4096},
            'runner': {'per_model_http_timeout_seconds': {'qwen3.8-flash': 300}}}


def test_factory_receives_model_specific_timeout_and_does_not_mutate_original_settings(monkeypatch):
    registry = load_registry()
    captured = []
    def fake(spec, **kwargs):
        captured.append(kwargs['settings'])
        return object()
    monkeypatch.setattr('vcrb.providers.dashscope_anthropic.DashScopeAnthropicProvider', fake)
    flash = replace(registry.get('qwen3.8-flash'), params={'api_mode': 'dashscope_anthropic'})
    maximum = replace(registry.get('qwen3.8-max'), params={'api_mode': 'dashscope_anthropic'})
    source = settings(); before = deepcopy(source)
    _build_provider(flash, registry, source)
    _build_provider(maximum, registry, source)
    assert captured[0]['retry']['timeout_seconds'] == 300
    assert captured[1]['retry']['timeout_seconds'] == 120
    assert source == before and captured[0] is not source and captured[0]['retry'] is not source['retry']
    assert captured[0]['generation'] == before['generation']
    assert flash.params == {'api_mode': 'dashscope_anthropic'}


@pytest.mark.parametrize('value', [0, -1, True, False, float('nan'), float('inf'), float('-inf'), '300', None])
def test_invalid_override_is_rejected_before_provider_or_http(value, monkeypatch):
    registry = load_registry()
    model = replace(registry.get('qwen3.8-flash'), params={'api_mode': 'dashscope_anthropic'})
    monkeypatch.setattr('vcrb.providers.dashscope_anthropic.DashScopeAnthropicProvider',
                        lambda *args, **kwargs: pytest.fail('Invalid timeout must not construct provider'))
    source = settings(); source['runner']['per_model_http_timeout_seconds'][model.id] = value
    with pytest.raises(ValueError, match='finite positive'):
        _build_provider(model, registry, source)


@pytest.mark.parametrize('mapping', [None, [], 300, 'qwen3.8-flash=300'])
def test_timeout_overrides_require_mapping(mapping):
    source = settings(); source['runner']['per_model_http_timeout_seconds'] = mapping
    with pytest.raises(ValueError, match='mapping'):
        _effective_http_timeout_seconds(load_registry().get('qwen3.8-flash'), source)


def test_request_and_fairness_metadata_record_effective_timeout_with_legacy_defaults():
    flash = load_registry().get('qwen3.8-flash')
    maximum = load_registry().get('qwen3.8-max')
    source = settings()
    assert _request_tool_metadata(flash, source)['http_timeout_seconds'] == 300
    assert _request_tool_metadata(maximum, source)['http_timeout_seconds'] == 120
    assert _request_tool_metadata(flash)['http_timeout_seconds'] == 120
    meta = _fairness_meta([flash, maximum], source)
    assert meta['per_model_notes'][flash.id]['http_timeout_seconds'] == 300
    assert meta['per_model_notes'][maximum.id]['http_timeout_seconds'] == 120
    assert meta['search_policy'][flash.id]['http_timeout_seconds'] == 300
    assert meta['generation_params']['max_tokens'] == 4096


def test_actual_saved_request_records_same_effective_timeout(tmp_path):
    dataset_dir = tmp_path / 'dataset'
    samples, works = build_tiny_dataset(dataset_dir, n_per_work=1)
    model = load_registry().get('qwen3.8-flash')
    store = RunStore(tmp_path / 'runs.db')
    store.create_run(run_id='timeout-test', experiments=['A'], models=[model.id], prompt_version='v1',
                     dataset_hash='test', dataset_path=str(dataset_dir), fairness_meta={})
    summary = execute_run('timeout-test', [(model, MockProvider())], samples[:1], works, ['A'], store, settings(),
                          dataset_dir, marked_dir=tmp_path/'marked', raw_dir=tmp_path/'raw')
    assert summary.ok == 1
    row = store.conn.execute('select raw_json from records where run_id=?', ('timeout-test',)).fetchone()
    import json
    assert json.loads(row[0])['request']['http_timeout_seconds'] == 300
