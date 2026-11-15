"""Plan an isolated Flash search revision; paid requests require --execute.

Probe/preflight and full use separate runs. All inherit the existing cumulative
budget; no previous responses or global model settings are rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcrb.config import load_settings, load_yaml
from vcrb.models import load_registry
from vcrb.multi.runner import plan, run_multi, select_samples
from vcrb.multi.schema import dataset_hash, load_dataset

MODEL = 'qwen3.8-flash'
PARENT = 'multi_v2_20261115_135745_d3bd12'
CAP = 500.0
CONFIRMED_HASH = 'f0fda2d9d420ffd3e07563c2e4e6430e3e14b76af9050d977227eb0afa8c67a7'
STATE = ROOT / 'results/multi/c_search/flash_revision'
PROBE_IDS = ['qingyunian2_0004']
PREFLIGHT_IDS = ['kuangbiao_0011', 'qingyunian2_0004', 'jianlai_0003', 'gongfunvzu_0003']
CONFIG_FILES = ('models.yaml', 'settings.yaml', 'works.yaml', 'c_search_pricing.yaml')


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def _reject_credentials(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {'api_key', 'authorization', 'access_token', 'secret', 'password'} and item:
                raise ValueError('Profile configuration must not contain actual credentials')
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)


def prepare_profile(profile: str, phase: str) -> tuple[Path, str]:
    """Persist configuration YAML only; .env and runtime secrets are never copied."""
    directory = STATE / 'profiles' / profile
    directory.mkdir(parents=True, exist_ok=True)
    documents = {}
    for name in CONFIG_FILES:
        source = directory / name
        documents[name] = deepcopy(load_yaml(source if source.is_file() else ROOT / 'config' / name))
        _reject_credentials(documents[name])
    registry = documents['models.yaml']
    matches = [m for m in registry['models'] if m['id'] == MODEL]
    if len(matches) != 1 or matches[0].get('provider') != 'dashscope':
        raise ValueError('The Flash revision must use the original DashScope qwen3.8-flash model')
    model = matches[0]
    params = model.setdefault('params', {})
    extra = params.setdefault('extra_body', {})
    if extra.get('enable_thinking') is not False:
        raise ValueError('The Flash revision must retain disabled thinking')
    extra['enable_search'] = True
    params['search_enabled'] = True
    options = extra.setdefault('search_options', {})
    if profile == 'auto':
        if params.get('api_mode') != 'dashscope_anthropic' or params.get('search_max_uses') != 8:
            raise ValueError('AUTO must preserve the existing Anthropic protocol and nominal max_uses=8')
        # This is the only model-parameter difference from the forced control.
        options['forced_search'] = False
    else:
        params['api_mode'] = 'dashscope_multimodal_search'
        options.update(forced_search=True, search_strategy='max', enable_source=True)
    settings = documents['settings.yaml']
    settings.setdefault('generation', {}).update(temperature=.1, max_tokens=4096)
    settings.setdefault('retry', {}).update(max_attempts=4 if phase == 'full' else 1,
                                            timeout_seconds=300 if phase == 'full' else 120)
    settings.setdefault('runner', {}).setdefault('per_model_http_timeout_seconds', {})[MODEL] = 300 if phase == 'full' else 120
    fingerprint = {'profile': profile, 'model': {'id': MODEL, 'provider': model['provider'], 'params': params},
                   'provider': registry['providers']['dashscope'], 'generation': settings['generation'],
                   'pricing': documents['c_search_pricing.yaml']['models'][MODEL]}
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    previous = directory / 'manifest.json'
    if previous.is_file() and json.loads(previous.read_text())['profile_sha256'] != digest:
        raise ValueError('Persisted Flash protocol profile changed; create an explicitly reviewed revision instead of overwriting')
    for name, document in documents.items():
        (directory / name).write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
    _write_json(previous, {'profile': profile, 'profile_sha256': digest, 'phase': phase,
        'model_id': MODEL, 'config_files': list(CONFIG_FILES), 'generation': settings['generation'],
        'http_timeout_seconds': settings['retry']['timeout_seconds'], 'max_http_attempts_per_task': settings['retry']['max_attempts'],
        'budget_parent_run_id': PARENT, 'max_cost_cny': CAP,
        'limit_note': 'Native server max_uses is nominal and has not reliably bounded internal search rounds.'})
    return directory, digest


@contextmanager
def use_profile(directory: Path):
    previous = os.environ.get('VCRB_CONFIG_DIR')
    os.environ['VCRB_CONFIG_DIR'] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('VCRB_CONFIG_DIR', None)
        else:
            os.environ['VCRB_CONFIG_DIR'] = previous


def _require_verified(profile: str, digest: str) -> dict:
    path = STATE / 'profile_selection.json'
    if not path.is_file():
        raise ValueError('Full execution requires the verified Flash profile_selection.json manifest')
    value = json.loads(path.read_text())
    if (value.get('state') != 'verified' or value.get('profile') != profile
            or value.get('profile_sha256') != digest or value.get('confirmed_dataset_hash') != CONFIRMED_HASH
            or not isinstance(value.get('preflight_run_id'), str) or not value['preflight_run_id']):
        raise ValueError('Flash verification manifest does not match this profile and confirmed dataset')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--profile', choices=['auto', 'multimodal'], default='auto')
    parser.add_argument('--phase', choices=['probe', 'preflight', 'full'], default='probe')
    parser.add_argument('--run-id')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    dataset = ROOT / 'data/v2/dataset.json'
    complete = load_dataset(dataset)
    if len(complete) != 160 or dataset_hash(complete) != CONFIRMED_HASH:
        raise ValueError('Flash revision requires the unchanged confirmed 160-image dataset')
    selected = PROBE_IDS if args.phase == 'probe' else PREFLIGHT_IDS if args.phase == 'preflight' else None
    samples = select_samples(complete, sample_ids=selected) if selected else complete
    directory, digest = prepare_profile(args.profile, args.phase)
    target = STATE / f'{args.profile}_{args.phase}_latest.json'
    with use_profile(directory):
        model = load_registry().get(MODEL)
        pricing = load_yaml(directory / 'c_search_pricing.yaml')
        info = plan(samples, [model], ['C'], load_settings(), pricing)
        overview = {'mode': 'execute' if args.execute else 'plan_only', 'phase': args.phase,
            'profile': args.profile, 'model_id': MODEL, 'sample_ids': selected, 'samples': len(samples),
            'tasks': info['planned_requests'], 'profile_sha256': digest, 'profile_config_dir': str(directory),
            'confirmed_dataset_hash': CONFIRMED_HASH, 'budget_parent_run_id': PARENT, 'cumulative_cap_cny': CAP,
            'http_attempts_per_task': 4 if args.phase == 'full' else 1,
            'cost_upper_bound_per_task_cny': info['cost_upper_bound_per_task_cny'],
            'verification_manifest': str(STATE / 'profile_selection.json'),
            'note': 'Probe/preflight are separate paid diagnostic runs; full creates a separate 160-image cohort. No existing response is replaced.'}
        if not args.execute:
            print(json.dumps(overview, ensure_ascii=False, indent=2))
            return overview
        verification = _require_verified(args.profile, digest) if args.phase == 'full' else None
        run_id = args.run_id
        if run_id is None and target.is_file():
            run_id = json.loads(target.read_text()).get('run_id')
        def ready(rid):
            _write_json(target, {**overview, 'run_id': rid, 'status': 'running'})
        result = run_multi(dataset, mock=False, yes=True, model_ids=[MODEL], experiments=('C',),
            sample_ids=selected, max_requests=None if args.phase == 'full' else len(selected),
            run_id=run_id, retry_failed=args.retry_failed, budget_parent_run_id=None if run_id else PARENT,
            max_cost_cny=CAP, pricing=pricing, on_run_ready=ready,
            approval_note='用户授权原七模型原厂联网C组、思考沿用基线、累计500元；Flash独立修订先核验自动工具选择或原厂多模态搜索，保留全部旧费用和结果。')
        output = {**overview, **result, 'verification': verification,
                  'report_source_override': {MODEL: result.get('run_id')} if args.phase == 'full' else {}}
        _write_json(target, output)
        _write_json(STATE / 'latest.json', output)
        print(json.dumps({k: output[k] for k in ('run_id', 'status', 'phase', 'profile', 'executed') if k in output}, ensure_ascii=False, indent=2))
        return output


if __name__ == '__main__':
    main()
