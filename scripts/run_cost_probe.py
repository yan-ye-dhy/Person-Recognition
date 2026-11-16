"""Seven models, five fixed C images; offline plan unless --execute is explicit.

Actual money is imported from official billing evidence, never from token pricing.
All execution state, balances and receipts stay under ignored results/cost_probe/.
Repeated execution skips every started batch, including failures/interruptions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcrb.config import load_settings, load_works, load_yaml
from vcrb.models import load_registry
from vcrb.multi.runner import InputBuilder, MODEL_IDS, MULTI_DB, plan, run_multi, select_samples
from vcrb.multi.schema import dataset_hash, load_dataset
from vcrb.runner.store import RunStore

CONFIRMED_HASH = 'aa67135a722270b3035b0ad666dd1e099b307098cd6773d053709e18d7386981'
PARENT = 'multi_v2_20261115_135745_d3bd12'
CAP = 500.0
STATE = ROOT / 'results/cost_probe'
REPORT = ROOT / 'reports/2026-11-16-cost-probe/index.html'
SOURCE = ROOT / 'reports/2026-11-15-c-search/summary.json'
SAMPLE_IDS = ['kuangbiao_0011', 'qingyunian2_0004', 'gongfunvzu_0003',
              'jianlai_0003', 'gongfunvzu_0031']
EXPECTED_ENCODED = {
    'kuangbiao_0011': '154cbced390219fb7098189b1a327d0e898c59d773163c0836e04479187dd769',
    'qingyunian2_0004': '88a45165ed15b4b3cb1ff26b072cd51240e9abab7010005ea5d4bbc3236d7176',
    'gongfunvzu_0003': '7a3e076baa3d66e111c941f5c81b287e0ff39b0092cf387e02c688966ff33e7b',
    'jianlai_0003': '399bfa9b1e9f328838c73bce408cd514ce5ab9d071096877ffffefaedbc294d0',
    'gongfunvzu_0031': '3ecacc2e0819a74e142bd8dd4a1fa0adacd359412c437ab5496f8480550f52fe',
}
CONFIG_FILES = ('models.yaml', 'settings.yaml', 'works.yaml', 'c_search_pricing.yaml')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    os.chmod(temporary, 0o600)
    temporary.replace(path)


@contextmanager
def use_profile(directory):
    previous = os.environ.get('VCRB_CONFIG_DIR')
    os.environ['VCRB_CONFIG_DIR'] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('VCRB_CONFIG_DIR', None)
        else:
            os.environ['VCRB_CONFIG_DIR'] = previous


def prepare_profile(directory):
    """Use each model's selected final search protocol, including Flash SSE."""
    directory = Path(directory)
    documents = {name: deepcopy(load_yaml(ROOT / 'config' / name)) for name in CONFIG_FILES}
    sources = json.loads(SOURCE.read_text(encoding='utf-8'))['model_sources']
    by_id = {m['id']: m for m in documents['models.yaml']['models']}
    for mid in MODEL_IDS:
        by_id[mid]['params'] = deepcopy(sources[mid]['request_params'])
        if by_id[mid]['params'].get('search_enabled') is not True:
            raise ValueError(f'{mid}: final source does not enable search')
    if by_id['qwen3.8-flash']['params']['api_mode'] != 'dashscope_multimodal_search':
        raise ValueError('Flash must retain the verified native multimodal SSE profile')
    settings = documents['settings.yaml']
    settings['retry']['max_attempts'] = 1
    settings['runner']['concurrency'] = 1
    settings['runner']['per_model_concurrency'] = {mid: 1 for mid in MODEL_IDS}
    fingerprint = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / 'profile.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['sha256'] != fingerprint:
            raise ValueError('Frozen cost-probe configuration changed; do not resume with a different profile')
        # Verify on-disk contents too; do not silently repair tampering.
        if any(load_yaml(directory / name) != value for name, value in documents.items()):
            raise ValueError('Frozen cost-probe YAML was modified')
    else:
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in documents.items():
            (directory / name).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')
        write_json(manifest_path, {'sha256': fingerprint,
            'model_sources': {mid: sources[mid]['run_id'] for mid in MODEL_IDS},
            'max_http_attempts': 1, 'concurrency': 1})
    return fingerprint


def verify_inputs(dataset, directory):
    samples = load_dataset(dataset)
    if len(samples) != 160 or dataset_hash(samples) != CONFIRMED_HASH:
        raise ValueError('Cost probe requires the approved reviewed 160-image dataset')
    selected = select_samples(samples, sample_ids=SAMPLE_IDS)
    works = {w.slug: w for w in load_works()}
    builder = InputBuilder(dataset.parent, Path(directory) / 'unused_marked')
    inputs = []
    for sample in selected:
        prompt, _data_url, meta = builder(sample, 'C', works[sample.work_slug])
        if meta['encoded_sha256'] != EXPECTED_ENCODED[sample.sample_id]:
            raise ValueError(f'{sample.sample_id}: C image differs from formal benchmark')
        inputs.append({'sample_id': sample.sample_id, 'work_name': sample.work_name,
            'encoded_sha256': meta['encoded_sha256'], 'annotation_mode': meta['annotation_mode'],
            'user_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    return selected, inputs


def refresh_counts(row, directory):
    """Receipts count submitted images; responses separately count completed tasks."""
    starts, ends = [], []
    audit = Path(directory) / 'attempts' / f"{row['model_id']}.jsonl"
    if audit.exists():
        for line in audit.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'http_attempt_start':
                starts.append(event)
            elif event.get('event') == 'http_attempt_end':
                ends.append(event)
    row['api_attempts'] = sum(x.get('kind') == 'inference' for x in starts)
    row['search_attempts'] = sum(x.get('kind') == 'search' for x in starts)
    submitted = {x.get('image_encoded_sha256') for x in starts
                 if x.get('kind') == 'inference' and x.get('image_encoded_sha256')}
    row['submitted_count'] = len(submitted)
    row['submitted_sample_ids'] = [sid for sid in SAMPLE_IDS if EXPECTED_ENCODED[sid] in submitted]
    native_units, known = {}, 0
    for event in ends:
        usage = event.get('search_usage') or {}
        if event.get('kind') == 'inference' and usage.get('known_units') is True:
            known += 1
            for key, value in (usage.get('units') or {}).items():
                if type(value) is int and value >= 0:
                    native_units[key] = native_units.get(key, 0) + value
    row['native_tool_units'] = native_units
    row['native_tool_usage_known_attempts'] = known
    row['success_count'] = row['failure_count'] = 0
    if row.get('run_id'):
        store = RunStore(MULTI_DB)
        try:
            records = store.records(row['run_id'])
        finally:
            store.close()
        # Only transport-start receipts prove submission; local failures can
        # have the legacy ProviderResult default attempts=1 without any HTTP.
        attempted = [r for r in records if r['sample_id'] in row['submitted_sample_ids']]
        row['success_count'] = sum(bool(r['ok']) for r in attempted)
        row['failure_count'] = len(attempted) - row['success_count']
        row['records_without_http_receipt'] = len(records) - len(attempted)
        write_json(Path(directory) / 'tasks' / f"{row['model_id']}.json", records)


def export(manifest, directory):
    from vcrb.cost_probe_report import render_report
    billing_path = Path(directory) / 'billing.json'
    billing = json.loads(billing_path.read_text()) if billing_path.exists() else None
    write_json(Path(directory) / 'manifest.json', manifest)
    render_report(manifest, REPORT, billing=billing)


def execute(manifest, directory):
    from vcrb.cost_probe_balance import snapshot_balance, public_balance_status
    dataset = ROOT / 'data/v2/dataset.json'
    profile = Path(directory) / 'config'
    for row in manifest['models']:
        # Never silently repeat an uncertain submitted/started batch.
        if row['status'] != 'pending':
            refresh_counts(row, directory)
            if row['status'] == 'running':
                row.update(status='stopped', finished_at=now(), stop_reason='Interrupted batch; no automatic replay')
            export(manifest, directory)
            continue
        mid = row['model_id']
        before = snapshot_balance(mid)
        write_json(Path(directory) / 'balances' / f'{mid}_before.json', before)
        row['balance_before'] = public_balance_status(before)
        row.update(status='running', started_at=now())
        export(manifest, directory)  # durable before the first paid request
        print(json.dumps({'model_id': mid, 'status': 'running', 'started_at': row['started_at']}, ensure_ascii=False), flush=True)
        def ready(rid):
            row['run_id'] = rid
            export(manifest, directory)
        try:
            with use_profile(profile):
                result = run_multi(dataset, mock=False, yes=True, model_ids=[mid], experiments=('C',),
                    sample_ids=SAMPLE_IDS, max_requests=5, retry_failed=False,
                    budget_parent_run_id=PARENT, max_cost_cny=CAP,
                    pricing=load_yaml(profile / 'c_search_pricing.yaml'),
                    output_root=Path(directory) / 'runs',
                    on_run_ready=ready, attempt_audit_path=Path(directory) / 'attempts' / f'{mid}.jsonl',
                    approval_note='用户明确批准七个模型各5张原厂联网C组实际费用实测；逐模型串行、单次HTTP不自动重试；继承累计500元保护和全部历史费用，实际金额只认官方账单。')
            write_json(Path(directory) / 'run_results' / f'{mid}.json', result)
            refresh_counts(row, directory)
            row['status'] = 'completed' if row['submitted_count'] == 5 else 'limited'
        except Exception as exc:
            # Exception type is public; full transport messages may contain private IDs.
            row.update(status='failed', stop_reason=type(exc).__name__)
            write_json(Path(directory) / 'errors' / f'{mid}.json', {'error_type': type(exc).__name__, 'message': str(exc)})
            refresh_counts(row, directory)
        finally:
            row['finished_at'] = now()
            after = snapshot_balance(mid)
            write_json(Path(directory) / 'balances' / f'{mid}_after.json', after)
            row['balance_after'] = public_balance_status(after)
            export(manifest, directory)
        print(json.dumps({k: row.get(k) for k in ('model_id', 'status', 'submitted_count', 'success_count', 'failure_count', 'api_attempts', 'search_attempts', 'finished_at')}, ensure_ascii=False), flush=True)
    manifest['finished_at'] = now()
    manifest['status'] = 'completed' if all(r['submitted_count'] == 5 for r in manifest['models']) else 'limited'
    export(manifest, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.execute and args.report_only:
        parser.error('--execute and --report-only are mutually exclusive')
    STATE.mkdir(parents=True, exist_ok=True)
    # One lock covers execution, balance snapshots, profile freezing and state.
    with (STATE / '.execution.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = STATE / 'latest.json'
        if pointer.exists():
            directory = Path(json.loads(pointer.read_text())['directory'])
            manifest = json.loads((directory / 'manifest.json').read_text())
        else:
            if args.report_only:
                raise ValueError('No probe manifest exists')
            directory = STATE / ('cost_probe_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
            manifest = None
        if args.report_only:
            # Reconcile frozen evidence even if labels/configuration later change.
            for row in manifest['models']:
                refresh_counts(row, directory)
            export(manifest, directory)
            print('report:', REPORT, flush=True)
            return manifest
        digest = prepare_profile(directory / 'config')
        with use_profile(directory / 'config'):
            samples, inputs = verify_inputs(ROOT / 'data/v2/dataset.json', directory)
            registry = load_registry()
            info = plan(samples, [registry.get(mid) for mid in MODEL_IDS], ['C'], load_settings(),
                        load_yaml(directory / 'config/c_search_pricing.yaml'))
        overview = {'mode': 'execute' if args.execute else 'report_only' if args.report_only else 'plan_only',
            'models': list(MODEL_IDS), 'sample_ids': SAMPLE_IDS, 'tasks': info['planned_requests'],
            'concurrency': 1, 'http_attempts_per_request': 1, 'profile_sha256': digest,
            'cumulative_protection_cap_cny': CAP, 'official_costs': 'pending_official_bills'}
        if not args.execute and not args.report_only:
            print(json.dumps(overview, ensure_ascii=False, indent=2))
            return overview
        if manifest is None:
            manifest = {'probe_id': directory.name, 'status': 'running', 'created_at': now(),
                'dataset_hash': CONFIRMED_HASH, 'planned_sample_ids': SAMPLE_IDS, 'inputs': inputs,
                'profile_sha256': digest, 'experiment': 'C', 'models': [
                    {'model_id': mid, 'vendor': registry.get(mid).vendor, 'planned_count': 5,
                     'submitted_count': 0, 'success_count': 0, 'failure_count': 0,
                     'status': 'pending', 'api_attempts': 0, 'search_attempts': 0} for mid in MODEL_IDS]}
            write_json(directory / 'manifest.json', manifest)
            write_json(pointer, {'directory': str(directory), 'probe_id': directory.name})
        if manifest['profile_sha256'] != digest or manifest['dataset_hash'] != CONFIRMED_HASH:
            raise ValueError('Existing cost probe scope/configuration mismatch')
        execute(manifest, directory)
        print('report:', REPORT, flush=True)
        print('private_manifest:', directory / 'manifest.json', flush=True)
        return manifest


if __name__ == '__main__':
    main()
