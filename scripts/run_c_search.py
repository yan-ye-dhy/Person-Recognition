"""Run the authorized original-vendor C benchmark; default is an offline plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcrb.config import load_settings, load_yaml
from vcrb.multi.runner import MODEL_IDS, MULTI_DB, plan, resolve_models, run_multi, pilot_first_samples
from vcrb.multi.schema import load_dataset
from vcrb.runner.store import RunStore

PARENT = 'multi_v2_20261115_135745_d3bd12'
STATE = ROOT / 'results/multi/c_search'
CAP = 500.0


def profile_hash():
    return hashlib.sha256(json.dumps({m.id: m.params for m in resolve_models(MODEL_IDS)},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', choices=['probe', 'preflight', 'full'], default='preflight')
    parser.add_argument('--run-id')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--retry-failed-model', action='append', help='Limit existing-failure retries to these authorized model IDs')
    parser.add_argument('--retry-failed-type', action='append', help='Limit existing-failure retries to these error types, e.g. insufficient_balance')
    parser.add_argument('--new-run', action='store_true', help='Explicitly start a new formal cohort instead of resuming latest')
    parser.add_argument('--probe-model', action='append', help='Probe one of the seven authorized models; repeat for more models')
    args = parser.parse_args()
    if args.retry_failed_model and (not args.retry_failed or any(mid not in MODEL_IDS for mid in args.retry_failed_model)):
        raise ValueError('--retry-failed-model requires --retry-failed and authorized seven-model IDs')
    if args.retry_failed_type and (not args.retry_failed or any(not value.strip() for value in args.retry_failed_type)):
        raise ValueError('--retry-failed-type requires --retry-failed and nonempty error types')
    dataset = ROOT / 'data/v2/dataset.json'
    pricing = load_yaml(ROOT / 'config/c_search_pricing.yaml')
    if args.probe_model and (args.phase != 'probe' or any(mid not in MODEL_IDS for mid in args.probe_model)):
        raise ValueError('--probe-model is limited to authorized seven-model capability probes')
    ids = (args.probe_model or ['glm-5.3-flash', 'glm-4.6v']) if args.phase == 'probe' else list(MODEL_IDS)
    samples = load_dataset(dataset)
    selected = ['gongfunvzu_0003'] if args.phase == 'probe' else None
    if selected:
        samples = [s for s in samples if s.sample_id in selected]
    info = plan(samples, resolve_models(ids), ['C'], load_settings(), pricing)
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'phase': args.phase, 'models': ids,
            'samples': len(samples), 'tasks': info['planned_requests'], 'cumulative_cap_cny': CAP,
            'profile_sha256': profile_hash(), 'cost_upper_bound_per_task_cny': info['cost_upper_bound_per_task_cny']},
            ensure_ascii=False, indent=2))
        return
    if args.phase != 'probe':
        selection = STATE / 'profile_selection.json'
        if not selection.is_file() or json.loads(selection.read_text()).get('profile_sha256') != profile_hash():
            raise ValueError('Freeze the validated GLM native/function profiles before creating or resuming the formal C run')
        latest = STATE / 'latest.json'
        if not args.run_id and not args.new_run and latest.is_file():
            args.run_id = json.loads(latest.read_text())['run_id']
    cap = len(ids) if args.phase == 'probe' else (56 if args.phase == 'preflight' else None)
    if args.phase == 'preflight' and args.run_id:
        store = RunStore(MULTI_DB)
        try:
            first = {s.sample_id for s in pilot_first_samples(samples)[:8]}
            attempted = [r for r in store.records(args.run_id) if r['sample_id'] in first and r['experiment'] == 'C']
            complete = sum(bool(r['ok']) or args.retry_failed_model is not None and r['model_id'] not in args.retry_failed_model
                           or args.retry_failed_type is not None and ((r.get('raw') or {}).get('response') or {}).get('error_type') not in args.retry_failed_type
                           for r in attempted) if args.retry_failed else len(attempted)
            cap = max(0, 56 - complete)
            if cap == 0:
                print(json.dumps({'run_id':args.run_id,'status':'preflight_already_attempted','new_requests':0}))
                return
        finally:
            store.close()
    STATE.mkdir(parents=True, exist_ok=True)
    target = STATE / ('probe.json' if args.phase == 'probe' else 'latest.json')
    def ready(rid):
        target.write_text(json.dumps({'run_id':rid,'status':'running','phase':args.phase,
            'profile_sha256':profile_hash()},ensure_ascii=False,indent=2)+'\n')
    result = run_multi(dataset, mock=False, yes=True, model_ids=ids, experiments=('C',),
        sample_ids=selected, max_requests=cap,
        run_id=args.run_id, retry_failed=args.retry_failed, budget_parent_run_id=PARENT if not args.run_id else None,
        max_cost_cny=CAP, pricing=pricing, on_run_ready=ready,
        retry_failed_models=set(args.retry_failed_model) if args.retry_failed_model else None,
        retry_failed_error_types=set(args.retry_failed_type) if args.retry_failed_type else None,
        approval_note='用户授权原七模型原厂联网，仅C160；GLM可接原厂搜索循环，优先准确率，沿用思考设置；用户明确将累计保护提高到500元并保留全部旧记录继续。')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps({k: result[k] for k in ('run_id','status','planned','executed','skipped','progress') if k in result}, ensure_ascii=False, indent=2))
    print('result_file:', target)


if __name__ == '__main__':
    main()
