"""Plan/run an isolated 15-image C web-search pilot, inheriting the existing cap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.export_gongfunvzu_html import SAMPLE_IDS
from vcrb.config import load_yaml
from vcrb.multi.runner import run_multi,plan,resolve_models,select_samples
from vcrb.multi.schema import load_dataset
from vcrb.config import load_settings

PARENT='multi_v2_20261113_075729_2e4f15'
CAP=250.0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Send real authorized requests; otherwise only print a plan')
    parser.add_argument('--run-id',default=None)
    parser.add_argument('--max-tasks',type=int,default=15)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    policy_path=ROOT/'results/multi/search_pilot/tool_policy_audit.json'
    if args.execute and policy_path.is_file() and json.loads(policy_path.read_text()).get('additional_search_requests_on_hold'):
        raise ValueError('Previous service responses exceeded the declared search-use cap; review the transport before sending more search requests')
    dataset=ROOT/'data/v2/dataset.json'
    pricing=load_yaml(ROOT/'config/search_pilot_pricing.yaml')
    model=resolve_models(['qwen3.8-max'])[0]
    if (model.params.get('api_mode')!='dashscope_anthropic' or model.params.get('search_max_uses')!=1
            or model.extra_body.get('enable_search') is not True or model.extra_body.get('enable_thinking') is not False):
        raise ValueError('Pilot profile must retain requested max_uses=1 and disabled thinking; re-plan other profiles separately')
    if not args.execute:
        samples=select_samples(load_dataset(dataset),sample_ids=list(SAMPLE_IDS))
        info=plan(samples,resolve_models(['qwen3.8-max']),['C'],load_settings(),pricing)
        print(json.dumps({'mode':'plan_only','samples':info['samples'],'persons':info['persons'],
                          'planned_requests':info['planned_requests'],'parent_budget_run':PARENT,'cumulative_cap_cny':CAP,
                          'profile':'web_search max_uses=1; thinking disabled; actual search evidence required'},ensure_ascii=False,indent=2))
        return
    result=run_multi(dataset,mock=False,yes=True,model_ids=['qwen3.8-max'],experiments=('C',),
                     sample_ids=list(SAMPLE_IDS),max_requests=args.max_tasks,run_id=args.run_id,
                     retry_failed=args.retry_failed,budget_parent_run_id=PARENT,max_cost_cny=CAP,pricing=pricing,
                     approval_note='用户明确允许联网搜索以提高准确率；先用15张功夫女足做独立C组检索对照，继承累计250元上限和旧费用记录。')
    path=ROOT/'results/multi/search_pilot/latest.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('run_id','status','planned','executed','skipped','progress') if k in result},ensure_ascii=False,indent=2))
    print('result_file:',path)


if __name__=='__main__':
    main()
