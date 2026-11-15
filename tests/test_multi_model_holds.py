"""Run-scoped interface holds are administrative state, not model profiles."""
from __future__ import annotations

import json

import pytest

from tests.test_multi_runner import demo, offline_only, manual_test_fixture, prices, read_run
from tests.test_multi_budget import UsageProbe
from vcrb.multi import runner

FLASH = 'qwen3.8-flash'
OTHERS = ['qwen3.8-max', 'qwen3.7-plus']


def set_holds(**updates):
    gate = json.loads(runner.REAL_CALL_GATE.read_text())
    gate.update(updates)
    runner.REAL_CALL_GATE.write_text(json.dumps(gate))


def paid(options, mids):
    return {**options,'mock':False,'yes':True,'experiments':('C',),'model_ids':mids,
            'limit':1,'pricing':prices(mids),'max_cost_cny':350,'approval_note':'OFFLINE TEST ONLY'}


def test_main_run_model_hold_allows_siblings_and_revision_but_vendor_hold_still_wins(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    mids = [FLASH, *OTHERS]
    probes = {mid:UsageProbe() for mid in mids}
    kwargs = paid(options, mids)
    main = runner.run_multi(dataset, **kwargs, max_requests=0, provider_overrides=probes)
    hold = {'state':'held','run_id':main['run_id'],'kind':'interface_revision','reason':'Separate auto-search cohort'}
    set_holds(model_holds={FLASH:hold})
    resumed = runner.run_multi(dataset, **kwargs, run_id=main['run_id'], provider_overrides=probes)
    assert resumed['executed'] == 2 and not probes[FLASH].calls
    assert all(len(probes[mid].calls) == 1 for mid in OTHERS)
    assert resumed['budget']['blocked_models'] == [FLASH] and not resumed['budget']['blocked_providers']
    main_meta, _ = read_run(options, main['run_id'])
    assert main_meta['fairness_meta']['model_holds'] == {FLASH:hold}

    # Same actual model ID, distinct frozen run: the old-run hold does not apply.
    revision_probe = UsageProbe()
    revision = runner.run_multi(dataset, **paid(options,[FLASH]), budget_parent_run_id=main['run_id'],
                                provider_overrides={FLASH:revision_probe})
    assert revision['run_id'] != main['run_id'] and revision['executed'] == 1
    assert len(revision_probe.calls) == 1 and revision['budget']['blocked_models'] == []
    revision_meta, _ = read_run(options, revision['run_id'])
    assert revision_meta['fairness_meta']['model_holds'] == {}

    set_holds(provider_holds={'dashscope':{'reason':'balance requires confirmation'}})
    blocked_probe = UsageProbe()
    blocked = runner.run_multi(dataset, **paid(options,[FLASH]), budget_parent_run_id=main['run_id'],
                               provider_overrides={FLASH:blocked_probe})
    assert blocked['executed'] == 0 and not blocked_probe.calls
    assert blocked['budget']['blocked_providers'] == ['dashscope']
    set_holds(provider_holds={})
    clear = runner.run_multi(dataset, **paid(options,[FLASH]), run_id=blocked['run_id'],
                             provider_overrides={FLASH:blocked_probe})
    assert clear['executed'] == 1 and len(blocked_probe.calls) == 1
    assert clear['budget']['blocked_models'] == []
    # Removing a vendor balance hold does not remove the separate old interface hold.
    still_held = runner.run_multi(dataset, **kwargs, run_id=main['run_id'], provider_overrides=probes)
    assert still_held['executed'] == 0 and not probes[FLASH].calls
    assert still_held['budget']['blocked_models'] == [FLASH]


def test_model_hold_without_run_id_applies_to_every_revision(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    set_holds(model_holds={FLASH:{'state':'held','reason':'Global model hold'}})
    first_probe, second_probe = UsageProbe(), UsageProbe()
    first = runner.run_multi(dataset, **paid(options,[FLASH]), provider_overrides={FLASH:first_probe})
    second = runner.run_multi(dataset, **paid(options,[FLASH]), budget_parent_run_id=first['run_id'],
                              provider_overrides={FLASH:second_probe})
    assert first['executed'] == second['executed'] == 0
    assert not first_probe.calls and not second_probe.calls
    assert first['budget']['blocked_models'] == second['budget']['blocked_models'] == [FLASH]


@pytest.mark.parametrize('holds', [
    [], {FLASH:None}, {'unknown-model':{'state':'held'}},
    {FLASH:{'state':'held','run_id':None}}, {FLASH:{'state':'held','run_id':''}},
    {FLASH:{'state':'released'}},
])
def test_malformed_model_holds_fail_closed_before_any_call(demo, holds):
    dataset, options = demo
    manual_test_fixture(dataset)
    set_holds(model_holds=holds)
    probe = UsageProbe()
    result = runner.run_multi(dataset, **paid(options,[FLASH]), provider_overrides={FLASH:probe})
    assert result['executed'] == 0 and not probe.calls
    assert result['budget']['real_call_gate_blocked'] is True
