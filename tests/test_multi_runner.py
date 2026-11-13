"""Network-disabled v2 execution, persistence, input fairness, and budget checks.

All files/SQLite stores are temporary. The paid-path tests use locally drawn
stand-ins labelled as manual test fixtures plus an injected offline provider;
no real film image, credential, client, or paid API is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from vcrb.config import load_settings, load_works
from vcrb.imageutils import encode_for_request
from vcrb.multi.mock import create_demo_dataset
from vcrb.multi.prompts import SYSTEM_PROMPT, PROMPT_VERSION
from vcrb.multi import runner
from vcrb.multi.schema import load_dataset
from vcrb.providers.base import ProviderResult, VisionProvider
from vcrb.runner.store import RunStore


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/client/credentials forbidden in multi-runner tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    import vcrb.cli as main_cli
    monkeypatch.setattr(main_cli, "_build_provider", forbidden)
    monkeypatch.setattr(runner, "REAL_CALL_GATE", tmp_path / "real_call_gate.json")
    settings = deepcopy(load_settings())
    settings["runner"] = {"concurrency": 1}
    settings["retry"] = {"max_attempts": 2}
    settings["generation"] = {"temperature": .1, "max_tokens": 100}
    monkeypatch.setattr(runner, "load_settings", lambda: deepcopy(settings))
    yield


@pytest.fixture
def demo(tmp_path):
    dataset = create_demo_dataset(tmp_path / "fixture")
    return dataset, {"db_path": tmp_path / "runs.db", "output_root": tmp_path / "outputs"}


def read_run(options, run_id):
    store = RunStore(options["db_path"])
    try:
        return store.get_run(run_id), store.records(run_id)
    finally:
        store.close()


def all_bytes(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def manual_test_fixture(dataset: Path, *, approved=True):
    """Relabel only temporary geometric test data to exercise the real gate."""
    payload = json.loads(dataset.read_text())
    payload.pop("synthetic_mock_fixture", None)
    payload.pop("mock", None)
    payload["notice"] = "TEST ONLY: generated stand-ins for manual approval gate; no real requests"
    for sample in payload["samples"]:
        sample["sample_id"] = sample["sample_id"].replace("synthetic_", "offline_manual_", 1)
        sample["evidence"] = [{"type": "manual_test_fixture", "actor": "unit-test"}]
        sample["note"] = "Temporary offline test stand-in, never a real benchmark"
        if not approved:
            sample["review_status"], sample["foreground_complete"] = "draft", False
        for person in sample["persons"]:
            person["evidence"] = [{"type": "manual_test_fixture", "actor": "unit-test"}]
            if not approved:
                person["identity_status"], person["review_status"] = "needs_review", "pending"
    dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    approve_offline_fixture(dataset)
    return dataset


def approve_offline_fixture(dataset):
    """Only the temporary test gate; never approve a workspace dataset."""
    assert 'pytest-' in str(runner.REAL_CALL_GATE)
    runner.REAL_CALL_GATE.write_text(json.dumps({'state':'confirmed_by_user',
        'confirmed_dataset_hash':runner.dataset_hash(load_dataset(dataset)),
        'confirmed_at':'2026-11-13T00:00:00+00:00','user_instruction':'OFFLINE UNIT TEST ONLY'}))


def prices(models=None):
    return {"models": {model_id: {
        "input_cny_per_million": 2., "output_cny_per_million": 3.,
        "input_token_upper_bound": 1_048_576, "source_url": "https://example.org/official-test-pricing",
    } for model_id in (models or runner.MODEL_IDS)}}


def test_persistent_review_gate_blocks_real_new_and_resume_but_allows_mock(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    mid = runner.MODEL_IDS[0]
    probe = OfflineProbeProvider()
    kwargs = paid_kwargs(options, mid)
    prior = runner.run_multi(dataset, **kwargs, max_cost_cny=100, max_requests=0,
                             provider_overrides={mid: probe})
    gate = runner.REAL_CALL_GATE
    gate.write_text(json.dumps({"state": "awaiting_user_confirmation", "reason": "Test: annotation review pending"}))
    for resume in (None, prior['run_id']):
        with pytest.raises(ValueError, match="等待用户确认|确认门禁"):
            runner.run_multi(dataset, **kwargs, run_id=resume, max_cost_cny=100,
                             provider_overrides={mid: probe})
    assert not probe.calls
    assert json.loads(gate.read_text())['state'] == 'awaiting_user_confirmation'
    result = runner.run_multi(dataset, **options, mock=True, model_ids=[mid], max_requests=1)
    assert result['is_mock'] and result['executed'] == 1


@pytest.mark.parametrize('content', [None, '{invalid', '{}', '{"state":"ready"}', '{"state":"confirmed_by_user","confirmed_dataset_hash":"wrong"}'])
def test_missing_damaged_unknown_or_stale_confirmation_is_not_authorization(demo, content):
    dataset, options = demo
    manual_test_fixture(dataset)
    if content is None:
        runner.REAL_CALL_GATE.unlink()
    else:
        runner.REAL_CALL_GATE.write_text(content)
    with pytest.raises(ValueError, match='门禁|等待用户确认|hash不匹配'):
        runner.run_multi(dataset, **paid_kwargs(options, runner.MODEL_IDS[0]), max_cost_cny=100)
    assert not options['db_path'].exists()  # Refused before provider/store setup.


def test_confirmation_hash_is_canonical_full_dataset_and_subset_resume_is_bound(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    payload = json.loads(dataset.read_text())
    # Reformatting JSON is not a semantic annotation change.
    dataset.write_text(json.dumps(payload, ensure_ascii=False, indent=4))
    mid, probe = runner.MODEL_IDS[0], OfflineProbeProvider()
    kwargs = paid_kwargs(options, mid)
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=100, max_requests=1,
                             provider_overrides={mid:probe})
    run, _ = read_run(options, first['run_id'])
    full_hash = runner.dataset_hash(load_dataset(dataset))
    assert run['fairness_meta']['confirmed_dataset_hash'] == full_hash
    assert run['dataset_hash'] != full_hash  # Only two of eight samples selected.
    resumed = runner.run_multi(dataset, **kwargs, run_id=first['run_id'], max_requests=1,
                               provider_overrides={mid:probe})
    assert resumed['executed'] == 1
    # A new full-dataset confirmation cannot authorize this older snapshot,
    # even when the changed sample lies outside its two-sample subset.
    payload['samples'][-1]['note'] += '; new independent offline annotation'
    dataset.write_text(json.dumps(payload, ensure_ascii=False))
    approve_offline_fixture(dataset)
    with pytest.raises(ValueError, match='旧快照|已确认完整dataset'):
        runner.run_multi(dataset, **kwargs, run_id=first['run_id'],
                         provider_overrides={mid:probe})
    assert len(probe.calls) == 2


def test_prompt_revision_cannot_be_sent_during_old_run_resume(demo, monkeypatch):
    dataset, options = demo
    first = runner.run_multi(dataset, **options, mock=True, max_requests=1)
    _, before = read_run(options, first['run_id'])
    monkeypatch.setattr(runner, 'PROMPT_VERSION', runner.PROMPT_VERSION + '-offline-test-revision')
    with pytest.raises(ValueError, match='提示词版本|新口径'):
        runner.run_multi(dataset, **options, mock=True, run_id=first['run_id'])
    assert read_run(options, first['run_id'])[1] == before


class OfflineProbeProvider(VisionProvider):
    name = "offline_test_probe"

    def __init__(self, hook=None):
        self.calls = []
        self.hook = hook

    def recognize(self, model, image_data_url, system_prompt, user_prompt, gen):
        if self.hook:
            self.hook(self, model)
        self.calls.append((model.id, system_prompt, user_prompt))
        return ProviderResult(ok=True, raw_text='{"persons":[]}', served_model=model.id + "#OFFLINE-TEST", http_status=200)


def inject_provider(monkeypatch, provider):
    import vcrb.cli as main_cli
    monkeypatch.setattr(main_cli, "_build_provider", lambda *_args, **_kwargs: provider)


def paid_kwargs(options, model_id):
    return {**options, "mock": False, "yes": True, "approval_note": "TEST ONLY: offline injected provider",
            "model_ids": [model_id], "pricing": prices([model_id]), "limit": 2}


def test_synthetic_demo_168_tasks_is_a_distinct_mock_and_resume_is_idempotent(demo):
    dataset, options = demo
    original = all_bytes(dataset.parent)
    result = runner.run_multi(dataset, **options, mock=True)
    assert result["is_mock"] is True and result["planned"] == result["executed"] == 168
    run, records = read_run(options, result["run_id"])
    assert run["prompt_version"] == PROMPT_VERSION
    assert "MOCK" in run["label"]
    assert run["fairness_meta"]["benchmark_protocol"] == "multi-v2"
    assert run["fairness_meta"]["is_mock"] and run["fairness_meta"]["synthetic_fixture"]
    assert len(records) == 168 and all(r["ok"] for r in records)
    assert Counter((r["model_id"], r["experiment"]) for r in records) == {
        (model, exp): 8 for model in runner.MODEL_IDS for exp in ("A", "B", "C")}
    assert all(r["served_model"].endswith("#MOCK") for r in records)
    assert all(r["raw"]["request"]["system"] == SYSTEM_PROMPT for r in records)
    assert all(r["latency_ms"] == 0 and r["usage"] == {} for r in records)
    raw_before = all_bytes(options["output_root"] / "raw")
    resumed = runner.run_multi(dataset, **options, mock=True, run_id=result["run_id"])
    assert resumed["executed"] == 0 and resumed["skipped"] == 168
    assert resumed["progress"]["total"] == 168
    assert all_bytes(options["output_root"] / "raw") == raw_before
    assert all_bytes(dataset.parent) == original


def test_request_cap_is_total_across_models_and_resume_only_adds_new_keys(demo):
    dataset, options = demo
    first = runner.run_multi(dataset, **options, mock=True, max_requests=5)
    assert first["planned"] == 168 and first["executed"] == first["progress"]["total"] == 5
    run, rows_before = read_run(options, first["run_id"])
    assert run["status"] == "limited"
    # Scheduler rotates the model queues before applying the global cap.
    assert len({r["model_id"] for r in rows_before}) == 5
    resumed = runner.run_multi(dataset, **options, mock=True, run_id=first["run_id"], max_requests=7)
    assert resumed["executed"] == 7 and resumed["progress"]["total"] == 12
    _, after = read_run(options, first["run_id"])
    old = {(r["model_id"], r["experiment"], r["sample_id"]): r["raw_json"] for r in rows_before}
    assert len({(r["model_id"], r["experiment"], r["sample_id"]) for r in after}) == 12
    assert all(old[k] == next(r["raw_json"] for r in after if (r["model_id"], r["experiment"], r["sample_id"]) == k) for k in old)
    frozen_raw = all_bytes(options["output_root"] / "raw")
    zero = runner.run_multi(dataset, **options, mock=True, run_id=first["run_id"], max_requests=0)
    assert zero["executed"] == 0 and zero["progress"]["total"] == 12
    assert all_bytes(options["output_root"] / "raw") == frozen_raw


def test_old_single_target_run_and_mock_real_cross_reuse_are_rejected(demo):
    dataset, options = demo
    store = RunStore(options["db_path"])
    store.create_run("old-v1", ["A", "B"], list(runner.MODEL_IDS), "v1", "old-hash", str(dataset), {})
    store.close()
    with pytest.raises(ValueError, match="旧单目标|v2|复用"):
        runner.run_multi(dataset, **options, run_id="old-v1", mock=True)
    created = runner.run_multi(dataset, **options, mock=True, max_requests=0)
    with pytest.raises(ValueError, match="真实|Mock|复用"):
        runner.run_multi(dataset, **options, run_id=created["run_id"], mock=False, yes=True)


def test_resume_rejects_changed_annotations_model_or_experiment_scope(demo):
    dataset, options = demo
    model = runner.MODEL_IDS[0]
    result = runner.run_multi(dataset, **options, model_ids=[model], experiments=("A",), mock=True, max_requests=1)
    common = {**options, "mock": True, "run_id": result["run_id"]}
    with pytest.raises(ValueError, match="范围|模型|实验"):
        runner.run_multi(dataset, **common, model_ids=[model], experiments=("A", "C"))
    with pytest.raises(ValueError, match="范围|模型|实验"):
        runner.run_multi(dataset, **common, model_ids=[runner.MODEL_IDS[1]], experiments=("A",))
    payload = json.loads(dataset.read_text())
    payload["samples"][0]["persons"][0]["note"] = "review changed"
    dataset.write_text(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(ValueError, match="标注|变化|输入"):
        runner.run_multi(dataset, **common, model_ids=[model], experiments=("A",))
    _, rows = read_run(options, result["run_id"])
    assert len(rows) == 1


def test_draft_real_gate_blocks_before_any_provider_and_mock_does_not_approve(demo):
    dataset, options = demo
    manual_test_fixture(dataset, approved=False)
    source = dataset.read_bytes()
    with pytest.raises(ValueError, match="门禁|复核|approved|foreground"):
        runner.run_multi(dataset, **paid_kwargs(options, runner.MODEL_IDS[0]), max_cost_cny=100)
    # Draft workflow testing must preserve draft status rather than asserting GT.
    mock = runner.run_multi(dataset, **options, model_ids=[runner.MODEL_IDS[0]], mock=True, max_requests=1)
    assert mock["executed"] == 1 and dataset.read_bytes() == source
    run, rows = read_run(options, mock["run_id"])
    frozen = load_dataset(Path(run["dataset_path"]))
    assert frozen[0].review_status == "draft" and frozen[0].persons[0].identity_status == "needs_review"
    assert json.loads(rows[0]["raw"]["response"]["raw_text"]) == {"persons": []}


def test_synthetic_fixture_is_rejected_from_real_path_even_when_approved(demo):
    dataset, options = demo
    approve_offline_fixture(dataset)
    with pytest.raises(ValueError, match="synthetic|合成|Mock|MOCK"):
        runner.run_multi(dataset, **paid_kwargs(options, runner.MODEL_IDS[0]), max_cost_cny=100)


@pytest.mark.parametrize("yes,note", [(False, "approved"), (True, ""), (True, "   ")])
def test_explicit_approval_and_note_are_required_for_real_calls(demo, yes, note):
    dataset, options = demo
    manual_test_fixture(dataset)
    kwargs = paid_kwargs(options, runner.MODEL_IDS[0])
    kwargs.update(yes=yes, approval_note=note)
    with pytest.raises(ValueError, match="yes|授权|approval|确认"):
        runner.run_multi(dataset, **kwargs, max_cost_cny=100)


def test_missing_prices_and_zero_budget_do_not_mean_free_calls(demo):
    dataset, options = demo
    manual_test_fixture(dataset)
    kwargs = paid_kwargs(options, runner.MODEL_IDS[0])
    kwargs["pricing"] = None
    with pytest.raises(ValueError, match="价格|费用|cost"):
        runner.run_multi(dataset, **kwargs, max_cost_cny=100)
    kwargs["pricing"] = prices([runner.MODEL_IDS[0]])
    with pytest.raises(ValueError, match="价格|费用|cost"):
        runner.run_multi(dataset, **kwargs, max_cost_cny=0)


@pytest.mark.parametrize("field,value", [
    ("input_cny_per_million", None), ("input_cny_per_million", True),
    ("output_cny_per_million", -1), ("output_cny_per_million", float("nan")),
    ("output_cny_per_million", float("inf")), ("input_token_upper_bound", 1000),
    ("input_token_upper_bound", True), ("source_url", "http://example.org/pricing"),
])
def test_uncertain_pricing_is_not_a_cost_upper_bound(field, value):
    model = runner.MODEL_IDS[0]
    configured = prices([model])
    configured["models"][model][field] = value
    assert runner.task_upper_cost(model, runner.load_settings(), configured) is None


def test_cost_upper_bound_includes_all_retries_and_output_budget():
    model = runner.MODEL_IDS[0]
    actual = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    assert actual == pytest.approx((1_048_576 * 2 + 100 * 3) / 1_000_000 * 2)


def test_budget_is_reserved_durably_before_calls_and_cannot_grow_on_resume(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    def before_call(provider, _model):
        with sqlite3.connect(options["db_path"]) as conn:
            row = conn.execute("select fairness_meta from runs where status='running'").fetchone()
        reservation = json.loads(row[0])["cost_reservation"]
        assert reservation["reserved_tasks"] >= len(provider.calls) + 1
        assert reservation["reserved_cny"] >= (len(provider.calls) + 1) * unit - 1e-9
    probe = OfflineProbeProvider(before_call)
    inject_provider(monkeypatch, probe)
    kwargs = paid_kwargs(options, model)
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 2.1, max_requests=1)
    assert first["executed"] == 1
    second = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], max_cost_cny=unit * 100, max_requests=5)
    assert second["executed"] == 1 and second["progress"]["total"] == 2
    stopped = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], max_cost_cny=unit * 100)
    assert stopped["executed"] == 0 and len(probe.calls) == 2
    run, rows = read_run(options, first["run_id"])
    reservation = run["fairness_meta"]["cost_reservation"]
    assert reservation["reserved_tasks"] == 2 and reservation["reserved_cny"] == pytest.approx(unit * 2)
    assert run["fairness_meta"]["max_cost_cny"] == pytest.approx(unit * 2.1)
    assert run["status"] == "limited" and len(rows) == 2


def test_budget_below_one_task_makes_no_request(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    probe = OfflineProbeProvider()
    inject_provider(monkeypatch, probe)
    result = runner.run_multi(dataset, **paid_kwargs(options, model), max_cost_cny=unit * .9)
    assert result["executed"] == 0 and result["progress"]["total"] == 0 and not probe.calls
    run, _ = read_run(options, result["run_id"])
    assert run["fairness_meta"]["cost_reservation"]["reserved_cny"] == 0


def test_interruption_keeps_prior_response_and_does_not_refund_uncertain_call(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    interrupted = False
    def interrupt_second(provider, _model):
        nonlocal interrupted
        if len(provider.calls) == 1 and not interrupted:
            # The first successful response has already been committed.
            with sqlite3.connect(options["db_path"]) as conn:
                assert conn.execute("select count(*) from records").fetchone()[0] == 1
            interrupted = True
            raise KeyboardInterrupt
    probe = OfflineProbeProvider(interrupt_second)
    inject_provider(monkeypatch, probe)
    kwargs = paid_kwargs(options, model)
    first = runner.run_multi(dataset, **kwargs, max_cost_cny=unit * 3.1, max_requests=2)
    assert first["executed"] == first["progress"]["total"] == 1
    run, before = read_run(options, first["run_id"])
    assert run["status"] == "interrupted"
    assert run["fairness_meta"]["cost_reservation"]["reserved_cny"] == pytest.approx(unit * 2)
    resumed = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], max_cost_cny=unit * 3.1)
    assert resumed["executed"] == 1 and resumed["progress"]["total"] == 2
    run, after = read_run(options, first["run_id"])
    assert run["fairness_meta"]["cost_reservation"]["reserved_tasks"] == 3
    assert any(row["raw_json"] == before[0]["raw_json"] for row in after)


def test_abc_inputs_share_ab_image_and_c_never_exposes_gt_counts_or_boxes(demo, tmp_path):
    dataset, _options = demo
    sample = load_dataset(dataset)[3]
    work = next(w for w in load_works() if w.slug == sample.work_slug)
    builder = runner.InputBuilder(dataset.parent, tmp_path / "marked")
    before = (dataset.parent / sample.image_path).read_bytes()
    a, image_a, meta_a = builder(sample, "A", work)
    b, image_b, meta_b = builder(sample, "B", work)
    c, image_c, meta_c = builder(sample, "C", work)
    assert image_a == image_b and meta_a == meta_b
    assert b == f"这张图片出自{work.media_type}《{work.name}》。\n" + a
    assert work.name not in a and work.name in b and work.name in c
    assert meta_a["annotation_mode"] == "numbered" and meta_c["annotation_mode"] == "unmarked"
    assert image_c == encode_for_request(dataset.parent / sample.image_path)[0]
    assert image_a != image_c
    assert all(p.character not in a + b + c for p in sample.persons)
    changed = deepcopy(sample)
    changed.persons = changed.persons[:1]
    changed.persons[0].bbox = [.213456, .312345, .678912, .891234]
    changed.persons[0].character = "绝不进入输入的真实GT测试哨兵"
    changed.persons[0].person_id = 87
    independent = runner.InputBuilder(dataset.parent, tmp_path / "other-marked")
    c_changed, image_c_changed, _ = independent(changed, "C", work)
    assert c_changed == c and image_c_changed == image_c
    assert "87" not in c and ".213456" not in c and "测试哨兵" not in c
    assert (dataset.parent / sample.image_path).read_bytes() == before


def test_frozen_snapshot_image_tampering_blocks_resume(demo):
    dataset, options = demo
    first = runner.run_multi(dataset, **options, mock=True, max_requests=1)
    run, _ = read_run(options, first["run_id"])
    snapshot = Path(run["dataset_path"])
    sample = load_dataset(snapshot)[0]
    image = snapshot.parent / sample.image_path
    original_source = (dataset.parent / sample.image_path).read_bytes()
    image.write_bytes(b"modified frozen image")
    with pytest.raises(ValueError, match="哈希|SHA256|输入|image"):
        runner.run_multi(dataset, **options, mock=True, run_id=first["run_id"])
    assert (dataset.parent / sample.image_path).read_bytes() == original_source


def test_resume_refuses_model_generation_parameter_drift(demo, monkeypatch):
    dataset, options = demo
    model = runner.MODEL_IDS[0]
    first = runner.run_multi(dataset, **options, mock=True, model_ids=[model], max_requests=1)
    previous_resolve = runner.resolve_models
    def changed_models(ids=None):
        return [replace(m, params={**m.params, "extra_body": {"thinking": {"type": "enabled"}}}) for m in previous_resolve(ids)]
    monkeypatch.setattr(runner, "resolve_models", changed_models)
    with pytest.raises(ValueError, match="参数|模型|冻结|配置"):
        runner.run_multi(dataset, **options, mock=True, model_ids=[model], run_id=first["run_id"])


def test_database_execution_lease_prevents_two_schedulers(demo):
    import fcntl
    dataset, options = demo
    lock_path = options["db_path"].with_suffix(".execution.lock")
    with lock_path.open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="正在|并行|数据库"):
            runner.run_multi(dataset, **options, mock=True, max_requests=1)
    result = runner.run_multi(dataset, **options, mock=True, max_requests=1)
    assert result["executed"] == 1


def test_failed_request_budget_is_kept_and_retry_charged_again(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    unit = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    class FailFirstProbe(OfflineProbeProvider):
        def recognize(self, model, image_data_url, system_prompt, user_prompt, gen):
            result = super().recognize(model, image_data_url, system_prompt, user_prompt, gen)
            if len(self.calls) == 1:
                return ProviderResult(ok=False, error_type="timeout", attempts=2, error_message="offline test timeout")
            return result
    probe = FailFirstProbe()
    inject_provider(monkeypatch, probe)
    kwargs = paid_kwargs(options, model)
    first = runner.run_multi(dataset, **kwargs, max_requests=1, max_cost_cny=unit * 3.1)
    assert first["progress"] == {"total": 1, "ok": 0, "failed": 1}
    # An ordinary resume skips the failed key and spends one more task elsewhere.
    second = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], max_requests=1)
    assert second["progress"] == {"total": 2, "ok": 1, "failed": 1}
    retried = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], max_requests=1, retry_failed=True)
    assert retried["progress"] == {"total": 2, "ok": 2, "failed": 0}
    run, rows = read_run(options, first["run_id"])
    assert len(probe.calls) == 3 and len(rows) == 2
    assert run["fairness_meta"]["cost_reservation"]["reserved_tasks"] == 3
    assert run["fairness_meta"]["cost_reservation"]["reserved_cny"] == pytest.approx(unit * 3)
    raw_files = list((options["output_root"] / "raw").rglob("*.jsonl"))
    assert sum(len(p.read_text().splitlines()) for p in raw_files) == 3


def test_resume_uses_frozen_generation_and_retry_settings(demo, monkeypatch):
    dataset, options = demo
    model = runner.MODEL_IDS[0]
    first = runner.run_multi(dataset, **options, model_ids=[model], mock=True, max_requests=1)
    changed = runner.load_settings()
    changed["generation"] = {"temperature": .9, "max_tokens": 999}
    changed["retry"] = {"max_attempts": 9}
    monkeypatch.setattr(runner, "load_settings", lambda: deepcopy(changed))
    resumed = runner.run_multi(dataset, **options, model_ids=[model], mock=True, run_id=first["run_id"], max_requests=1)
    assert resumed["executed"] == 1
    run, rows = read_run(options, first["run_id"])
    assert run["fairness_meta"]["settings_snapshot"]["generation"] == {"temperature": .1, "max_tokens": 100}
    assert run["fairness_meta"]["settings_snapshot"]["retry"]["max_attempts"] == 2
    for row in rows:
        request = row["raw"]["request"]
        assert request.get("max_tokens", request.get("max_completion_tokens")) == 100
        assert request["temperature"] == .1


def test_unknown_resume_id_is_rejected_without_creating_or_calling(demo):
    dataset, options = demo
    with pytest.raises(ValueError, match="不存在|未知|run"):
        runner.run_multi(dataset, **options, mock=True, run_id="mistyped-resume-id", max_requests=1)
    store = RunStore(options["db_path"])
    try:
        assert store.latest_run_id() is None
        assert store.conn.execute("select count(*) from records").fetchone()[0] == 0
    finally:
        store.close()
    assert not options["output_root"].exists()


def test_lowercase_experiments_normalize_for_resume_and_budget_keys(demo, monkeypatch):
    dataset, options = demo
    manual_test_fixture(dataset)
    model = runner.MODEL_IDS[0]
    probe = OfflineProbeProvider()
    inject_provider(monkeypatch, probe)
    unit = runner.task_upper_cost(model, runner.load_settings(), prices([model]))
    kwargs = paid_kwargs(options, model)
    first = runner.run_multi(dataset, **kwargs, experiments=("a", "b", "c"), max_cost_cny=unit * 20)
    assert first["planned"] == first["executed"] == 6
    run_before, rows_before = read_run(options, first["run_id"])
    assert run_before["experiments"] == ["A", "B", "C"]
    assert {r["experiment"] for r in rows_before} == {"A", "B", "C"}
    # Lower/mixed case resumes must share keys and reserve no already-completed work.
    for experiments in [("a", "b", "c"), ("A", "b", "C")]:
        resumed = runner.run_multi(dataset, **kwargs, run_id=first["run_id"], experiments=experiments)
        assert resumed["executed"] == 0 and resumed["skipped"] == 6
    run_after, rows_after = read_run(options, first["run_id"])
    assert run_after["fairness_meta"]["cost_reservation"] == run_before["fairness_meta"]["cost_reservation"]
    assert len(probe.calls) == 6 and len(rows_after) == 6
    assert [r["raw_json"] for r in rows_after] == [r["raw_json"] for r in rows_before]
