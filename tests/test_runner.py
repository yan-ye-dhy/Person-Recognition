"""Runner 测试：Mock 全链路（幂等、断点续跑、raw 落盘、预检抽样）。"""

from __future__ import annotations

import json

from tests.conftest import build_tiny_dataset, make_sample
from vcrb.models import ModelSpec
from vcrb.providers import MockProvider
from vcrb.runner.executor import (
    dataset_hash,
    execute_run,
    select_pilot_samples,
    _fairness_meta,
)
from vcrb.runner.store import RunStore, new_run_id


def mock_model(model_id: str = "mock-a") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="mock",
        vendor="Mock",
        display_name=f"Mock {model_id}",
        official_series="测试桩",
        tier="standard",
        supports_vision="doc",
        enabled=True,
    )


def test_dataset_hash_stable():
    samples = [make_sample(), make_sample(sample_id="kuangbiao_0002")]
    assert dataset_hash(samples) == dataset_hash([make_sample(), make_sample(sample_id="kuangbiao_0002")])
    assert dataset_hash(samples) != dataset_hash([make_sample()])


def test_fairness_meta_contains_model_notes():
    meta = _fairness_meta([mock_model("m1")], {"generation": {"temperature": 0.1, "max_tokens": 512}})
    assert meta["prompt_version"] == "v1"
    assert "m1" in meta["per_model_notes"]
    assert "marking" in meta


def _setup_run(tmp_path):
    dataset_dir = tmp_path / "dataset"
    samples, works = build_tiny_dataset(dataset_dir, n_per_work=2)
    store = RunStore(tmp_path / "runs.db")
    run_id = new_run_id()
    store.create_run(
        run_id=run_id,
        experiments=["A", "B"],
        models=["mock-a"],
        prompt_version="v1",
        dataset_hash=dataset_hash(samples),
        dataset_path=str(dataset_dir),
        fairness_meta={},
    )
    return dataset_dir, samples, works, store, run_id


def test_execute_run_full_and_idempotent(tmp_path):
    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    provider = MockProvider()
    marked_dir = tmp_path / "marked"

    summary = execute_run(
        run_id=run_id,
        tasks=[(mock_model(), provider)],
        samples=samples,
        works_by_slug=works,
        experiments=["A", "B"],
        store=store,
        settings={"generation": {"temperature": 0.1, "max_tokens": 128}},
        dataset_dir=dataset_dir,
        marked_dir=marked_dir,
        raw_dir=tmp_path / "raw_responses",
    )
    n = len(samples)
    assert summary.executed == 2 * n
    assert summary.ok == 2 * n
    assert summary.api_failures == 0
    records = store.records(run_id)
    assert len(records) == 2 * n
    # marked 图生成且 A/B 复用同一份
    assert len(list(marked_dir.glob("*.jpg"))) == n
    # raw JSONL 落盘且与记录一致
    raw_files = list((tmp_path / "raw_responses").rglob("*.jsonl"))
    assert raw_files, "raw JSONL 未生成"
    assert sum(len(p.read_text(encoding='utf-8').splitlines()) for p in raw_files) == 2 * n
    # B 组 prompt 含作品名、A 组不含
    b_rec = next(r for r in records if r["experiment"] == "B" and "langyabang" == r["work_slug"])
    assert "《琅琊榜》" in b_rec["raw"]["request"]["user"]
    a_rec = next(r for r in records if r["experiment"] == "A" and "langyabang" == r["work_slug"])
    assert "琅琊榜" not in a_rec["raw"]["request"]["user"]
    assert "琅琊榜" not in a_rec["raw"]["request"]["system"]
    # 请求不含答案
    for r in records:
        assert "梅长苏" not in json.dumps(r["raw"]["request"], ensure_ascii=False)
        assert "高启强" not in json.dumps(r["raw"]["request"], ensure_ascii=False)

    # ---- 幂等：重跑全部跳过 ----
    summary2 = execute_run(
        run_id=run_id,
        tasks=[(mock_model(), MockProvider())],
        samples=samples,
        works_by_slug=works,
        experiments=["A", "B"],
        store=store,
        settings={"generation": {"temperature": 0.1, "max_tokens": 128}},
        dataset_dir=dataset_dir,
        marked_dir=marked_dir,
    )
    assert summary2.executed == 0
    assert summary2.skipped_existing == 2 * n


def test_execute_run_resume_after_failure(tmp_path):
    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    failing = MockProvider(script={"mock-a": [{"fail": "server_error"}]})
    model = mock_model()

    s1 = execute_run(
        run_id=run_id, tasks=[(model, failing)], samples=samples, works_by_slug=works,
        experiments=["A"], store=store,
        settings={"generation": {"temperature": 0.1, "max_tokens": 64}},
        dataset_dir=dataset_dir, marked_dir=tmp_path / "marked",
    )
    assert s1.api_failures == len(samples)
    prog = store.run_progress(run_id)
    assert prog["failed"] == len(samples) and prog["ok"] == 0

    # 续跑（默认不重试失败）→ 全部跳过
    s2 = execute_run(
        run_id=run_id, tasks=[(model, MockProvider())], samples=samples, works_by_slug=works,
        experiments=["A"], store=store,
        settings={"generation": {"temperature": 0.1, "max_tokens": 64}},
        dataset_dir=dataset_dir, marked_dir=tmp_path / "marked",
    )
    assert s2.executed == 0

    # --retry-failed → 重试并修复
    s3 = execute_run(
        run_id=run_id, tasks=[(model, MockProvider())], samples=samples, works_by_slug=works,
        experiments=["A"], store=store, retry_failed=True,
        settings={"generation": {"temperature": 0.1, "max_tokens": 64}},
        dataset_dir=dataset_dir, marked_dir=tmp_path / "marked",
    )
    assert s3.executed == len(samples)
    assert s3.ok == len(samples)
    prog3 = store.run_progress(run_id)
    assert prog3["ok"] == len(samples)


def test_select_pilot_samples_stratified():
    samples = []
    for i in range(8):
        samples.append(make_sample(sample_id=f"kuangbiao_{i:04d}", target_character="高启强",
                                   character_importance="main"))
    for i in range(4):
        samples.append(make_sample(sample_id=f"jianlai_{i:04d}", target_character="陈平安",
                                   work_slug="jianlai", is_multi_person=True))
    picked = select_pilot_samples(samples, per_work=2)
    by_work = {}
    for s in picked:
        by_work.setdefault(s.work_slug, []).append(s.sampling_group)
    assert len(by_work["kuangbiao"]) == 2
    assert len(by_work["jianlai"]) == 2
    # 每作品尽量覆盖不同组
    assert set(by_work["jianlai"]) == {"multi_person"}


def test_run_store_latest_and_verify(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    rid = new_run_id()
    store.create_run(rid, ["A"], ["m"], "v1", "h", "p", {})
    assert store.latest_run_id() == rid
    store.save_verify("m", True, served_model="m#1.0", raw_text_head='{"ok":1}')
    verify = store.latest_verify()
    assert verify["m"]["ok"] == 1
    assert verify["m"]["served_model"] == "m#1.0"
