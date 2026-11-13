"""Runner 测试：Mock 全链路（幂等、断点续跑、raw 落盘、预检抽样）。"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from tests.conftest import build_tiny_dataset, make_sample
from vcrb.models import ModelSpec
from vcrb.providers import MockProvider
from vcrb.providers.base import ProviderResult
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
    b_rec = next(r for r in records if r["experiment"] == "B" and "qingyunian2" == r["work_slug"])
    assert "《庆余年第二季》" in b_rec["raw"]["request"]["user"]
    a_rec = next(r for r in records if r["experiment"] == "A" and "qingyunian2" == r["work_slug"])
    assert "庆余年第二季" not in a_rec["raw"]["request"]["user"]
    assert "庆余年第二季" not in a_rec["raw"]["request"]["system"]
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


class _ConcurrentProbe:
    """Offline provider that measures actual overlapping calls, not queue size."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0
        self.by_model = Counter()
        self.model_maximum = Counter()
        self.calls = []
        self.initialization_threads = []

    def _get_client(self):
        self.initialization_threads.append(threading.get_ident())

    def recognize(self, model, image, system, prompt, gen):
        with self.lock:
            self.calls.append((model.id, system, prompt))
            self.active += 1
            self.by_model[model.id] += 1
            self.maximum = max(self.maximum, self.active)
            self.model_maximum[model.id] = max(
                self.model_maximum[model.id], self.by_model[model.id]
            )
        try:
            time.sleep(0.02)
            return ProviderResult(
                ok=True,
                raw_text='{"character_name":null,"is_uncertain":true,"reason":"offline"}',
                reasoning_text="kept separately for audit",
                finish_reason="stop",
                http_status=200,
            )
        finally:
            with self.lock:
                self.active -= 1
                self.by_model[model.id] -= 1


def test_concurrent_run_is_bounded_balanced_and_idempotent(tmp_path):
    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    probe = _ConcurrentProbe()
    tasks = [(mock_model(name), probe) for name in ("m-a", "m-b", "m-c")]
    main_thread = threading.get_ident()
    original_upsert, original_append = store.upsert_record, store.append_raw

    def main_thread_upsert(record):
        assert threading.get_ident() == main_thread
        original_upsert(record)

    def main_thread_append(path, payload):
        assert threading.get_ident() == main_thread
        original_append(path, payload)

    store.upsert_record = main_thread_upsert
    store.append_raw = main_thread_append
    kwargs = dict(
        run_id=run_id, tasks=tasks, samples=samples, works_by_slug=works,
        experiments=["A", "B"], store=store,
        settings={"runner": {"concurrency": 4}}, dataset_dir=dataset_dir,
        marked_dir=tmp_path / "marked", raw_dir=tmp_path / "raw",
    )
    result = execute_run(**kwargs)
    expected = len(tasks) * 2 * len(samples)
    assert result.executed == result.ok == expected
    assert 1 < probe.maximum <= 4
    assert max(probe.model_maximum.values()) <= 2
    assert {row[0] for row in probe.calls[:3]} == {"m-a", "m-b", "m-c"}
    assert probe.initialization_threads == [main_thread] * 3
    assert len(store.records(run_id)) == expected
    raw_files = list((tmp_path / "raw").rglob("*.jsonl"))
    assert sum(len(p.read_text().splitlines()) for p in raw_files) == expected
    for record in store.records(run_id):
        assert record["raw"]["response"]["reasoning_text"] == "kept separately for audit"
        assert record["raw"]["response"]["output_issue"] is None
        request = json.dumps(record["raw"]["request"], ensure_ascii=False)
        for sample in samples:
            assert sample.target_character not in request

    # A completed paid key must never be submitted again, even in parallel mode.
    calls_before, initializations_before = len(probe.calls), len(probe.initialization_threads)
    resumed = execute_run(**kwargs)
    assert resumed.executed == 0 and resumed.skipped_existing == expected
    assert len(probe.calls) == calls_before
    assert len(probe.initialization_threads) == initializations_before


def test_concurrent_interrupt_drains_paid_calls_without_submitting_backlog(tmp_path):
    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    probe = _ConcurrentProbe()
    tasks = [(mock_model(name), probe) for name in ("m-a", "m-b", "m-c")]

    def interrupt_after_first(_done, _label):
        raise KeyboardInterrupt

    kwargs = dict(
        run_id=run_id, tasks=tasks, samples=samples, works_by_slug=works,
        experiments=["A", "B"], store=store,
        settings={"runner": {"concurrency": 3}}, dataset_dir=dataset_dir,
        marked_dir=tmp_path / "marked", raw_dir=tmp_path / "raw",
    )
    interrupted = execute_run(**kwargs, progress_cb=interrupt_after_first)
    assert interrupted.interrupted
    assert 1 <= len(probe.calls) <= 3
    assert interrupted.executed == len(store.records(run_id)) == len(probe.calls)
    assert store.get_run(run_id)["status"] == "interrupted"

    # Only genuinely unstarted keys remain for continuation.
    expected = len(tasks) * 2 * len(samples)
    resumed = execute_run(**kwargs)
    assert not resumed.interrupted
    assert len(probe.calls) == expected
    assert len(store.records(run_id)) == expected
    assert resumed.executed + interrupted.executed == expected


def test_concurrent_run_respects_individual_model_caps(tmp_path):
    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    probe = _ConcurrentProbe()
    models = [mock_model(name) for name in ("m-a", "m-b", "m-c")]
    settings = {
        "runner": {
            "concurrency": 6,
            "per_model_concurrency": {"m-a": 1, "m-b": 2},
        }
    }
    summary = execute_run(
        run_id=run_id, tasks=[(model, probe) for model in models],
        samples=samples, works_by_slug=works, experiments=["A", "B"],
        store=store, settings=settings, dataset_dir=dataset_dir,
        marked_dir=tmp_path / "marked", raw_dir=tmp_path / "raw",
    )
    assert summary.executed == summary.ok == len(models) * len(samples) * 2
    assert probe.model_maximum["m-a"] == 1
    assert probe.model_maximum["m-b"] == 2
    assert probe.model_maximum["m-c"] <= 2  # Unspecified models retain the default.
    assert 1 < probe.maximum <= 5  # Sum of effective caps, below the global limit.
    metadata = _fairness_meta(models, settings)
    assert metadata["execution"]["per_model_concurrency"] == {
        "m-a": 1, "m-b": 2, "m-c": 2,
    }


def test_run_audit_records_the_budget_parameter_actually_sent(tmp_path):
    from vcrb.models import ProviderSpec
    from vcrb.providers.openai_compat import OpenAICompatProvider

    dataset_dir, samples, works, store, run_id = _setup_run(tmp_path)
    models = [
        mock_model("default-budget"),
        replace(mock_model("completion-budget"), params={
            "token_limit_parameter": "max_completion_tokens",
        }),
    ]
    sent = {}

    def capture_request(**kwargs):
        sent[kwargs["model"]] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"character_name":null,"is_uncertain":true}',
                                        reasoning_content=None),
                finish_reason="stop",
            )],
            model=kwargs["model"], usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=capture_request),
    ))
    provider = OpenAICompatProvider(
        ProviderSpec("offline", "https://unused.invalid", "UNUSED_KEY"),
        client_factory=lambda: client,
    )
    result = execute_run(
        run_id=run_id, tasks=[(model, provider) for model in models],
        samples=samples[:1], works_by_slug=works, experiments=["A"],
        store=store, settings={"generation": {"max_tokens": 4096}},
        dataset_dir=dataset_dir, marked_dir=tmp_path / "marked", raw_dir=tmp_path / "raw",
    )
    assert result.ok == 2
    for record in store.records(run_id):
        expected = "max_tokens" if record["model_id"] == "default-budget" else "max_completion_tokens"
        other = "max_completion_tokens" if expected == "max_tokens" else "max_tokens"
        audit = record["raw"]["request"]
        actual = sent[record["model_id"]]
        assert audit[expected] == actual[expected] == 4096
        assert other not in audit and other not in actual
    metadata = _fairness_meta(models, {})["per_model_notes"]
    assert metadata["default-budget"]["token_limit_parameter"] == "max_tokens"
    assert metadata["completion-budget"]["token_limit_parameter"] == "max_completion_tokens"
