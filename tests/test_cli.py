"""CLI 测试（离线子命令 + dry-run 红线）。"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vcrb.cli import app

runner = CliRunner()


def test_models_list_defaults():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "deepseek-flash" in result.output
    assert "kimi-k3.1-pro" in result.output
    assert "glm-5.3-flash" in result.output
    assert "qwen3.8-flash" in result.output
    # 不支持视觉的模型有明确提示
    assert "不支持视觉" in result.output


def test_models_list_all_includes_optional():
    result = runner.invoke(app, ["models", "list", "--all"])
    assert result.exit_code == 0
    assert "glm-4v-flash" in result.output  # 可选免费档
    assert "deepseek-v4-pro" in result.output  # 纯文本登记


def test_models_verify_dry_run_by_default():
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    # 未 --yes 不应有任何真实调用提示表以外的动作
    assert "连通性检查计划" in result.output


@pytest.mark.parametrize("issue", ["empty_final_content", "truncated_output"])
def test_verify_rejects_incomplete_http_success(tmp_path, monkeypatch, issue):
    from vcrb import cli
    from vcrb.providers import ProviderResult
    from vcrb.runner import store as store_module

    store = store_module.RunStore(tmp_path / "verify.db")

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def recognize(self, *args):
            return ProviderResult(ok=True, raw_text='', output_issue=issue,
                                  reasoning_text='{"character_name":"高启强"}')

    monkeypatch.setattr(cli, "get_api_key", lambda name: "offline-test-key")
    monkeypatch.setattr(cli, "OpenAICompatProvider", FakeProvider)
    monkeypatch.setattr(cli.paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(store_module, "RunStore", lambda: store)
    result = runner.invoke(app, ["models", "verify", "--models", "deepseek-flash", "--yes"])
    assert result.exit_code == 0, result.output
    verify = store.latest_verify()["deepseek-flash"]
    assert not verify["ok"]
    assert verify["error_type"] == issue
    assert verify["raw_text_head"] == ''


def test_small_run_scores_selected_snapshot_only(tmp_path, monkeypatch):
    from tests.conftest import build_tiny_dataset
    from vcrb import cli
    from vcrb.runner import executor, store as store_module

    dataset_dir = tmp_path / "dataset"
    samples, _ = build_tiny_dataset(dataset_dir, n_per_work=3)
    from dataclasses import replace

    originals = {sample.work_slug: sample for sample in samples}
    samples.extend(replace(sample, sample_id=f"{slug}_9999")
                   for slug, sample in originals.items())
    dataset_path = dataset_dir / "dataset.json"
    dataset_path.write_text(json.dumps({"samples": [s.to_dict() for s in samples]}))
    store = store_module.RunStore(tmp_path / "runs.db")
    monkeypatch.setattr(store_module, "RunStore", lambda: store)
    monkeypatch.setattr(cli, "DATASET_JSON", dataset_path)
    monkeypatch.setattr(cli.paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(executor, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(store_module, "RAW_RESPONSES_DIR", tmp_path / "raw")
    monkeypatch.setattr(cli, "SCORES_DIR", tmp_path / "scores")
    result = runner.invoke(app, ["run", "--small", "--models", "mock", "--yes"])
    assert result.exit_code == 0, result.output
    rid = store.latest_run_id()
    run = store.get_run(rid)
    selected = cli._load_run_samples(run)
    assert len(selected) == 4 < len(samples)  # 两个作品，每作品从3张取2张
    assert len(store.records(rid)) == 8
    result = runner.invoke(app, ["score", "--run-id", rid])
    assert result.exit_code == 0, result.output
    import csv

    with (tmp_path / "scores" / rid / "detail.csv").open() as handle:
        detail = list(csv.DictReader(handle))
    assert len(detail) == 8
    assert not any(r['outcome'] == 'not_run' for r in detail)
    mismatch = runner.invoke(app, ["run", "--models", "mock", "--run-id", rid, "--yes"])
    assert mismatch.exit_code == 1
    assert "快照不一致" in mismatch.output
    assert len(store.records(rid)) == 8


def test_roster_show():
    result = runner.invoke(app, ["roster", "show", "--work", "kuangbiao"])
    assert result.exit_code == 0
    assert "高启强" in result.output
    assert "张颂文" in result.output


def test_roster_show_unknown_work():
    result = runner.invoke(app, ["roster", "show", "--work", "nope"])
    assert result.exit_code == 1


def test_roster_show_replacement_work():
    result = runner.invoke(app, ["roster", "show", "--work", "qingyunian2"])
    assert result.exit_code == 0
    assert "范闲" in result.output and "张若昀" in result.output
    retired = runner.invoke(app, ["roster", "show", "--work", "langyabang"])
    assert retired.exit_code == 1


def test_collect_web_unknown_work():
    result = runner.invoke(app, ["collect", "web", "--work", "nope"])
    assert result.exit_code == 1


def test_run_requires_dataset(monkeypatch, tmp_path):
    # dataset.json 缺失 → 明确报错
    from vcrb.cli import DATASET_JSON
    from vcrb import paths

    monkeypatch.setattr("vcrb.cli.DATASET_JSON", tmp_path / "nope.json")
    result = runner.invoke(app, ["run", "--experiment", "A"])
    assert result.exit_code == 1
    assert "缺少" in result.output or "export-dataset" in result.output


def test_validate_dataset_missing():
    from vcrb.cli import DATASET_JSON

    if DATASET_JSON.exists():
        return  # 已有本地数据集的机器上跳过该用例
    result = runner.invoke(app, ["validate-dataset"])
    assert result.exit_code == 1


def test_review_list_without_runs(tmp_path, monkeypatch):
    from vcrb import paths
    from vcrb.runner.store import RunStore

    monkeypatch.setattr(paths, "RUNS_DB", tmp_path / "runs.db")
    # CLI 内部直接实例化 RunStore()（默认真实路径）；此处仅验证无 run 时的报错路径
    store = RunStore(tmp_path / "empty.db")
    assert store.latest_run_id() is None


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run", "score", "report", "clean", "annotate", "export-dataset", "status"]:
        assert cmd in result.output


@pytest.mark.parametrize(
    ("args", "group_target", "work_target"),
    [([], 3, 17), (["--target", "2", "--target-per-work", "12"], 2, 12)],
)
def test_export_uses_settings_and_explicit_overrides(
    tmp_path, monkeypatch, args, group_target, work_target
):
    from vcrb.dataset import store as store_module

    class FakeStore:
        def verified(self):
            return [{"work_slug": "kuangbiao", "attribution_status": "verified"}]

    received = {}

    def fake_export(store, works, out_dir, **kwargs):
        received.update(kwargs)
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps({"total": 1}), encoding="utf-8")
        return path

    monkeypatch.setattr(store_module, "AnnotationStore", FakeStore)
    monkeypatch.setattr(store_module, "export_dataset", fake_export)
    monkeypatch.setattr("vcrb.cli.load_settings", lambda: {
        "export": {"target_per_group": 3, "target_per_work": 17},
    })
    result = runner.invoke(app, ["export-dataset", *args])
    assert result.exit_code == 0, result.output
    assert received == {"target_per_group": group_target, "target_per_work": work_target}
    assert f"目标 {work_target}" in result.output


def test_screen_report_uses_configured_target(tmp_path, monkeypatch):
    from vcrb.dataset import store as store_module

    store = store_module.AnnotationStore(tmp_path / "annotations.db")
    monkeypatch.setattr(store_module, "AnnotationStore", lambda: store)
    monkeypatch.setattr("vcrb.cli.load_settings", lambda: {
        "export": {"target_per_group": 3, "target_per_work": 17},
    })
    result = runner.invoke(app, ["screen", "report", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert all(row["target_per_work"] == 17 and row["gap_to_target"] == 17 for row in rows)


def _audited_revision_fixture(tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.conftest import build_tiny_dataset
    from tests.test_scoring import _record
    from vcrb import cli
    from vcrb.runner import executor, store as store_module

    dataset_dir = tmp_path / "dataset"
    originals, _ = build_tiny_dataset(dataset_dir, n_per_work=2)
    originals = originals[:2]
    revised = [replace(originals[0], target_character="安欣", character_aliases=[]), originals[1]]
    dataset_path = dataset_dir / "dataset.json"
    dataset_path.write_text(json.dumps({"samples": [s.to_dict() for s in revised]}))
    input_path, evaluation_path, revision_path = (
        tmp_path / name for name in ("input_snapshot.json", "evaluation_snapshot.json", "revision.json")
    )
    input_path.write_text(json.dumps({"samples": [s.to_dict() for s in originals]}))
    evaluation_path.write_text(json.dumps({"samples": [s.to_dict() for s in revised]}))
    revision_path.write_text(json.dumps({"sample_id": originals[0].sample_id,
                                        "old_gt": "高启强", "new_gt": "安欣"}))
    metadata = {
        "dataset_snapshot_path": str(input_path),
        "evaluation_snapshot_path": str(evaluation_path),
        "evaluation_dataset_hash": executor.dataset_hash(revised),
        "annotation_revision_path": str(revision_path),
    }
    store = store_module.RunStore(tmp_path / "runs.db")
    run_id = "run_audited_revision"
    store.create_run(run_id, ["A"], ["mock"], "v1", executor.dataset_hash(originals),
                     str(input_path), metadata)
    raw_path = store.raw_file_for(run_id, "mock", "A", base=tmp_path / "raw")
    existing = _record(run_id, "mock", "A", originals[0].sample_id, raw_text=json.dumps({
        "character_name": "安欣", "is_uncertain": False, "reason": "offline fixture",
    }, ensure_ascii=False))
    existing["raw_file"] = str(raw_path)
    store.upsert_record(existing)
    store.append_raw(raw_path, json.loads(existing["raw_json"]))

    monkeypatch.setattr(store_module, "RunStore", lambda: store)
    monkeypatch.setattr(store_module, "RAW_RESPONSES_DIR", tmp_path / "raw")
    monkeypatch.setattr(cli, "DATASET_JSON", dataset_path)
    monkeypatch.setattr(cli.paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(executor, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "SCORES_DIR", tmp_path / "scores")
    monkeypatch.setattr(cli, "load_settings", lambda: {
        "generation": {"temperature": 0.1, "max_tokens": 64},
        "scoring": {"fuzzy_review_threshold": 0.9, "refusal_keywords": []},
        "runner": {"concurrency": 1},
    })
    return cli, executor, store, run_id, originals, revised, dataset_path, metadata, raw_path


def test_gt_revision_scores_offline_then_resumes_only_unfinished_input_keys(tmp_path, monkeypatch):
    from pathlib import Path
    from vcrb.providers.base import ProviderResult

    cli, executor, store, rid, originals, revised, dataset_path, metadata, raw_path = _audited_revision_fixture(tmp_path, monkeypatch)
    input_path = Path(metadata["dataset_snapshot_path"])
    input_before, raw_before = input_path.read_bytes(), raw_path.read_bytes()
    records_before = store.records(rid)

    def forbidden_provider(*args):
        raise AssertionError("Offline score must not construct a provider")

    monkeypatch.setattr(cli, "_build_provider", forbidden_provider)
    score = runner.invoke(app, ["score", "--run-id", rid])
    assert score.exit_code == 0, score.output
    summary = json.loads((tmp_path / "scores" / rid / "summary.json").read_text())
    assert summary["metrics"]["mock|A"]["counts"]["correct"] == 1
    assert summary["dataset_hash"] == metadata["evaluation_dataset_hash"]
    assert summary["input_dataset_hash"] == executor.dataset_hash(originals)
    assert summary["annotation_revision_path"] == metadata["annotation_revision_path"]
    assert raw_path.read_bytes() == raw_before
    assert store.records(rid) == records_before

    calls, dispatched_samples = [], []

    class FakeProvider:
        def recognize(self, *args):
            calls.append(args)
            return ProviderResult(ok=True, raw_text=json.dumps({
                "character_name": "陈书婷", "is_uncertain": False,
            }, ensure_ascii=False), http_status=200)

    monkeypatch.setattr(cli, "_build_provider", lambda *args: FakeProvider())
    original_execute = executor.execute_run

    def capture_original_snapshot(**kwargs):
        dispatched_samples.extend(kwargs["samples"])
        return original_execute(**kwargs)

    monkeypatch.setattr(executor, "execute_run", capture_original_snapshot)
    resumed = runner.invoke(app, ["run", "--models", "mock", "--experiment", "A", "--run-id", rid, "--yes"])
    assert resumed.exit_code == 0, resumed.output
    assert len(calls) == 1
    assert [s.to_dict() for s in dispatched_samples] == [s.to_dict() for s in originals]
    assert len(store.records(rid)) == 2
    assert store.records(rid)[0] == records_before[0]
    assert input_path.read_bytes() == input_before
    assert store.get_run(rid)["dataset_hash"] == executor.dataset_hash(originals)
    assert store.get_run(rid)["fairness_meta"] == metadata


@pytest.mark.parametrize("change", ["unregistered_gt", "registered_bbox"])
def test_resume_rejects_unregistered_gt_and_even_registered_input_changes(tmp_path, monkeypatch, change):
    from dataclasses import replace

    cli, executor, store, rid, originals, revised, dataset_path, metadata, raw_path = _audited_revision_fixture(tmp_path, monkeypatch)
    if change == "unregistered_gt":
        metadata = {"dataset_snapshot_path": metadata["dataset_snapshot_path"]}
    else:
        revised[0] = replace(revised[0], target_bbox=[0.1, 0.1, 0.9, 0.9])
        payload = json.dumps({"samples": [s.to_dict() for s in revised]})
        dataset_path.write_text(payload)
        from pathlib import Path

        Path(metadata["evaluation_snapshot_path"]).write_text(payload)
        metadata["evaluation_dataset_hash"] = executor.dataset_hash(revised)
    store.conn.execute("UPDATE runs SET fairness_meta=? WHERE run_id=?", (json.dumps(metadata), rid))
    store.conn.commit()
    records_before = store.records(rid)

    class ForbiddenProvider:
        def recognize(self, *args):
            raise AssertionError("Changed inputs must not reach any paid provider")

    monkeypatch.setattr(cli, "_build_provider", lambda *args: ForbiddenProvider())
    result = runner.invoke(app, ["run", "--models", "mock", "--experiment", "A", "--run-id", rid, "--yes"])
    assert result.exit_code == 1, result.output
    assert ("未登记" if change == "unregistered_gt" else "target_bbox") in result.output
    assert store.records(rid) == records_before
