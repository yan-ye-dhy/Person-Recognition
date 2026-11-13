"""Offline, temporary-fixture tests for immutable GT-person stratification."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import socket

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi import runner
from vcrb.multi.mock import create_demo_dataset
from vcrb.multi.reports import save_reports
from vcrb.multi.schema import dataset_hash, load_dataset
from vcrb.multi.scoring import score_run
from vcrb.multi.stratification import (
    freeze_stratification, load_frozen_stratification, load_stratification,
    validate_stratification,
)
from vcrb.runner.store import RunStore
from tests.test_multi_scoring import sample, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stratification tests must never access network or providers")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    import vcrb.cli
    monkeypatch.setattr(vcrb.cli, "_build_provider", forbidden)


def tags_for(samples):
    return {"schema_version": "1.0", "dataset_hash": dataset_hash(samples), "persons": [
        {"sample_id": s.sample_id, "image_sha256": s.image_sha256,
         "person_id": p.person_id, "annotation_id": p.annotation_id,
         "face_orientation": "front" if p.person_id == 1 else "profile",
         "visibility": "clear" if p.person_id == 1 else "occluded_or_blurry",
         "reviewer": "unit-test", "note": "STRATIFICATION_DO_NOT_SEND_SECRET_TAG"}
        for s in samples for p in s.persons]}


def scoring_fixture(tmp_path):
    s1 = sample(image_sha256="a" * 64)
    s2 = deepcopy(s1)
    s2.sample_id, s2.image_path, s2.image_sha256 = "s2", "s2.jpg", "b" * 64
    s2.persons = s2.persons[:1]
    s2.persons[0].annotation_id = "s2_p1"
    s3 = deepcopy(s1)
    s3.sample_id, s3.image_path, s3.image_sha256 = "s3", "s3.jpg", "c" * 64
    s3.persons = s3.persons[1:]
    s3.persons[0].person_id, s3.persons[0].annotation_id = 1, "s3_p1"
    samples = [s1, s2, s3]
    tags = tags_for(samples)
    tags["persons"][-1]["face_orientation"] = "profile"
    store = RunStore(tmp_path / "temporary.sqlite")
    store.create_run("fixture", ["A", "B", "C"], ["offline"], "fixture-version",
                     dataset_hash(samples), str(tmp_path / "snapshot/dataset.json"),
                     {"benchmark_protocol": "multi-v2", "is_mock": True})
    matcher = CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [CharacterEntry("安欣"), CharacterEntry("李响")])})
    for exp in ("A", "B", "C"):
        persons = ([{"person_id": 1, "character": "安欣"}, {"person_id": 2, "character": None}]
                   if exp != "C" else [
                       {"bbox": s1.persons[0].bbox, "character": "安欣"},
                       {"bbox": s1.persons[1].bbox, "character": "安欣"},
                       {"bbox": [.42, .02, .5, .08], "character": "额外人物"}])
        for s, ok, raw in ((s1, 1, response(persons)), (s2, 0, "")):
            store.upsert_record({"run_id": "fixture", "model_id": "offline", "experiment": exp,
                                 "sample_id": s.sample_id, "work_slug": s.work_slug, "attempt": 1,
                                 "ok": ok, "http_status": 200 if ok else 503,
                                 "latency_ms": 0, "error_type": None if ok else "temporary_failure",
                                 "error_message": None, "served_model": "OFFLINE_FIXTURE", "usage": "{}",
                                 "raw_json": json.dumps({"response": {"raw_text": raw}}),
                                 "raw_file": "fixture-only", "created_at": "2026-11-13"})
    return samples, tags, store, matcher


@pytest.mark.parametrize("mutation,error", [
    (lambda t: t.update(dataset_hash="wrong"), "dataset_hash"),
    (lambda t: t["persons"][0].update(image_sha256="wrong"), "Stale"),
    (lambda t: t["persons"][0].update(annotation_id="another-person"), "Stale"),
    (lambda t: t["persons"][0].update(person_id=True), "integer"),
    (lambda t: t["persons"][0].update(sample_id="foreign"), "Foreign"),
    (lambda t: t["persons"].append(deepcopy(t["persons"][0])), "Duplicate"),
    (lambda t: t["persons"].pop(), "Incomplete"),
    (lambda t: t["persons"][0].update(face_orientation="guess"), "face_orientation"),
    (lambda t: t["persons"][0].update(visibility="unknown"), "visibility"),
])
def test_sidecar_rejects_stale_foreign_duplicate_or_incomplete_person_tags(mutation, error):
    samples = [sample(image_sha256="a" * 64)]
    payload = tags_for(samples)
    before = dataset_hash(samples)
    assert validate_stratification(payload, samples) == payload
    mutation(payload)
    with pytest.raises(ValueError, match=error):
        validate_stratification(payload, samples)
    assert dataset_hash(samples) == before


def test_person_strata_keep_failure_denominators_and_never_assign_pose_to_extra_predictions(tmp_path):
    samples, tags, store, matcher = scoring_fixture(tmp_path)
    before_meta = deepcopy(store.get_run("fixture")["fairness_meta"])
    result = score_run(store, "fixture", samples, matcher, {}, stratification=tags)
    for exp in ("A", "B"):
        front = result.by_face_orientation[f"offline|{exp}"]["front"]
        profile = result.by_face_orientation[f"offline|{exp}"]["profile"]
        assert front["sample_count"] == front["person_targets"] == 2
        assert front["person_accuracy"]["numerator"] == front["person_accuracy"]["denominator"] == 1
        assert front["person_accuracy_e2e"]["value"] == .5
        assert front["api_failure_persons"] == 1 and front["not_run_persons"] == 0
        assert profile["person_accuracy_e2e"]["denominator"] == 2 and profile["not_run_persons"] == 1
    c = result.by_face_orientation["offline|C"]
    assert c["front"]["localization_recall"]["value"] == .5
    assert c["front"]["end_to_end_recall"]["value"] == .5
    assert c["profile"]["localization_recall"]["value"] == .5
    assert c["profile"]["matched_identity_accuracy"]["value"] == 0
    assert c["profile"]["end_to_end_recall"]["value"] == 0
    assert sum(group["person_targets"] for group in c.values()) == 4
    assert result.metrics["offline|C"]["localization"]["fp"] == 1
    for groups in (result.by_face_orientation, result.by_visibility):
        for values in groups.values():
            for group in values.values():
                assert "image_exact_accuracy" not in group and "localization_precision" not in group
    extras = [p for r in result.records for p in r["person_details"] if p["identity_category"] == "extra"]
    assert extras and all("face_orientation" not in p and "visibility" not in p for p in extras)
    assert result.stratification["status"] == "explicit_posthoc"
    assert store.get_run("fixture")["fairness_meta"] == before_meta
    store.close()


def test_c_strata_exclude_incomplete_localization_and_unverified_name_truth(tmp_path):
    samples, tags, store, matcher = scoring_fixture(tmp_path)
    # A distinct temporary run avoids modifying an existing frozen truth.
    samples[0].foreground_complete = False
    samples[1].persons[0].identity_status = "unknown"
    samples[1].persons[0].character = None
    store.create_run("other", ["C"], ["offline"], "fixture-version", dataset_hash(samples),
                     str(tmp_path / "another/dataset.json"), {"benchmark_protocol": "multi-v2", "is_mock": True})
    tags["dataset_hash"] = dataset_hash(samples)
    result = score_run(store, "other", samples, matcher, {}, stratification=tags)
    front = result.by_face_orientation["offline|C"]["front"]
    assert front["localization_excluded_persons"] == 1
    assert front["localization_recall"]["denominator"] == 1  # Unknown still has a location.
    assert front["end_to_end_recall"]["denominator"] == 0
    assert front["end_to_end_recall"]["value"] is None
    assert front["matched_identity_accuracy"]["denominator"] == 0
    store.close()


def test_frozen_tags_are_required_for_new_metadata_but_legacy_runs_do_not_borrow_working_tags(tmp_path):
    samples, tags, store, matcher = scoring_fixture(tmp_path)
    legacy = score_run(store, "fixture", samples, matcher, {})
    assert legacy.by_face_orientation == legacy.by_visibility == {}
    assert legacy.stratification["status"] == "not_provided"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    frozen_meta = freeze_stratification(tags, samples, snapshot)
    # A sidecar sitting next to an old snapshot is not automatic permission to mix it in.
    assert score_run(store, "fixture", samples, matcher, {}).by_face_orientation == {}
    meta = deepcopy(store.get_run("fixture")["fairness_meta"])
    meta["stratification"] = frozen_meta
    store.conn.execute("UPDATE runs SET fairness_meta=? WHERE run_id='fixture'", (json.dumps(meta),))
    store.conn.commit()
    frozen = score_run(store, "fixture", samples, matcher, {})
    assert frozen.stratification["status"] == "frozen" and frozen.by_visibility
    with pytest.raises(ValueError, match="cannot be overridden"):
        score_run(store, "fixture", samples, matcher, {}, stratification=tags)
    path = snapshot / "stratification.json"
    changed = json.loads(path.read_text())
    changed["persons"][0]["visibility"] = "occluded_or_blurry"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SHA256"):
        score_run(store, "fixture", samples, matcher, {})
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_frozen_stratification(store.get_run("fixture"), samples)
    store.close()


def test_reports_export_person_level_strata_without_pose_image_exactness(tmp_path):
    samples, tags, store, matcher = scoring_fixture(tmp_path)
    result = score_run(store, "fixture", samples, matcher, {}, stratification=tags)
    files = save_reports(result, tmp_path / "report")
    summary = json.loads(files["summary.json"].read_text())
    assert summary["by_face_orientation"]["offline|A"]["front"]["person_targets"] == 2
    for name in ("by_face_orientation.csv", "by_visibility.csv"):
        with files[name].open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 6
        assert {r["experiment"] for r in rows} == {"A", "B", "C"}
        assert "image_exact_accuracy_denominator" not in rows[0]
        assert "localization_recall_denominator" in rows[0]
        assert "person_accuracy_e2e_denominator" in rows[0]
    html = files["index.html"].read_text()
    assert "人物朝向" in html and "明显遮挡/模糊" in html
    assert "不强行分组" in html and "不可相加" in html
    store.close()


def test_runner_freezes_subset_sidecar_and_ignores_later_working_labels(tmp_path):
    dataset = create_demo_dataset(tmp_path / "fixture")
    samples = load_dataset(dataset)
    sidecar = dataset.parent / "stratification.json"
    sidecar.write_text(json.dumps(tags_for(samples), ensure_ascii=False))
    source_bytes = sidecar.read_bytes()
    options = {"mock": True, "db_path": tmp_path / "runs.sqlite", "output_root": tmp_path / "outputs",
               "model_ids": [runner.MODEL_IDS[0]], "limit": 2, "max_requests": 1}
    result = runner.run_multi(dataset, **options)
    store = RunStore(options["db_path"])
    run = store.get_run(result["run_id"])
    metadata = run["fairness_meta"]["stratification"]
    frozen = Path(run["dataset_path"]).parent / "stratification.json"
    frozen_bytes = frozen.read_bytes()
    selected = load_dataset(Path(run["dataset_path"]))
    assert metadata["dataset_hash"] == dataset_hash(selected) != dataset_hash(samples)
    assert metadata["sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
    assert len(load_frozen_stratification(run, selected)["persons"]) == metadata["person_labels"]
    assert sidecar.read_bytes() == source_bytes
    for record in store.records(run["run_id"]):
        assert "STRATIFICATION_DO_NOT_SEND_SECRET_TAG" not in json.dumps(record["raw"]["request"])
    sidecar.write_text('{"working_labels":"changed after freeze"}')
    runner.run_multi(dataset, **(options | {"max_requests": 0, "run_id": run["run_id"]}))
    assert frozen.read_bytes() == frozen_bytes
    frozen.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="SHA256"):
        runner.run_multi(dataset, **(options | {"max_requests": 0, "run_id": run["run_id"]}))
    store.close()
