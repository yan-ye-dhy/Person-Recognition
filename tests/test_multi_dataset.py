"""Offline v2 annotation/migration and source-integrity regression tests."""

import hashlib
import json

import pytest
from PIL import Image

from vcrb.multi.migration import migrate_v1
from vcrb.multi.schema import IgnoreRegion, MultiSample, PersonAnnotation, dataset_stats, load_dataset, renumber_persons, validate_dataset
from vcrb.multi.store import ConcurrentModificationError, MultiAnnotationStore


def source_dataset(tmp_path):
    source = tmp_path / "final" / "dataset.json"
    source.parent.mkdir()
    image = source.parent / "kuangbiao" / "kuangbiao_0001.jpg"
    image.parent.mkdir()
    Image.new("RGB", (400, 200), "#225588").save(image)
    source.write_text(json.dumps({"samples": [{"sample_id": "kuangbiao_0001", "work_slug": "kuangbiao", "work_name": "狂飙", "work_category": "modern_drama", "image_path": "kuangbiao/kuangbiao_0001.jpg", "target_character": "安欣", "target_bbox": [0.5, 0.1, 0.9, 0.9], "image_width": 400, "image_height": 200, "annotation_status": "verified", "source_url": "/private/user/disk/cache/video.hls#t=14.250"}]}), encoding="utf-8")
    return source, image


def test_migration_retains_original_bytes_and_demotes_all_quality_gates(tmp_path):
    source, image = source_dataset(tmp_path)
    source_bytes, image_bytes = source.read_bytes(), image.read_bytes()
    destination = tmp_path / "v2" / "dataset.json"
    payload = migrate_v1(source, destination)
    samples = load_dataset(destination)
    sample = samples[0]
    assert sample.image_path == "../final/kuangbiao/kuangbiao_0001.jpg"
    assert sample.image_sha256 == hashlib.sha256(image_bytes).hexdigest()
    assert sample.review_status == "draft" and sample.foreground_complete is False
    assert sample.persons[0].character == "安欣"
    assert sample.persons[0].identity_status == "needs_review"
    assert sample.persons[0].review_status == "pending"
    assert sample.persons[0].evidence[0]["time_seconds"] == 14.25
    assert "/private/user" not in json.dumps(payload)
    errors, _ = validate_dataset(samples, destination.parent)
    assert not errors
    errors, _ = validate_dataset(samples, destination.parent, for_run=True)
    assert any("foreground_complete" in error for error in errors)
    assert source.read_bytes() == source_bytes and image.read_bytes() == image_bytes
    with pytest.raises(FileExistsError):
        migrate_v1(source, destination)


def test_unknown_verified_and_pending_have_different_semantics():
    unknown = PersonAnnotation(1, [0.1, 0.1, 0.4, 0.9], identity_status="unknown", review_status="approved")
    sample = MultiSample("s1", "kuangbiao", "狂飙", "modern_drama", "a.jpg", [unknown], review_status="approved", foreground_complete=True)
    assert not sample.validate(for_run=True)
    unknown.character = "猜测姓名"
    assert any("character=null" in error for error in sample.validate())
    unknown.character, unknown.identity_status = "安欣", "verified"
    unknown.evidence = [{"type": "v1_inheritance"}]
    assert any("independent" in error for error in sample.validate())
    unknown.evidence.append({"type": "user_hypothesis", "candidate": "安欣", "confirmed": False})
    unknown.evidence.append({"type": "model_consensus", "candidate": "安欣"})
    assert any("independent" in error for error in sample.validate())
    unknown.evidence.append({"type": "official_character_reference", "url": "https://example.org/official"})
    assert not sample.validate(for_run=True)
    unknown.identity_status = "needs_review"
    assert sample.validate(for_run=True)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-0.1, 0, 1, 1], [0, 0, float("nan"), 1], [False, 0, 1, 1], [0, 1]])
def test_invalid_geometry_is_rejected(box):
    assert PersonAnnotation(1, box).validate()


def test_store_audits_old_new_values_and_rejects_stale_writers(tmp_path):
    source, _ = source_dataset(tmp_path)
    destination = tmp_path / "v2/dataset.json"
    migrate_v1(source, destination)
    first, second = MultiAnnotationStore(destination), MultiAnnotationStore(destination)
    sample = first.get("kuangbiao_0001")
    original_id = sample.persons[0].annotation_id
    sample.persons.append(PersonAnnotation(2, [0.05, 0.1, 0.3, 0.9], identity_status="unknown"))
    saved = first.save_sample(sample, actor="user", reason="补充左侧可见人物，身份未知")
    assert [p.person_id for p in saved.persons] == [1, 2]
    assert saved.persons[1].annotation_id == original_id
    assert saved.persons[1].character == "安欣"
    audit = json.loads(first.audit_path.read_text().strip())
    assert len(audit["before"]["persons"]) == 1 and len(audit["after"]["persons"]) == 2
    assert audit["actor"] == "user" and "补充" in audit["reason"]
    with pytest.raises(ConcurrentModificationError):
        second.save_sample(second.get("kuangbiao_0001"), actor="user", reason="stale save")
    restored = MultiAnnotationStore(destination).get(sample.sample_id)
    assert restored.to_dict() == saved.to_dict()


def test_store_cannot_change_source_fields_or_save_invalid_approval(tmp_path):
    source, _ = source_dataset(tmp_path)
    destination = tmp_path / "v2/dataset.json"
    migrate_v1(source, destination)
    store = MultiAnnotationStore(destination)
    sample = store.get("kuangbiao_0001")
    sample.image_path = "../final/replacement.jpg"
    with pytest.raises(ValueError, match="source field"):
        store.save_sample(sample, actor="user", reason="cannot swap images")
    sample = store.get("kuangbiao_0001")
    sample.review_status = "approved"
    with pytest.raises(ValueError, match="foreground_complete"):
        store.save_sample(sample, actor="user", reason="premature approval")
    assert not store.audit_path.exists()


def test_ignore_regions_stats_and_image_hash_validation(tmp_path):
    source, image = source_dataset(tmp_path)
    destination = tmp_path / "v2/dataset.json"
    migrate_v1(source, destination)
    sample = load_dataset(destination)[0]
    sample.ignore_regions = [IgnoreRegion.from_dict([0.01, 0.01, 0.1, 0.15])]
    assert not sample.validate()
    stats = dataset_stats([sample])
    assert stats["ignore_regions"] == 1 and stats["needs_review_persons"] == 1 and stats["approved_samples"] == 0
    image.write_bytes(b"corrupted original")
    errors, _ = validate_dataset([sample], destination.parent)
    assert any("SHA256 mismatch" in error for error in errors)


def test_numbering_preserves_annotation_identity_and_name():
    right = PersonAnnotation(1, [0.7, 0.1, 0.9, 0.9], "安欣")
    left = PersonAnnotation(2, [0.1, 0.1, 0.4, 0.9], "李响")
    ordered = renumber_persons([right, left])
    assert [(p.person_id, p.character) for p in ordered] == [(1, "李响"), (2, "安欣")]
    assert right.annotation_id == ordered[1].annotation_id
