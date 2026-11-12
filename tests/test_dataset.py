"""数据集 schema 测试（三维度独立字段 + 配额导出）。"""

from __future__ import annotations

import json

from tests.conftest import make_sample, write_dataset_json
from vcrb.dataset.schema import (
    DIFFICULTY_LEVELS,
    IMPORTANCE_LEVELS,
    Sample,
    dataset_stats,
    load_dataset,
    validate_dataset,
)


def test_valid_sample_no_errors():
    s = make_sample()
    assert s.validate() == []
    assert s.sampling_group == "main_character"


def test_sampling_group_derivation():
    assert make_sample(is_multi_person=True).sampling_group == "multi_person"
    assert (
        make_sample(visual_difficulty="side_face").sampling_group == "hard_case"
    )
    assert (
        make_sample(character_importance="key_supporting").sampling_group
        == "key_supporting_character"
    )
    # 多人优先于难度；难度优先于重要度
    s = make_sample(is_multi_person=True, visual_difficulty="blurry")
    assert s.sampling_group == "multi_person"
    s2 = make_sample(visual_difficulty="occluded", character_importance="supporting")
    assert s2.sampling_group == "hard_case"


def test_invalid_enum_values():
    s = make_sample(character_importance="lead", visual_difficulty="hard")
    errors = s.validate()
    assert any("character_importance" in e for e in errors)
    assert any("visual_difficulty" in e for e in errors)


def test_invalid_bbox():
    errors = make_sample(target_bbox=[0.5, 0.1, 0.2, 0.9]).validate()
    assert any("bbox" in e for e in errors)
    errors2 = make_sample(target_bbox=[-0.1, 0.0, 0.5, 1.5]).validate()
    assert any("bbox" in e for e in errors2)


def test_requires_verified_status():
    assert any("verified" in e for e in make_sample(annotation_status="pending").validate())


def test_gongfunvzu_work_category():
    s = make_sample(work_slug="gongfunvzu", sample_id="gongfunvzu_0001", target_character="待定")
    assert s.work_category == "recent_movie"


def test_enums_complete():
    assert IMPORTANCE_LEVELS == {"main", "key_supporting", "supporting"}
    assert DIFFICULTY_LEVELS == {
        "normal", "side_face", "occluded", "blurry", "costume_change",
    }


def test_load_dataset_roundtrip(tmp_path):
    samples = [make_sample(), make_sample(sample_id="kuangbiao_0002", target_character="安欣")]
    path = write_dataset_json(tmp_path, samples)
    loaded = load_dataset(path)
    assert [s.sample_id for s in loaded] == ["kuangbiao_0001", "kuangbiao_0002"]
    assert loaded[0].target_character == "高启强"


def test_validate_dataset_missing_image(tmp_path):
    samples = [make_sample(image_path="kuangbiao/nope.jpg")]
    errors, stats = validate_dataset(samples, tmp_path)
    assert any("图片不存在" in e for e in errors)
    assert stats["total"] == 1


def test_dataset_stats_groups():
    samples = [
        make_sample(),
        make_sample(sample_id="k2", target_character="安欣", character_importance="supporting"),
        make_sample(sample_id="k3", target_character="李响", is_multi_person=True),
    ]
    stats = dataset_stats(samples)
    assert stats["total"] == 3
    bucket = stats["by_work"]["kuangbiao"]
    assert bucket["main_character"] == 1
    assert bucket["supporting_character"] == 1
    assert bucket["multi_person"] == 1
    assert stats["characters_per_work"]["kuangbiao"] == ["安欣", "李响", "高启强"]
