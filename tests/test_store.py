"""标注存储与匿名化导出测试。"""

from __future__ import annotations

import json

from tests.conftest import make_work
from vcrb.dataset.store import AnnotationStore, CandidateItem, export_dataset


def _register_two(store: AnnotationStore, slug: str = "kuangbiao"):
    items = [
        CandidateItem(slug, "cand_0001.jpg", 640, 360, source_type="official_still", source_url="https://x/1"),
        CandidateItem(slug, "cand_0002.jpg", 640, 360, source_type="web_image"),
    ]
    return store.register_candidates(slug, items, None)


def test_register_and_stats(tmp_path):
    store = AnnotationStore(tmp_path / "annotations.db")
    added = _register_two(store)
    assert added == 2
    assert _register_two(store) == 0  # 幂等
    stats = store.stats()
    assert stats["kuangbiao"]["pending"] == 2


def test_update_annotation_roundtrip(tmp_path):
    store = AnnotationStore(tmp_path / "db.sqlite")
    _register_two(store)
    row = store.samples("kuangbiao")[0]
    store.update_annotation(
        row["pk"],
        {
            "status": "verified",
            "target_character": "高启强",
            "character_importance": "main",
            "visual_difficulty": "side_face",
            "is_multi_person": 1,
            "character_aliases": ["强哥", "老高"],
            "tags": ["side_face"],
            "target_bbox": [0.1, 0.1, 0.5, 0.8],
        },
    )
    (updated,) = store.samples("kuangbiao", status="verified")
    assert updated["target_character"] == "高启强"
    assert updated["character_aliases"] == ["强哥", "老高"]
    assert updated["target_bbox"] == [0.1, 0.1, 0.5, 0.8]
    assert updated["is_multi_person"] == 1
    with_default = {"status": "invalid"}
    store.update_annotation(updated["pk"], with_default)
    assert store.samples("kuangbiao", status="invalid")


def test_update_annotation_rejects_unknown_field(tmp_path):
    store = AnnotationStore(tmp_path / "db.sqlite")
    _register_two(store)
    row = store.samples("kuangbiao")[0]
    try:
        store.update_annotation(row["pk"], {"pk": 99})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_export_dataset_anonymized(tmp_path, make_image):
    candidates = tmp_path / "candidates" / "kuangbiao"
    candidates.mkdir(parents=True)
    make_image(name="cand_0001.jpg", size=(640, 360))
    make_image(name="cand_0002.jpg", size=(640, 360))
    # make_image 写到 tmp_path 根目录，复制到候选目录
    import shutil as _shutil

    _shutil.move(tmp_path / "cand_0001.jpg", candidates / "cand_0001.jpg")
    _shutil.move(tmp_path / "cand_0002.jpg", candidates / "cand_0002.jpg")

    store = AnnotationStore(tmp_path / "annotations.db")
    _register_two(store)
    rows = store.samples("kuangbiao")
    store.update_annotation(
        rows[0]["pk"],
        {
            "status": "verified", "target_character": "高启强",
            "character_importance": "main", "visual_difficulty": "normal",
            "is_multi_person": 0, "character_aliases": ["强哥"],
            "target_bbox": [0.1, 0.1, 0.6, 0.9],
        },
    )
    store.update_annotation(
        rows[1]["pk"],
        {
            "status": "verified", "target_character": "安欣",
            "character_importance": "supporting", "visual_difficulty": "normal",
            "is_multi_person": 0, "target_bbox": None,
        },
    )

    out_dir = tmp_path / "final"
    path = export_dataset(
        store, [make_work("kuangbiao")], out_dir=out_dir,
        candidates_dir=tmp_path / "candidates", target_per_group=20,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 2
    assert payload["quota_per_group"]["kuangbiao"]["main_character"] == 1
    # 匿名化：文件名 = slug_000N.jpg，绝不含角色名
    for s in payload["samples"]:
        assert s["target_character"] not in s["image_path"]
        assert s["sample_id"] in s["image_path"]
        assert s["annotation_status"] == "verified"
    assert (out_dir / "kuangbiao" / "kuangbiao_0001.jpg").is_file()
    assert (out_dir / "kuangbiao" / "kuangbiao_0002.jpg").is_file()


def test_export_skips_unverified(tmp_path, make_image):
    candidates = tmp_path / "candidates" / "kuangbiao"
    candidates.mkdir(parents=True)
    make_image(name="cand_0001.jpg")
    import shutil as _shutil

    _shutil.move(tmp_path / "cand_0001.jpg", candidates / "cand_0001.jpg")
    store = AnnotationStore(tmp_path / "annotations.db")
    _register_two(store)  # 均为 pending
    path = export_dataset(
        store, [make_work("kuangbiao")], out_dir=tmp_path / "final",
        candidates_dir=tmp_path / "candidates",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 0
