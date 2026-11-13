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


# ------------------------------------------------------- sync_candidates

def _make_jpeg(path, color, size=(64, 48)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_sync_registers_and_is_idempotent(tmp_path):
    store = AnnotationStore(tmp_path / "db.sqlite")
    cand = tmp_path / "candidates" / "kuangbiao"
    _make_jpeg(cand / "cand_0001.jpg", (200, 30, 30))
    _make_jpeg(cand / "cand_0002.jpg", (30, 30, 200))
    (cand / "manifest.jsonl").write_text(
        json.dumps({"file": "cand_0001.jpg", "source_type": "official_still",
                    "source_url": "https://x/1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    r1 = store.sync_candidates("kuangbiao", cand)
    assert r1["registered"] == 2
    assert r1["skipped"] == 0
    r2 = store.sync_candidates("kuangbiao", cand)
    assert r2["registered"] == 0 and r2["skipped"] == 2
    rows = store.samples("kuangbiao")
    assert len(rows) == 2
    by_name = {r["filename"]: r for r in rows}
    assert by_name["cand_0001.jpg"]["source_type"] == "official_still"
    assert by_name["cand_0001.jpg"]["source_url"] == "https://x/1"
    assert by_name["cand_0001.jpg"]["width"] == 64
    assert by_name["cand_0002.jpg"]["source_type"] == "unknown"


def test_sync_renumber_preserves_annotation(tmp_path):
    """重跑 clean 重新编号：同内容换文件名 → relink 且标注保留。"""
    store = AnnotationStore(tmp_path / "db.sqlite")
    cand = tmp_path / "candidates" / "kuangbiao"
    _make_jpeg(cand / "cand_0001.jpg", (10, 200, 10))
    store.sync_candidates("kuangbiao", cand)
    row = store.samples("kuangbiao")[0]
    store.update_annotation(row["pk"], {"status": "verified", "target_character": "高启强"})

    # 模拟重编号：同一张图改名为 cand_0007.jpg（旧文件消失）
    (cand / "cand_0001.jpg").rename(cand / "cand_0007.jpg")
    r = store.sync_candidates("kuangbiao", cand)
    assert r["relinked"] == 1 and r["registered"] == 0
    rows = store.samples("kuangbiao")
    assert len(rows) == 1
    assert rows[0]["filename"] == "cand_0007.jpg"
    assert rows[0]["status"] == "verified"
    assert rows[0]["target_character"] == "高启强"


def test_sync_filename_collision_renames_on_disk(tmp_path):
    """文件名被另一张图占用 → 磁盘改名登记，两行互不干扰。"""
    store = AnnotationStore(tmp_path / "db.sqlite")
    cand = tmp_path / "candidates" / "kuangbiao"
    _make_jpeg(cand / "cand_0001.jpg", (0, 0, 0))
    store.sync_candidates("kuangbiao", cand)
    # 磁盘上 cand_0001.jpg 被替换为另一张图（重编号后同位置不同内容）
    (cand / "cand_0001.jpg").unlink()
    _make_jpeg(cand / "cand_0001.jpg", (255, 255, 255))
    r = store.sync_candidates("kuangbiao", cand)
    assert r["renamed"] == 1 and r["registered"] == 1
    rows = store.samples("kuangbiao")
    names = {row["filename"] for row in rows}
    assert len(names) == 2
    assert "cand_0001.jpg" in names  # 旧行保留（其文件已被 clean 重跑清掉）
    renamed = [n for n in names if n != "cand_0001.jpg"]
    assert len(renamed) == 1 and renamed[0].startswith("cand_0001_")
    assert (cand / renamed[0]).exists()  # 新图改名后在磁盘上可访问


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
    # 导出门禁：归属核验通过才可进入正式集
    store.set_attribution([rows[0]["pk"], rows[1]["pk"]], "verified",
                          ["测试用：可信域+作品标识"], actor="user")

    out_dir = tmp_path / "final"
    path = export_dataset(
        store, [make_work("kuangbiao")], out_dir=out_dir,
        candidates_dir=tmp_path / "candidates",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 2
    assert payload["target_per_group"] == 8
    assert payload["target_per_work"] == 40
    assert payload["gap_per_work"]["kuangbiao"] == 38
    assert payload["quota_per_group"]["kuangbiao"]["main_character"] == 1
    # 匿名化：文件名 = slug_000N.jpg，绝不含角色名
    for s in payload["samples"]:
        assert s["target_character"] not in s["image_path"]
        assert s["sample_id"] in s["image_path"]
        assert s["annotation_status"] == "verified"
    assert (out_dir / "kuangbiao" / "kuangbiao_0001.jpg").is_file()
    assert (out_dir / "kuangbiao" / "kuangbiao_0002.jpg").is_file()


def test_export_quarantines_unverified_attribution(tmp_path, make_image):
    """归属未核实（rejected/needs_review/pending）的样本不得进入正式集。"""
    import shutil as _shutil

    candidates = tmp_path / "candidates" / "kuangbiao"
    candidates.mkdir(parents=True)
    for name in ("cand_0001.jpg", "cand_0002.jpg", "cand_0003.jpg"):
        make_image(name=name)
        _shutil.move(tmp_path / name, candidates / name)

    store = AnnotationStore(tmp_path / "annotations.db")
    store.register_candidates(
        "kuangbiao",
        [CandidateItem("kuangbiao", n, 640, 360) for n in
         ("cand_0001.jpg", "cand_0002.jpg", "cand_0003.jpg")],
        None,
    )
    rows = store.samples("kuangbiao")
    for r, attr in zip(rows, ("verified", "rejected", "needs_review")):
        store.update_annotation(
            r["pk"],
            {"status": "verified", "target_character": "高启强",
             "character_importance": "main", "visual_difficulty": "normal",
             "is_multi_person": 0},
        )
        store.set_attribution([r["pk"]], attr, ["测试"], actor="user")

    path = export_dataset(
        store, [make_work("kuangbiao")], out_dir=tmp_path / "final",
        candidates_dir=tmp_path / "candidates",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 1  # 仅归属 verified 的那张
    assert payload["quarantined_not_exported"]["kuangbiao"] == 2


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


def _grouped_export_store(tmp_path, counts):
    """主角先全部登记，保证限额不能靠截取前 100 行满足五组覆盖。"""
    store = AnnotationStore(tmp_path / "annotations.db")
    candidates = tmp_path / "candidates" / "kuangbiao"
    attributes = [
        ("main", "normal", False),
        ("key_supporting", "normal", False),
        ("supporting", "normal", False),
        ("main", "side_face", False),
        ("main", "normal", True),
    ]
    index = 0
    for count, (importance, difficulty, multi) in zip(counts, attributes):
        for _ in range(count):
            index += 1
            name = f"cand_{index:04d}.jpg"
            _make_jpeg(candidates / name, (index % 255, 40, 80))
            store.register_candidates(
                "kuangbiao", [CandidateItem("kuangbiao", name, 64, 48)], None
            )
            row = store.samples("kuangbiao")[-1]
            store.update_annotation(row["pk"], {
                "status": "verified", "target_character": f"角色{index}",
                "character_importance": importance, "visual_difficulty": difficulty,
                "is_multi_person": multi,
            })
            store.set_attribution([row["pk"]], "verified", ["测试"], actor="user")
    return store


def test_export_caps_each_work_and_preserves_all_sampling_groups(tmp_path):
    store = _grouped_export_store(tmp_path, [60, 24, 24, 24, 24])
    # 在较早的主角候选里隔离一张，分层选择仍应凑齐每组8张。
    first = store.samples("kuangbiao")[0]
    store.set_attribution([first["pk"]], "needs_review", ["归属待复核"], actor="user")
    kwargs = dict(
        out_dir=tmp_path / "final", candidates_dir=tmp_path / "candidates",
    )
    path = export_dataset(store, [make_work()], **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 40
    assert set(payload["quota_per_group"]["kuangbiao"].values()) == {8}
    assert payload["gap_per_work"]["kuangbiao"] == 0
    assert set(payload["quota_shortfall"]["kuangbiao"].values()) == {0}
    assert payload["eligible_per_work"]["kuangbiao"] == 155
    assert payload["quarantined_not_exported"]["kuangbiao"] == 1
    assert "角色1" not in {sample["target_character"] for sample in payload["samples"]}
    assert len(store.samples("kuangbiao")) == 156
    assert len(list((tmp_path / "candidates" / "kuangbiao").glob("*.jpg"))) == 156
    first_export = payload["samples"]
    path = export_dataset(store, [make_work()], **kwargs)
    assert json.loads(path.read_text(encoding="utf-8"))["samples"] == first_export


def test_export_fills_group_shortfall_from_remaining_eligible_samples(tmp_path):
    store = _grouped_export_store(tmp_path, [92, 2, 2, 2, 2])
    path = export_dataset(
        store, [make_work()], out_dir=tmp_path / "final",
        candidates_dir=tmp_path / "candidates",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 40
    assert payload["quota_per_group"]["kuangbiao"]["main_character"] == 32
    assert payload["quota_per_group"]["kuangbiao"]["hard_case"] == 2
    assert payload["quota_shortfall"]["kuangbiao"]["hard_case"] == 6
    assert payload["gap_per_work"]["kuangbiao"] == 0


def test_export_obeys_custom_work_limit(tmp_path):
    store = _grouped_export_store(tmp_path, [4, 4, 4, 4, 4])
    path = export_dataset(
        store, [make_work()], out_dir=tmp_path / "final",
        candidates_dir=tmp_path / "candidates", target_per_group=2, target_per_work=10,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 10
    assert payload["target_per_work"] == 10
    assert set(payload["quota_per_group"]["kuangbiao"].values()) == {2}
