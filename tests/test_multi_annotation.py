"""Geometry/controller tests exercise both canvas editing and persistence."""

import json

import pytest
from PIL import Image

from vcrb.multi.annotation_app import apply_canvas_objects, canvas_bbox, canvas_objects
from vcrb.multi.schema import IgnoreRegion, MultiSample, PersonAnnotation
from vcrb.multi.store import MultiAnnotationStore, atomic_write_json


def sample():
    return MultiSample("s1", "kuangbiao", "狂飙", "modern_drama", "a.jpg", persons=[PersonAnnotation(1, [0.5, 0.1, 0.8, 0.9], "安欣", "verified", "approved", evidence=[{"type": "human_review", "text": "independent source"}])], review_status="approved", foreground_complete=True)


def test_canvas_add_move_reassign_number_and_preserve_existing_identity():
    original = sample()
    objects = canvas_objects(original, 800, 400)["objects"]
    objects.append({"type": "rect", "left": 40, "top": 40, "width": 200, "height": 320})
    updated = apply_canvas_objects(original, objects, 800, 400)
    assert len(updated.persons) == 2
    assert updated.persons[0].character is None and updated.persons[0].identity_status == "needs_review"
    assert updated.persons[1].character == "安欣"
    assert updated.persons[1].annotation_id == original.persons[0].annotation_id
    assert original.review_status == "approved" and len(original.persons) == 1
    assert updated.review_status == "draft" and not updated.foreground_complete
    objects = canvas_objects(updated, 800, 400)["objects"]
    objects[1]["left"] = 440
    moved = apply_canvas_objects(updated, objects, 800, 400)
    assert moved.persons[1].bbox[0] == 0.55
    assert moved.persons[1].review_status == "pending"


def test_canvas_scaling_and_normalization_ignore_rotation():
    box = canvas_bbox({"type": "rect", "left": -10, "top": 10, "width": 100, "height": 100, "scaleX": 2, "scaleY": 0.5}, 200, 100)
    assert box == [0.0, 0.1, 0.95, 0.6]
    with pytest.raises(ValueError):
        canvas_bbox({"type": "rect", "angle": 30, "left": 0, "top": 0, "width": 20, "height": 30}, 100, 100)


def test_canvas_ignore_regions_do_not_change_person_identity():
    original = sample()
    objects = [{"type": "rect", "left": 20, "top": 20, "width": 40, "height": 30}]
    updated = apply_canvas_objects(original, objects, 200, 100, ignore=True)
    assert updated.persons[0].to_dict() == original.persons[0].to_dict()
    assert updated.ignore_regions[0].bbox == [0.1, 0.2, 0.3, 0.5]
    assert not updated.foreground_complete
    updated.ignore_regions[0].reason = "远景严重模糊的背景人物"
    assert updated.to_dict()["ignore_regions"][0]["reason"].startswith("远景")


def test_canvas_restoration_has_same_scaled_geometry():
    original = sample()
    original.ignore_regions.append(IgnoreRegion([0.01, 0.02, 0.1, 0.2], "远景背景"))
    restored = apply_canvas_objects(original, canvas_objects(original, 640, 320)["objects"], 640, 320)
    assert restored.persons[0].bbox == original.persons[0].bbox
    assert restored.persons[0].character == original.persons[0].character
    assert restored.persons[0].review_status == "approved"


def test_actual_annotation_app_add_edit_delete_save_restore(tmp_path):
    """Exercise Streamlit widgets against an isolated dataset, never the real GT."""
    from streamlit.testing.v1 import AppTest

    original = sample()
    image = tmp_path / "a.jpg"
    Image.new("RGB", (400, 200), "#225588").save(image)
    path = tmp_path / "dataset.json"
    atomic_write_json(path, {"schema_version": "2.0", "samples": [original.to_dict()]})
    code = f"from pathlib import Path\nfrom vcrb.multi.annotation_app import run_app\nrun_app(Path({str(path)!r}))\n"
    app = AppTest.from_string(code, default_timeout=20).run()
    assert not app.exception

    def button(label):
        return next(b for b in app.button if b.label == label)

    button("新增人物框（数值编辑）").click().run()
    assert not app.exception
    assert len([t for t in app.text_input if t.label == "角色姓名（未知请留空）"]) == 2
    new_name = next(t for t in app.text_input if t.label == "角色姓名（未知请留空）")
    new_name.set_value("李响").run()
    assert not app.exception
    next(t for t in app.text_input if t.label.startswith("本次修改原因")).set_value("离线UI测试：新增并保存前景人物")
    button("保存此图v2标注与审计").click().run()
    assert not app.exception
    saved = MultiAnnotationStore(path).get("s1")
    assert len(saved.persons) == 2
    assert [(p.person_id, p.character) for p in saved.persons] == [(1, "李响"), (2, "安欣")]
    assert not saved.foreground_complete and saved.review_status == "draft"

    button("删除人物 1").click().run()
    assert not app.exception
    assert len([t for t in app.text_input if t.label == "角色姓名（未知请留空）"]) == 1
    button("恢复此图已保存标注").click().run()
    assert not app.exception
    assert len([t for t in app.text_input if t.label == "角色姓名（未知请留空）"]) == 2
    assert [w.value for w in app.number_input if w.label == "x1"] == [0.1, 0.5]
    assert [w.value for w in app.number_input if w.label == "x2"] == [0.35, 0.8]
    assert MultiAnnotationStore(path).get("s1").to_dict() == saved.to_dict()


def test_editing_approved_geometry_invalidates_person_and_sample_approvals(tmp_path):
    from streamlit.testing.v1 import AppTest

    original = sample()
    Image.new("RGB", (400, 200), "#225588").save(tmp_path / "a.jpg")
    path = tmp_path / "dataset.json"
    atomic_write_json(path, {"schema_version": "2.0", "samples": [original.to_dict()]})
    app = AppTest.from_string(f"from pathlib import Path\nfrom vcrb.multi.annotation_app import run_app\nrun_app(Path({str(path)!r}))", default_timeout=20).run()
    next(w for w in app.number_input if w.label == "x1").set_value(0.45).run()
    assert not app.exception
    status = next(w for w in app.selectbox if w.label == "此框复核状态")
    assert status.value == "pending"
    assert all(not widget.value for widget in app.checkbox)
    assert MultiAnnotationStore(path).get("s1").review_status == "approved"  # draft edits have not saved
    next(button for button in app.button if button.label == "恢复此图已保存标注").click().run()
    assert not app.exception
    assert next(w for w in app.number_input if w.label == "x1").value == 0.5
    assert next(w for w in app.number_input if w.label == "x2").value == 0.8
    assert next(w for w in app.selectbox if w.label == "此框复核状态").value == "approved"


def test_deep_link_preview_and_multi_person_filter(tmp_path):
    from streamlit.testing.v1 import AppTest

    original = sample()
    multi = MultiSample("jianlai_0001", "jianlai", "剑来", "animation", "a.jpg", persons=[PersonAnnotation(1, [0.05, 0.1, 0.4, 0.9]), PersonAnnotation(2, [0.6, 0.1, 0.9, 0.9])])
    Image.new("RGB", (400, 200), "#225588").save(tmp_path / "a.jpg")
    path = tmp_path / "dataset.json"
    atomic_write_json(path, {"schema_version": "2.0", "samples": [original.to_dict(), multi.to_dict()]})
    app = AppTest.from_string(f"from pathlib import Path\nfrom vcrb.multi.annotation_app import run_app\nrun_app(Path({str(path)!r}))", default_timeout=20)
    app.query_params["sample"] = "jianlai_0001"
    app.query_params["view"] = "preview"
    app.run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "样本").value == "jianlai_0001"
    assert app.tabs[0].label == "编号与忽略区预览"
    assert not any(button.label == "应用画布修改到草稿" for button in app.button)
    next(c for c in app.checkbox if c.label == "只看多人框（≥2）").check().run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "样本").options == ["jianlai_0001"]
    next(b for b in app.button if b.label == "进入画布编辑").click().run()
    assert not app.exception
    assert app.tabs[0].label == "画布编辑"
    assert any(button.label == "应用画布修改到草稿" for button in app.button)
    app.query_params["sample"] = "s1"
    app.run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "样本").value == "s1"
    assert next(c for c in app.checkbox if c.label == "只看多人框（≥2）").value is False
    assert MultiAnnotationStore(path).get("s1").to_dict() == original.to_dict()
