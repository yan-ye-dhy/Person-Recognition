from hashlib import sha256

import pytest
from PIL import Image

from vcrb.imageutils import draw_target_boxes, encode_for_request
from vcrb.multi.mock import create_demo_dataset
from vcrb.multi.runner import InputBuilder
from vcrb.multi.schema import load_dataset
from vcrb.config import load_works


def test_numbered_boxes_preserve_original_and_never_render_identity(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (900, 500), "white").save(source)
    original = source.read_bytes()
    people = [{"person_id": 1, "bbox": [.05, .05, .4, .9], "character": "secret identity"},
              {"person_id": 2, "bbox": [.55, .05, .95, .9], "character": "another secret"}]
    first = draw_target_boxes(source, tmp_path / "first.jpg", people)
    for person in people:
        person["character"] = "different identity must not affect pixels"
    second = draw_target_boxes(source, tmp_path / "second.jpg", people)
    assert first.read_bytes() == second.read_bytes()
    assert source.read_bytes() == original
    assert Image.open(first).size == (900, 500)
    assert first.read_bytes() != original
    with pytest.raises(ValueError, match="覆盖"):
        draw_target_boxes(source, source, people)


@pytest.mark.parametrize("people", [
    [{"person_id": 1, "bbox": [0, 0, 1, 1]}, {"person_id": 1, "bbox": [0, 0, .5, .5]}],
    [{"person_id": True, "bbox": [0, 0, 1, 1]}],
    [{"person_id": 1, "bbox": [0, 0, 2, 1]}],
    [{"person_id": 1, "bbox": [0, 0, float("nan"), 1]}],
])
def test_bad_annotations_never_write_partial_output(tmp_path, people):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (100, 100), "white").save(source)
    with pytest.raises(ValueError):
        draw_target_boxes(source, tmp_path / "invalid.jpg", people)
    assert not (tmp_path / "invalid.jpg").exists()


def test_ab_identical_encoded_image_c_raw_and_has_no_target_hints(tmp_path):
    dataset = create_demo_dataset(tmp_path / "fixture")
    sample = load_dataset(dataset)[-1]
    works = {w.slug: w for w in load_works()}
    builder = InputBuilder(dataset.parent, tmp_path / "marked")
    ap, ai, am = builder(sample, "A", works[sample.work_slug])
    bp, bi, bm = builder(sample, "B", works[sample.work_slug])
    cp, ci, cm = builder(sample, "C", works[sample.work_slug])
    assert ai == bi and am == bm
    assert bp.endswith(ap) and sample.work_name not in ap
    raw, _ = encode_for_request(dataset.parent / sample.image_path)
    assert ci == raw and ci != ai
    assert cm["annotation_mode"] == "unmarked"
    assert all(p.character not in ap + bp + cp for p in sample.persons)
    sample.persons = sample.persons[:1]
    changed = InputBuilder(dataset.parent, tmp_path / "newmarked")
    cp2, ci2, _ = changed(sample, "C", works[sample.work_slug])
    assert cp2 == cp and ci2 == ci
