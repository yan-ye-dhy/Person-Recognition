"""Offline Mock contracts; no real images, answers, credentials, or API calls."""
import base64
import copy
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter

import pytest
from PIL import Image, ImageDraw

from vcrb.models import ModelSpec
from vcrb.multi.mock import MultiMockProvider, build_demo_responses, create_demo_dataset, encoded_image_sha256
from vcrb.multi.prompts import SYSTEM_PROMPT, build_user_prompt
from vcrb.multi.schema import load_dataset, validate_dataset
from vcrb.providers.base import GenerationParams, VisionProvider


@pytest.fixture
def model():
    return ModelSpec("configured-test-model", "unused-vendor", "demo", "Demo", "Demo", "standard", "doc", True)


def data_url(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def request(provider, model, image, prompt):
    return provider.recognize(model, image, SYSTEM_PROMPT, prompt, GenerationParams())


def test_mock_scripts_match_encoded_bytes_and_exact_prompt_and_mark_results(model):
    image = data_url(b"encoded-request-jpeg")
    key = (hashlib.sha256(b"encoded-request-jpeg").hexdigest(), "exact prompt")
    response = {"persons": [{"person_id": 1, "character": "测试角色甲"}]}
    provider = MultiMockProvider({key: response})
    response["persons"][0]["character"] = "caller changed later"
    result = request(provider, model, image, "exact prompt")
    assert isinstance(provider, VisionProvider) and provider.is_mock is True
    assert result.ok and result.http_status == 200
    assert result.served_model == "configured-test-model#MOCK"
    assert result.latency_ms == 0 and result.usage == {} and result.reasoning_text == ""
    assert json.loads(result.raw_text)["persons"][0]["character"] == "测试角色甲"
    assert json.loads(request(provider, model, image, "exact prompt ").raw_text) == {"persons": []}
    assert json.loads(request(provider, model, data_url(b"different JPEG"), "exact prompt").raw_text) == {"persons": []}


def test_explicit_string_is_preserved_for_parser_failure_demos(model):
    image = data_url(b"jpeg")
    provider = MultiMockProvider({(encoded_image_sha256(image), "p"): '{"persons":['})
    assert request(provider, model, image, "p").raw_text == '{"persons":['


def test_unscripted_fallback_never_fabricates_names_or_c_detections(model):
    provider = MultiMockProvider()
    image = data_url(b"unrecognized-image")
    ab = json.loads(request(provider, model, image, "编号：3, 7, 3，9 请逐个识别").raw_text)
    assert ab == {"persons": [{"person_id": n, "character": None} for n in (3, 7, 9)]}
    assert json.loads(request(provider, model, image, '格式示意：{"person_id":1,"character":null}').raw_text) == {"persons": []}
    assert json.loads(request(provider, model, image, '自主定位，bbox；编号：1,2').raw_text) == {"persons": []}


@pytest.mark.parametrize("bad", ["https://example.org/a.jpg", "data:image/jpeg;base64,", "data:image/jpeg;base64,@@@", "data:image/png;base64,YWJj"])
def test_invalid_data_url_is_rejected_without_fetching(bad):
    with pytest.raises(ValueError, match="Mock"):
        encoded_image_sha256(bad)


def test_module_and_provider_need_no_openai_import_credentials_or_network():
    code = '''
import builtins, socket, sys
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    if name == "openai" or name.startswith("openai."):
        raise AssertionError("OpenAI client import forbidden")
    return original_import(name, *args, **kwargs)
builtins.__import__ = offline_import
def no_network(*args, **kwargs):
    raise AssertionError("Network forbidden")
socket.create_connection = no_network
socket.socket = no_network
from vcrb.multi.mock import MultiMockProvider
from vcrb.models import ModelSpec
from vcrb.providers.base import GenerationParams
p = MultiMockProvider()
m = ModelSpec("x", "none", "none", "x", "x", "standard", "doc", True)
r = p.recognize(m, "data:image/jpeg;base64,YWJj", "", "bbox", GenerationParams())
assert r.served_model == "x#MOCK" and r.raw_text == '{"persons": []}'
assert not any(n == "openai" or n.startswith("openai.") for n in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_demo_fixture_is_independent_complete_and_reproducible(tmp_path):
    destination = tmp_path / "demo"
    path = create_demo_dataset(destination)
    payload = json.loads(path.read_text())
    samples = load_dataset(path)
    assert payload["synthetic_mock_fixture"] and payload["mock"]
    assert "not real" in payload["notice"]
    assert Counter(s.work_slug for s in samples) == {"kuangbiao": 2, "qingyunian2": 2, "gongfunvzu": 2, "jianlai": 2}
    errors, stats = validate_dataset(samples, destination, for_run=True)
    assert errors == [] and stats["persons"] == 20 and stats["ignore_regions"] == 4
    assert {len(s.persons) for s in samples} == {1, 2, 3, 4}
    for sample in samples:
        assert sample.foreground_complete and sample.review_status == "approved"
        assert sample.evidence[0]["type"] == "synthetic"
        assert all(p.identity_status == "verified" and p.review_status == "approved" for p in sample.persons)
        assert all(p.character.startswith("测试角色") for p in sample.persons)
        assert not any(p.character in sample.image_path for p in sample.persons)
        with Image.open(destination / sample.image_path) as image:
            assert image.size == (960, 540) and image.format == "JPEG"
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert create_demo_dataset(destination) == path
    assert before == {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert not (tmp_path / "data").exists()


def test_demo_refuses_to_overwrite_existing_real_files(tmp_path):
    destination = tmp_path / "real"
    destination.mkdir()
    image = destination / "original.jpg"
    image.write_bytes(b"real image must remain untouched")
    with pytest.raises(FileExistsError):
        create_demo_dataset(destination)
    assert image.read_bytes() == b"real image must remain untouched"
    assert not (destination / "dataset.json").exists()


def demo_builder(directory):
    def build(sample, experiment, work):
        prompt = build_user_prompt(experiment, None if experiment == "A" else work)
        image = Image.open(directory / sample.image_path).convert("RGB")
        if experiment != "C":
            draw = ImageDraw.Draw(image)
            for p in sample.persons:
                draw.rectangle([v * (image.width if i % 2 == 0 else image.height) for i, v in enumerate(p.bbox)], outline="red", width=3)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=90)
        return prompt, data_url(encoded.getvalue()), {"mock": True}
    return build


def test_demo_script_drives_all_abc_tasks_and_preserves_input_images(tmp_path, model):
    path = create_demo_dataset(tmp_path / "demo")
    samples = load_dataset(path)
    builder = demo_builder(path.parent)
    hashes_before = {s.sample_id: hashlib.sha256((path.parent / s.image_path).read_bytes()).hexdigest() for s in samples}
    scripts = build_demo_responses(samples, path.parent, builder)
    assert len(scripts) == 24
    provider = MultiMockProvider(scripts)
    from vcrb.config import load_works
    works = {w.slug: w for w in load_works()}
    for sample in samples:
        for experiment in ("A", "B", "C"):
            prompt, image, _ = builder(sample, experiment, works[sample.work_slug])
            persons = json.loads(request(provider, model, image, prompt).raw_text)["persons"]
            assert len(persons) == len(sample.persons)
            assert [p["character"] for p in persons] == [p.character for p in sample.persons]
            if experiment == "C":
                assert [p["bbox"] for p in persons] == [p.bbox for p in sample.persons]
                assert all("person_id" not in p for p in persons)
            else:
                assert [p["person_id"] for p in persons] == list(range(1, len(persons) + 1))
                assert all("bbox" not in p for p in persons)
        assert hashes_before[sample.sample_id] == hashlib.sha256((path.parent / sample.image_path).read_bytes()).hexdigest()


def test_real_or_mixed_annotations_are_rejected_before_builder_is_called(tmp_path):
    path = create_demo_dataset(tmp_path / "demo")
    samples = load_dataset(path)
    samples[-1].evidence = [{"type": "official_character_reference"}]
    def never_called(*args):
        raise AssertionError("Builder must not see any real ground truth")
    with pytest.raises(ValueError, match="non-synthetic"):
        build_demo_responses(samples, path.parent, never_called)
    samples = load_dataset(path)
    samples[0].persons[0].character = "真实人物姓名"
    with pytest.raises(ValueError, match="invented roles"):
        build_demo_responses(samples, path.parent, never_called)


def test_tampered_synthetic_image_and_script_collisions_are_rejected(tmp_path):
    path = create_demo_dataset(tmp_path / "demo")
    samples = load_dataset(path)
    constant = lambda *_: ("same prompt", data_url(b"same encoded bytes"), {})
    with pytest.raises(ValueError, match="collision"):
        build_demo_responses(samples, path.parent, constant)
    (path.parent / samples[0].image_path).write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_demo_responses(samples, path.parent, demo_builder(path.parent))


def test_explicit_synthetic_unknown_identity_stays_null(tmp_path):
    path = create_demo_dataset(tmp_path / "demo")
    samples = copy.deepcopy(load_dataset(path))
    person = samples[0].persons[0]
    person.character, person.identity_status = None, "unknown"
    scripts = build_demo_responses(samples[:1], path.parent, demo_builder(path.parent))
    assert len(scripts) == 3
    assert all(reply["persons"][0]["character"] is None for reply in scripts.values())
