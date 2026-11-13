"""Explicitly synthetic, zero-network support for multi-person A/B/C demos.

Neither this provider nor its fallback reads annotations. Correct response scripts
can only be built explicitly from the separate synthetic fixture below. These
results exercise plumbing and must never be reported as model recognition scores.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..config import Work, load_works
from ..models import ModelSpec
from ..providers.base import GenerationParams, ProviderResult, VisionProvider
from .schema import IgnoreRegion, MultiSample, PersonAnnotation, SCHEMA_VERSION, load_dataset, validate_dataset

DEFAULT_DEMO_DIRECTORY = Path("results/multi/demo_fixture")
DEMO_NOTICE = "SYNTHETIC MOCK DEMO: not real images, identities, or model accuracy"
ResponseKey = tuple[str, str]
InputBuilder = Callable[[MultiSample, str, Work | None], tuple[str, str, dict[str, Any]]]


def encoded_image_sha256(data_url: str) -> str:
    """Hash exactly the JPEG bytes sent in the data URL, not the source file."""
    header, separator, payload = data_url.partition(",")
    if not separator or header.lower() not in {"data:image/jpeg;base64", "data:image/jpg;base64"}:
        raise ValueError("Mock expects a base64 JPEG data URL")
    try:
        encoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Mock JPEG data URL has invalid base64") from exc
    if not encoded:
        raise ValueError("Mock JPEG data URL is empty")
    return hashlib.sha256(encoded).hexdigest()


class MultiMockProvider(VisionProvider):
    """Look up explicit scripts by (encoded JPEG SHA256, exact user prompt)."""

    name = "multi_mock"
    is_mock = True

    def __init__(self, responses: Mapping[ResponseKey, dict[str, Any] | str] | None = None) -> None:
        self.responses: dict[ResponseKey, str] = {}
        for key, value in (responses or {}).items():
            if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(k, str) for k in key):
                raise TypeError("Mock response keys must be (JPEG SHA256, user prompt)")
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            if not isinstance(value, str):
                raise TypeError("Mock responses must be JSON dictionaries or raw strings")
            self.responses[key] = value

    def recognize(
        self, model: ModelSpec, image_data_url: str, system_prompt: str,
        user_prompt: str, gen: GenerationParams,
    ) -> ProviderResult:
        key = (encoded_image_sha256(image_data_url), user_prompt)
        raw = self.responses.get(key)
        if raw is None:
            # No image recognition, annotation lookup, or fabricated identity.
            # A C-style prompt gets no detections. A/B may explicitly list IDs;
            # format examples are not assumed to describe the actual image.
            ids: list[int] = []
            if "bbox" not in user_prompt:
                explicit = re.search(r"编号(?:列表)?\s*[:：]\s*(\d+(?:\s*[,，、\s]\s*\d+)*)", user_prompt)
                if explicit:
                    ids = sorted({int(n) for n in re.findall(r"\d+", explicit.group(1)) if int(n) > 0})
            raw = json.dumps({"persons": [{"person_id": i, "character": None} for i in ids]}, ensure_ascii=False)
        return ProviderResult(
            ok=True, raw_text=raw, finish_reason="stop", served_model=f"{model.id}#MOCK",
            latency_ms=0, usage={}, http_status=200,
        )


def create_demo_dataset(destination: Path | str = DEFAULT_DEMO_DIRECTORY) -> Path:
    """Create eight geometric images separately from all real/v1 datasets.

    Four existing work groups are used only to exercise report grouping. Every
    face, role, and identity in this directory is invented. A nonempty unrelated
    destination is rejected; rerunning on our validated fixture is read-only.
    """
    destination = Path(destination)
    dataset_path = destination / "dataset.json"
    if destination.exists() and any(destination.iterdir()):
        if dataset_path.is_file():
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            if payload.get("synthetic_mock_fixture") is True:
                samples = load_dataset(dataset_path)
                _require_synthetic(samples, destination)
                errors, _ = validate_dataset(samples, destination, for_run=True)
                if not errors and len(samples) == 8:
                    return dataset_path
        raise FileExistsError("Refusing to overwrite a nonempty destination outside this synthetic Mock fixture")
    works = load_works()
    expected = {"kuangbiao", "qingyunian2", "gongfunvzu", "jianlai"}
    works = [w for w in works if w.slug in expected]
    if len(works) != 4 or {w.slug for w in works} != expected:
        raise ValueError("The demo requires the four configured work groups")
    destination.mkdir(parents=True, exist_ok=True)
    samples: list[MultiSample] = []
    names = ["测试角色甲", "测试角色乙", "测试角色丙", "测试角色丁"]
    colors = ["#2f71b8", "#d47a29", "#578c48", "#8a5db5"]
    width, height = 960, 540
    for work_index, work in enumerate(works):
        for image_index in range(2):
            count = (work_index + image_index) % 4 + 1
            sample_id = f"synthetic_{work.slug}_{image_index + 1:02d}"
            relative = Path("images") / f"{sample_id}.jpg"
            image_path = destination / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (width, height), (231 - work_index * 8, 238 - image_index * 10, 245))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 42), fill="#172638")
            draw.text((14, 12), "SYNTHETIC MOCK FIXTURE - NOT A REAL BENCHMARK", font=ImageFont.load_default(size=18), fill="white")
            persons = []
            for person_index in range(count):
                center = .12 + (person_index + .5) * .76 / count
                half = min(.105, .30 / count)
                box = [round(center - half, 6), .20, round(center + half, 6), .94]
                x1, y1, x2, y2 = [int(v * (width if i % 2 == 0 else height)) for i, v in enumerate(box)]
                head = min((x2 - x1) * .33, 44)
                cx = (x1 + x2) / 2
                draw.ellipse((cx - head, y1, cx + head, y1 + head * 2), fill=colors[person_index])
                torso_top, torso_bottom = y1 + head * 2 + 5, y1 + 255
                draw.rectangle((x1, torso_top, x2, torso_bottom), fill=colors[person_index])
                draw.rectangle((x1, torso_bottom, cx - 5, y2), fill=colors[person_index])
                draw.rectangle((cx + 5, torso_bottom, x2, y2), fill=colors[person_index])
                persons.append(PersonAnnotation(
                    person_index + 1, box, names[person_index], "verified", "approved",
                    evidence=[{"type": "synthetic", "source": "programmatically drawn geometric fixture; invented role"}],
                    note="仅合成 Mock；不存在真实影视身份。",
                    annotation_id=hashlib.sha256(f"{sample_id}:{person_index + 1}".encode()).hexdigest()[:32],
                ))
            ignores = []
            if image_index == 1:
                # Tiny gray background silhouette is intentionally out of scope.
                box = [.90, .12, .96, .28]
                draw.ellipse((876, 65, 910, 96), fill="#adb3ba")
                draw.rectangle((864, 97, 921, 151), fill="#adb3ba")
                ignores = [IgnoreRegion(box, "合成远景背景轮廓，不参与前景人物计分", [{"type": "synthetic"}])]
            image.save(image_path, "JPEG", quality=92)
            samples.append(MultiSample(
                sample_id, work.slug, work.name, work.category, relative.as_posix(), persons, ignores,
                image_width=width, image_height=height,
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                review_status="approved", foreground_complete=True,
                evidence=[{"type": "synthetic", "source": "create_demo_dataset", "notice": DEMO_NOTICE}],
                note="合成 Mock 流程样例；作品名仅作分组，不代表图片出自该作品。",
            ))
    errors, stats = validate_dataset(samples, destination, for_run=True)
    if errors:
        raise ValueError(f"Synthetic fixture is invalid: {errors}")
    payload = {"schema_version": SCHEMA_VERSION, "synthetic_mock_fixture": True, "mock": True,
               "notice": DEMO_NOTICE, "total": len(samples), "stats": stats,
               "samples": [s.to_dict() for s in samples]}
    dataset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_path


def _require_synthetic(samples: Sequence[MultiSample], dataset_dir: Path) -> None:
    if not samples:
        raise ValueError("Synthetic Mock scripting requires nonempty synthetic samples")
    for sample in samples:
        if not any(e.get("type") == "synthetic" for e in sample.evidence):
            raise ValueError("Refusing to create ground-truth scripts for non-synthetic samples")
        if not sample.sample_id.startswith("synthetic_"):
            raise ValueError("Synthetic Mock sample IDs must be explicitly marked")
        path = (dataset_dir / sample.image_path).resolve()
        if not path.is_relative_to(dataset_dir.resolve()):
            raise ValueError("Synthetic Mock images must stay inside the fixture directory")
        if not sample.persons or any(
            not any(e.get("type") == "synthetic" for e in p.evidence)
            or (p.character is not None and not p.character.startswith("测试角色"))
            for p in sample.persons
        ):
            raise ValueError("Synthetic Mock scripting requires invented roles with synthetic evidence")


def build_demo_responses(
    samples: Sequence[MultiSample], dataset_dir: Path | str, input_builder: InputBuilder,
) -> dict[ResponseKey, dict[str, Any]]:
    """Explicitly build correct A/B/C scripts ONLY for synthetic demo data.

    ``input_builder`` has the executor's (sample, experiment, work) signature and
    returns (prompt, encoded_image_data_url, metadata). All samples are checked
    before the builder is invoked, preventing accidental real-GT scripting.
    """
    directory = Path(dataset_dir)
    _require_synthetic(samples, directory)
    errors, _ = validate_dataset(list(samples), directory, for_run=True)
    if errors:
        raise ValueError(f"Synthetic fixture is invalid: {errors}")
    known_works = {w.slug: w for w in load_works()}
    responses = {}
    for sample in samples:
        work = known_works.get(sample.work_slug) or Work(
            sample.work_slug, sample.work_name, sample.work_category, "影视作品", None,
        )
        for experiment in ("A", "B", "C"):
            prompt, data_url, _meta = input_builder(sample, experiment, work)
            persons = [{"bbox": list(p.bbox), "character": p.character} for p in sample.persons] if experiment == "C" else [
                {"person_id": p.person_id, "character": p.character} for p in sample.persons
            ]
            key = (encoded_image_sha256(data_url), prompt)
            response = {"persons": persons}
            if key in responses and responses[key] != response:
                raise ValueError("Synthetic builder produced a script-key collision with different answers")
            responses[key] = response
    return responses
