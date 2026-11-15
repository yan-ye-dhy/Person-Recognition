"""Version 2 annotations, independent of the frozen single-target dataset."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IDENTITY_STATUSES = {"verified", "unknown", "needs_review"}
PERSON_REVIEW_STATUSES = {"approved", "pending"}
SAMPLE_REVIEW_STATUSES = {"draft", "approved"}
SCHEMA_VERSION = "2.0"


def evaluation_policy_errors(policy: Any, persons: list[Any]) -> list[str]:
    """Review policies affect offline name coverage, never the visible people."""
    if not isinstance(policy, dict):
        return ["evaluation_policy must be an object"]
    errors = []
    if set(policy) - {"optional_person_ids", "required_name_count"}:
        errors.append("unknown evaluation_policy field")
    optional = policy.get("optional_person_ids", [])
    if (not isinstance(optional, list)
            or any(type(pid) is not int or pid < 1 for pid in optional)):
        return errors + ["optional_person_ids must be a list of positive person IDs"]
    if len(set(optional)) != len(optional):
        errors.append("optional_person_ids cannot contain duplicates")
    def get(person, key, default=None):
        return person.get(key, default) if isinstance(person, dict) else getattr(person, key, default)
    verified = {get(p, "person_id") for p in persons
                if get(p, "identity_status") == "verified" and get(p, "review_status") == "approved"
                and get(p, "character")}
    if set(optional) - verified:
        errors.append("optional_person_ids must reference approved verified people")
    if "required_name_count" in policy:
        count = policy["required_name_count"]
        if type(count) is not int or not 1 <= count <= len(verified - set(optional)):
            errors.append("required_name_count must be within the verified required candidate count")
    return errors


def bbox_errors(box: Any) -> list[str]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return ["bbox must contain four normalized coordinates"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return ["bbox coordinates must be finite numbers"]
    x1, y1, x2, y2 = box
    return [] if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 else ["bbox requires 0<=x1<x2<=1 and 0<=y1<y2<=1"]


@dataclass
class PersonAnnotation:
    person_id: int
    bbox: list[float]
    character: str | None = None
    identity_status: str = "needs_review"
    review_status: str = "pending"
    character_aliases: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    annotation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def aliases(self) -> list[str]:
        return self.character_aliases

    def validate(self, for_run: bool = False) -> list[str]:
        errors = bbox_errors(self.bbox)
        if isinstance(self.person_id, bool) or not isinstance(self.person_id, int) or self.person_id < 1:
            errors.append("person_id must be a positive integer")
        if self.identity_status not in IDENTITY_STATUSES:
            errors.append("invalid identity_status")
        if self.review_status not in PERSON_REVIEW_STATUSES:
            errors.append("invalid person review_status")
        if not isinstance(self.evidence, list) or any(not isinstance(e, dict) for e in self.evidence):
            errors.append("evidence must be a list of objects")
        if not isinstance(self.character_aliases, list) or any(not isinstance(a, str) for a in self.character_aliases):
            errors.append("character_aliases must be a list of names")
        if self.character is not None and (not isinstance(self.character, str) or not self.character.strip()):
            errors.append("character must be a nonempty name or null")
        if self.identity_status == "unknown" and self.character is not None:
            errors.append("unknown identity requires character=null")
        if self.identity_status == "verified":
            if not self.character:
                errors.append("verified identity requires a character")
            unconfirmed_sources = {None, "v1_inheritance", "inheritance", "user_hypothesis", "identity_withdrawal", "model_consensus", "model_prediction"}
            if not isinstance(self.evidence, list) or not any(isinstance(e, dict) and e.get("type") not in unconfirmed_sources and e.get("confirmed") is not False for e in self.evidence):
                errors.append("verified identity requires independent review evidence; inheritance alone is insufficient")
        if self.review_status == "approved" and self.identity_status == "needs_review":
            errors.append("needs_review identity cannot be approved")
        if for_run and (self.review_status != "approved" or self.identity_status == "needs_review"):
            errors.append("formal run requires each person approved with verified or unknown identity")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PersonAnnotation":
        value = dict(value)
        if "aliases" in value and "character_aliases" not in value:
            value["character_aliases"] = value["aliases"]
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


@dataclass
class IgnoreRegion:
    bbox: list[float]
    reason: str = "不可辨识背景人物"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | list[float]) -> "IgnoreRegion":
        return cls(bbox=list(value)) if isinstance(value, (list, tuple)) else cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


@dataclass
class MultiSample:
    sample_id: str
    work_slug: str
    work_name: str
    work_category: str
    image_path: str
    persons: list[PersonAnnotation] = field(default_factory=list)
    ignore_regions: list[IgnoreRegion] = field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None
    image_sha256: str = ""
    review_status: str = "draft"
    foreground_complete: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    evaluation_policy: dict[str, Any] = field(default_factory=dict)

    def validate(self, for_run: bool = False) -> list[str]:
        errors = []
        for name in ("sample_id", "work_slug", "work_name", "work_category", "image_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                errors.append(f"{name} is required")
        if Path(self.image_path).is_absolute():
            errors.append("image_path must be relative to the v2 dataset")
        if self.review_status not in SAMPLE_REVIEW_STATUSES:
            errors.append("invalid sample review_status")
        if not isinstance(self.foreground_complete, bool):
            errors.append("foreground_complete must be boolean")
        if not self.persons:
            errors.append("sample needs at least one foreground person")
        ids = [p.person_id for p in self.persons]
        if len(set(ids)) != len(ids):
            errors.append("duplicate person_id")
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids) or sorted(ids) != list(range(1, len(ids) + 1)):
            errors.append("person_id must be consecutive starting at 1")
        for person in self.persons:
            errors.extend(f"person {person.person_id}: {e}" for e in person.validate(for_run))
        errors.extend(evaluation_policy_errors(self.evaluation_policy, self.persons))
        for i, region in enumerate(self.ignore_regions):
            errors.extend(f"ignore {i + 1}: {e}" for e in bbox_errors(region.bbox))
            if not region.reason.strip():
                errors.append(f"ignore {i + 1}: reason is required")
        if self.review_status == "approved" or for_run:
            if not self.foreground_complete:
                errors.append("sample must explicitly confirm foreground_complete")
            if any(p.review_status != "approved" or p.identity_status == "needs_review" for p in self.persons):
                errors.append("sample approval requires all foreground persons reviewed")
        if for_run and self.review_status != "approved":
            errors.append("formal run requires sample review_status=approved")
        return errors

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Historical frozen snapshots did not have this field. An absent policy
        # must retain their exact canonical serialization and dataset hashes.
        if value["evaluation_policy"] == {}:
            value.pop("evaluation_policy")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MultiSample":
        known = {k: v for k, v in value.items() if k in cls.__dataclass_fields__}
        known["persons"] = [p if isinstance(p, PersonAnnotation) else PersonAnnotation.from_dict(p) for p in value.get("persons", [])]
        known["ignore_regions"] = [r if isinstance(r, IgnoreRegion) else IgnoreRegion.from_dict(r) for r in value.get("ignore_regions", [])]
        return cls(**known)


def renumber_persons(persons: list[PersonAnnotation]) -> list[PersonAnnotation]:
    """Deterministic left-to-right, then top-to-bottom IDs; internal IDs retain identity."""
    ordered = sorted(persons, key=lambda p: (p.bbox[0], p.bbox[1], p.annotation_id))
    for index, person in enumerate(ordered, 1):
        person.person_id = index
    return ordered


def load_dataset(path: Path) -> list[MultiSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema_version", "")).split(".")[0] != "2":
        raise ValueError("not a version 2 dataset")
    return [MultiSample.from_dict(s) for s in payload["samples"]]


def dataset_hash(samples: list[MultiSample]) -> str:
    data = json.dumps([s.to_dict() for s in samples], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def dataset_stats(samples: list[MultiSample]) -> dict[str, Any]:
    persons = [p for s in samples for p in s.persons]
    by_work = {}
    for sample in samples:
        group = by_work.setdefault(sample.work_slug, {"samples": 0, "approved": 0, "draft": 0, "persons": 0})
        group["samples"] += 1
        group[sample.review_status] += 1
        group["persons"] += len(sample.persons)
    return {"total": len(samples), "samples": len(samples), "persons": len(persons), "approved_samples": sum(s.review_status == "approved" for s in samples), "draft_samples": sum(s.review_status == "draft" for s in samples), "foreground_incomplete": sum(not s.foreground_complete for s in samples), "verified_persons": sum(p.identity_status == "verified" for p in persons), "unknown_persons": sum(p.identity_status == "unknown" for p in persons), "needs_review_persons": sum(p.identity_status == "needs_review" for p in persons), "pending_persons": sum(p.review_status == "pending" for p in persons), "ignore_regions": sum(len(s.ignore_regions) for s in samples), "by_work": by_work, "person_count_distribution": dict(Counter(str(len(s.persons)) for s in samples))}


def validate_dataset(samples: list[MultiSample], dataset_dir: Path, for_run: bool = False) -> tuple[list[str], dict[str, Any]]:
    errors = []
    seen = set()
    for sample in samples:
        errors.extend(f"{sample.sample_id}: {e}" for e in sample.validate(for_run))
        if sample.sample_id in seen:
            errors.append(f"duplicate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        image = dataset_dir / sample.image_path
        if not image.is_file():
            errors.append(f"{sample.sample_id}: missing image")
        elif sample.image_sha256 and hashlib.sha256(image.read_bytes()).hexdigest() != sample.image_sha256:
            errors.append(f"{sample.sample_id}: original image SHA256 mismatch")
    return errors, dataset_stats(samples)
