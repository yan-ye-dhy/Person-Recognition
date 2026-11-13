"""Strict, auditable v2 protocol parsing; malformed output never becomes correct."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any

from ..evaluation.parser import extract_json_text, looks_like_refusal


@dataclass
class ParsedPerson:
    character: str | None
    person_id: int | None = None
    bbox: list[float] | None = None


@dataclass
class ParsedMultiResponse:
    raw_text: str = ""
    persons: list[ParsedPerson] = field(default_factory=list)
    parse_ok: bool = False
    refused: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    raw_person_count: int = 0
    missing_ids: list[int] = field(default_factory=list)
    extra_ids: list[int] = field(default_factory=list)

    @property
    def entities(self):
        return self.persons


def bbox_valid(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def parse_response(raw_text: str | None, experiment: str, expected_ids: list[int] | None = None,
                   refusal_keywords: list[str] | None = None) -> ParsedMultiResponse:
    experiment = experiment.upper()
    if experiment not in {"A", "B", "C"}:
        raise ValueError("experiment must be A, B, or C")
    parsed = ParsedMultiResponse(raw_text=raw_text or "")
    try:
        def unique_keys(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError(f"duplicate_json_key:{key}")
                obj[key] = value
            return obj
        data = json.loads(extract_json_text(parsed.raw_text), object_pairs_hook=unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite:{value}")))
        if not isinstance(data, dict) or not isinstance(data.get("persons"), list):
            raise ValueError("persons_must_be_array")
        if set(data) - {"persons", "reason"}:
            raise ValueError("unexpected_top_level_fields")
        if len(data["persons"]) > 100:
            raise ValueError("too_many_predictions")
        parsed.raw_person_count = len(data["persons"])
        seen = set()
        for index, item in enumerate(data["persons"]):
            if not isinstance(item, dict) or "character" not in item:
                parsed.issues.append(f"invalid_person:{index}")
                continue
            allowed = {"person_id", "character"} if experiment in {"A", "B"} else {"bbox", "character"}
            if set(item) - allowed:
                parsed.issues.append(f"unexpected_person_fields:{index}")
                continue
            name = item["character"]
            if name is not None and (not isinstance(name, str) or not name.strip()):
                parsed.issues.append(f"invalid_character:{index}")
                continue
            if isinstance(name, str):
                name = name.strip()
                if re.search(r"[|/、;；]|或|也可能|\bor\b", name, re.I):
                    parsed.issues.append(f"multiple_candidates:{index}")
                    continue
            if experiment in {"A", "B"}:
                pid = item.get("person_id")
                if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                    parsed.issues.append(f"invalid_person_id:{index}")
                    continue
                if pid in seen:
                    parsed.issues.append(f"duplicate_person_id:{pid}")
                seen.add(pid)
                parsed.persons.append(ParsedPerson(name, person_id=pid))
            else:
                box = item.get("bbox")
                if not bbox_valid(box):
                    parsed.issues.append(f"invalid_bbox:{index}")
                    continue
                if any(box_iou(box, other.bbox) >= .95 for other in parsed.persons):
                    parsed.issues.append(f"duplicate_position:{index}")
                parsed.persons.append(ParsedPerson(name, bbox=[float(x) for x in box]))
        if parsed.issues:
            raise ValueError(";".join(parsed.issues))
        parsed.parse_ok = True
        if experiment in {"A", "B"} and expected_ids is not None:
            parsed.missing_ids = sorted(set(expected_ids) - seen)
            parsed.extra_ids = sorted(seen - set(expected_ids))
            if parsed.missing_ids:
                parsed.issues.append("missing_ids:" + ",".join(map(str, parsed.missing_ids)))
            if parsed.extra_ids:
                parsed.issues.append("extra_ids:" + ",".join(map(str, parsed.extra_ids)))
        names = [p.character for p in parsed.persons if p.character is not None]
        if len(names) != len(set(names)):
            parsed.warnings.append("repeated_character_names_scored_by_id_or_position_not_deduplicated")
        parsed.refused = looks_like_refusal(parsed.raw_text, refusal_keywords) and not any(p.character for p in parsed.persons)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        parsed.error = str(exc)
        if not parsed.issues:
            parsed.issues.append("invalid_json_or_schema")
        parsed.refused = looks_like_refusal(parsed.raw_text, refusal_keywords) and not any(p.character for p in parsed.persons)
    return parsed
