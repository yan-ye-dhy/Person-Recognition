"""Offline Qwen Max identity diagnostics independent of coordinate/ID formatting.

This is a name-coverage diagnostic, not the original localization or person-ID
accuracy. Only JSON persons[].character fields are read. One prediction can
cover at most one approved GT person; nulls and missing names remain misses.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from ..annotation.rosters import load_roster
from ..config import Work
from ..evaluation.matcher import CharacterMatcher
from ..evaluation.parser import extract_json_text
from .parsing import parse_response
from .schema import MultiSample, dataset_hash, evaluation_policy_errors
from .scoring import identity_category, score_sample

RUN_ID = "multi_v2_20261113_075729_2e4f15"
MODEL_ID = "qwen3.8-max"
CORRECTIONS_PATH = Path("reports/2026-11-15-qwenmax-review/annotation_corrections.json")
MULTIPLE_NAMES = re.compile(r"[|/、;；]|或|也可能|\bor\b", re.I)


def _unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate_json_key:{key}")
        obj[key] = value
    return obj


def read_name_predictions(raw_text: str) -> tuple[list[dict], str | None]:
    """Read only actual name fields, ignoring box/ID/schema formatting constraints.

    Invalid JSON and duplicate JSON keys are not guessed or recovered with regex.
    Strings containing multiple candidates are preserved for review, not matched.
    """
    try:
        value = json.loads(extract_json_text(raw_text), object_pairs_hook=_unique_keys,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
        if not isinstance(value, dict) or not isinstance(value.get("persons"), list):
            raise ValueError("persons_must_be_array")
    except (ValueError, TypeError) as exc:
        return [], str(exc)
    predictions = []
    for index, person in enumerate(value["persons"]):
        name = person.get("character") if isinstance(person, dict) else None
        valid = isinstance(name, str) and bool(name.strip()) and not MULTIPLE_NAMES.search(name)
        predictions.append({
            "prediction_index": index,
            "character": name,
            "original_bbox": deepcopy(person.get("bbox")) if isinstance(person, dict) else None,
            "person_id": person.get("person_id") if isinstance(person, dict) else None,
            "matched_person_id": None,
            "outcome": "extra_name" if valid else "null" if isinstance(person, dict) and "character" in person and name is None else "invalid_name",
            "valid_name": bool(valid),
        })
    return predictions, None


def score_name_only(sample: dict, experiment: str, raw_text: str,
                    matcher: CharacterMatcher) -> dict[str, Any]:
    """Reviewed name coverage, with required-first matching and explicit quotas.

    Full visible-person coverage remains separate. Optional identities and a
    correctly named surplus quota candidate are accepted, never extra names.
    This policy contains no model-specific exceptions.
    """
    people = [p for p in sample["persons"] if identity_category(p) == "verified"]
    policy = sample.get("evaluation_policy", {})
    errors = evaluation_policy_errors(policy, sample["persons"])
    if errors:
        raise ValueError("Invalid name evaluation policy: " + "; ".join(errors))
    optional_ids = set(policy.get("optional_person_ids", []))
    required_indices = [i for i, p in enumerate(people) if p["person_id"] not in optional_ids]
    optional_indices = [i for i, p in enumerate(people) if p["person_id"] in optional_ids]
    quota = policy.get("required_name_count", len(required_indices))
    predictions, extraction_error = read_name_predictions(raw_text)
    edges: dict[int, list[int]] = {}
    matches = {}
    for i, person in enumerate(people):
        edges[i] = []
        for j, prediction in enumerate(predictions):
            if not prediction["valid_name"]:
                continue
            result = matcher.match(prediction["character"], sample["work_slug"], person["character"],
                                   person.get("character_aliases", []))
            if result.is_correct:
                edges[i].append(j)
                matches[(i, j)] = result.kind
    by_prediction: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in by_prediction or augment(by_prediction[j], seen):
                by_prediction[j] = i
                return True
        return False

    # Optional matching may reroute an augmenting path but cannot unmatch a
    # previously covered required person: required coverage has first priority.
    for i in required_indices + optional_indices:
        augment(i, set())
    by_person = {i: j for j, i in by_prediction.items()}
    required_matched = [i for i in required_indices if i in by_person]
    scored_indices = set(required_matched[:quota])
    for i in required_indices:
        if len(scored_indices) >= quota:
            break
        if i not in by_person:
            scored_indices.add(i)
    for j, i in by_prediction.items():
        predictions[j].update(matched_person_id=people[i]["person_id"], outcome="matched",
                              optional=i in optional_indices, counts_toward_required=i in scored_indices)
    remaining = iter(j for j in range(len(predictions)) if j not in by_prediction)
    details = []
    for i, person in enumerate(people):
        optional, scored = i in optional_indices, i in scored_indices
        if i in by_person:
            j = by_person[i]
            full_outcome, kind = "correct", matches[(i, j)]
        elif not scored:
            j, full_outcome, kind = None, "missing_name", "missing"
        else:
            # Display an unmatched response beside a missed GT only as a
            # deterministic audit aid; this does not infer a physical association.
            j = next(remaining, None)
            if j is None:
                full_outcome, kind = "missing_name", "missing"
            elif predictions[j]["outcome"] == "null":
                full_outcome, kind = "uncertain", "empty"
            elif not predictions[j]["valid_name"]:
                full_outcome, kind = "wrong_name", "invalid_name"
            else:
                result = matcher.match(predictions[j]["character"], sample["work_slug"], person["character"],
                                       person.get("character_aliases", []))
                full_outcome, kind = "wrong_name", result.kind
        outcome = ("optional_correct" if full_outcome == "correct" else "optional_missing") if optional else (
            full_outcome if scored else "not_required")
        details.append({
            "person_id": person["person_id"], "gt_character": person["character"],
            "predicted": predictions[j]["character"] if j is not None else None,
            "prediction_index": j, "outcome": outcome, "match_kind": kind,
            "assignment_is_name_only": True,
            "required": not optional, "optional": optional, "scored": scored,
            "counts_toward_required": scored, "required_credit": scored and full_outcome == "correct",
            "full_correct": full_outcome == "correct", "full_outcome": full_outcome, "full_scored": True,
        })
    named = sum(p["valid_name"] for p in predictions)
    correct = min(len(required_matched), quota)
    optional_correct = sum(i in by_person for i in optional_indices)
    full_correct = len(by_prediction)
    return {
        "sample_id": sample["sample_id"], "work_slug": sample["work_slug"], "experiment": experiment,
        "target_count": quota, "correct": correct, "named_predictions": named,
        "extra_names": named - full_correct, "image_names_complete": correct == quota,
        "required_candidate_count": len(required_indices), "required_matched_count": len(required_matched),
        "optional_target_count": len(optional_indices), "optional_correct": optional_correct,
        "optional_missing": len(optional_indices) - optional_correct,
        "full_target_count": len(people), "full_correct": full_correct,
        "full_names_complete": full_correct == len(people), "accepted_named_predictions": full_correct,
        "evaluation_policy": deepcopy(policy),
        "person_details": details, "predictions": predictions, "extraction_error": extraction_error,
        "null_predictions": sum(p["outcome"] == "null" for p in predictions),
        "invalid_name_predictions": sum(p["outcome"] == "invalid_name" for p in predictions),
    }


def summarize_name_rows(rows: list[dict]) -> dict:
    targets = sum(r["target_count"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    named = sum(r["named_predictions"] for r in rows)
    accepted = sum(r.get("accepted_named_predictions", r["correct"]) for r in rows)
    return {
        "images": len(rows), "target_count": targets, "correct": correct,
        "accuracy": correct / targets if targets else None,
        "named_predictions": named, "extra_names": sum(r["extra_names"] for r in rows),
        "precision": accepted / named if named else None,
        "accepted_named_predictions": accepted,
        "full_target_count": sum(r.get("full_target_count", r["target_count"]) for r in rows),
        "full_correct": sum(r.get("full_correct", r["correct"]) for r in rows),
        "optional_target_count": sum(r.get("optional_target_count", 0) for r in rows),
        "optional_correct": sum(r.get("optional_correct", 0) for r in rows),
        "image_names_complete": sum(r["image_names_complete"] for r in rows),
        "images_with_extra_names": sum(r["extra_names"] > 0 for r in rows),
        "additional_correct_vs_strict": sum(r.get("additional_correct_vs_strict", 0) for r in rows),
        "additional_correct_from_invalid_format": sum(r.get("additional_correct_vs_strict", 0) for r in rows if not r.get("strict_parse_ok", True)),
        "additional_correct_from_valid_output": sum(r.get("additional_correct_vs_strict", 0) for r in rows if r.get("strict_parse_ok", True)),
        "null_predictions": sum(r["null_predictions"] for r in rows),
        "invalid_name_predictions": sum(r["invalid_name_predictions"] for r in rows),
    }


def apply_corrections(samples: dict[str, dict], corrections: list[dict]) -> dict[str, dict]:
    """Replay explicit, fingerprint-checked human corrections on a copied snapshot."""
    result = deepcopy(samples)
    seen = set()
    for correction in corrections:
        sid, pid = correction["sample_id"], correction["person_id"]
        if (sid, pid) in seen:
            raise ValueError(f"Duplicate correction for {sid}/{pid}")
        seen.add((sid, pid))
        sample = result[sid]
        if sample["image_sha256"] != correction["image_sha256"]:
            raise ValueError(f"Correction source image mismatch: {sid}")
        person = next(p for p in sample["persons"] if p["person_id"] == pid)
        if person["annotation_id"] != correction["annotation_id"] or person["character"] != correction["old_character"]:
            raise ValueError(f"Correction does not match frozen identity: {sid}/{pid}")
        person["character"] = correction["new_character"]
        person["character_aliases"] = deepcopy(correction.get("new_character_aliases", []))
        if "new_person_evidence" in correction:
            person["evidence"] = deepcopy(correction["new_person_evidence"])
        if "new_person_note" in correction:
            person["note"] = correction["new_person_note"]
    return result


def build_identity_review(root: Path) -> dict:
    """Build corrected strict and name-only review rows, retaining original evidence."""
    root = Path(root).resolve()
    source = json.loads((root / "reports/2026-11-13-multi-v2/summary.json").read_text(encoding="utf-8"))
    if source["run_id"] != RUN_ID or source.get("is_mock"):
        raise ValueError("Expected approved completed real run")
    frozen_path = root / "results/multi" / RUN_ID / "dataset/dataset.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    original_samples = {s["sample_id"]: s for s in frozen["samples"]}
    source_hash = dataset_hash([MultiSample.from_dict(s) for s in original_samples.values()])
    if source_hash != source["dataset_hash"]:
        raise ValueError("Frozen dataset differs from original run summary")
    correction_path = root / CORRECTIONS_PATH
    corrections = json.loads(correction_path.read_text(encoding="utf-8"))["corrections"] if correction_path.exists() else []
    samples = apply_corrections(original_samples, corrections)
    review_hash = dataset_hash([MultiSample.from_dict(s) for s in samples.values()])
    rosters = {}
    for sample in samples.values():
        slug = sample["work_slug"]
        if slug not in rosters:
            work = Work(slug, sample["work_name"], sample["work_category"], "", None)
            rosters[slug] = load_roster(work, root / "config/rosters")
    with sqlite3.connect((root / "results/multi/runs.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        raw_records = {(exp, sid): json.loads(raw) for exp, sid, raw in conn.execute(
            "SELECT experiment,sample_id,raw_json FROM records WHERE run_id=? AND model_id=?",
            (RUN_ID, MODEL_ID))}
        meta = json.loads(conn.execute("SELECT fairness_meta FROM runs WHERE run_id=?", (RUN_ID,)).fetchone()[0])
    settings = meta["settings_snapshot"]
    matcher = CharacterMatcher(rosters, float(settings.get("scoring", {}).get("fuzzy_threshold", .9)))
    original_rows = deepcopy([r for r in source["records"] if r["model_id"] == MODEL_ID])
    if len(original_rows) != 480 or len(raw_records) != 480:
        raise ValueError("Expected all 480 completed Qwen Max tasks")
    affected = {c["sample_id"] for c in corrections}
    rows, name_rows = [], []
    scoring = settings.get("multi", {}).get("scoring", settings.get("multi_scoring", {}))
    for original in original_rows:
        sid, exp = original["sample_id"], original["experiment"]
        sample, raw = samples[sid], raw_records[(exp, sid)]
        row = deepcopy(original)
        if sid in affected:
            parsed = parse_response(raw["response"]["raw_text"], exp, [p["person_id"] for p in sample["persons"]],
                                    settings.get("scoring", {}).get("refusal_keywords"))
            corrected = score_sample(sample, exp, parsed, matcher, api_state=original["api_state"],
                                     iou_threshold=float(scoring.get("iou_threshold", .5)),
                                     ignore_overlap_threshold=float(scoring.get("ignore_overlap_threshold", .5)))
            tags = {p.get("person_id"):p for p in original["person_details"] if p.get("person_id") is not None}
            for person in corrected["person_details"]:
                for axis in ("face_orientation", "visibility"):
                    if axis in tags.get(person.get("person_id"), {}):
                        person[axis] = tags[person["person_id"]][axis]
            row.update(corrected)
            row["annotation_correction_applied"] = True
        diagnostic = score_name_only(sample, exp, raw["response"]["raw_text"], matcher)
        diagnostic.update(strict_correct=row["identity_correct"], strict_parse_ok=row["parse_ok"],
                          additional_correct_vs_strict=diagnostic["correct"] - row["identity_correct"])
        rows.append(row)
        name_rows.append(diagnostic)
    return {
        "summary": source, "samples": samples, "raw_records": raw_records,
        "original_rows": original_rows, "rows": rows, "name_rows": name_rows,
        "corrections": corrections, "source_dataset_hash": source_hash, "review_dataset_hash": review_hash,
        "matcher_source": "Current unchanged config/rosters; no roster snapshot was stored with this run",
        "diagnostic_summary": {exp: summarize_name_rows([r for r in name_rows if r["experiment"] == exp]) for exp in ("A","B","C")},
    }
