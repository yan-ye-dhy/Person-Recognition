"""Fingerprint-checked offline replay of explicit annotation reviews.

Frozen run inputs and raw model responses are never rewritten. The manifest
retains before/after samples for verification, independently of report layout.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import MultiSample, dataset_hash, evaluation_policy_errors

REVIEW_REVISION_PATH = Path("reports/2026-11-16-annotation-review/revision.json")
_IMAGE_FIELDS = ("sample_id", "work_slug", "work_name", "work_category", "image_path",
                 "image_width", "image_height", "image_sha256")


def _sample(value: dict[str, Any] | MultiSample) -> dict[str, Any]:
    if isinstance(value, MultiSample):
        return deepcopy(value.to_dict())
    if not isinstance(value, dict):
        raise ValueError("Review samples must be objects")
    unknown = set(value) - set(MultiSample.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown review sample fields: " + ",".join(sorted(unknown)))
    try:
        sample = MultiSample.from_dict(deepcopy(value)).to_dict()
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid review sample structure") from error
    errors = evaluation_policy_errors(sample.get("evaluation_policy", {}), sample["persons"])
    if errors:
        raise ValueError("Invalid review policy: " + "; ".join(errors))
    return sample


def canonical_sample_hash(sample: dict[str, Any] | MultiSample) -> str:
    """Canonical single-sample object hash (without a surrounding list)."""
    data = json.dumps(_sample(sample), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _samples(values: dict[str, dict[str, Any] | MultiSample]) -> dict[str, dict[str, Any]]:
    if not isinstance(values, dict):
        raise ValueError("Review datasets must be sample_id -> sample mappings")
    result = {sid: _sample(value) for sid, value in values.items()}
    if any(not isinstance(sid, str) or sample["sample_id"] != sid for sid, sample in result.items()):
        raise ValueError("Review sample mapping key mismatch")
    return result


def _dataset_hash(samples: dict[str, dict[str, Any]]) -> str:
    # Keep the original dataset order. Reordering a frozen snapshot is itself a
    # version change, and must not be silently normalized away.
    return dataset_hash([MultiSample.from_dict(sample) for sample in samples.values()])


def _fingerprint(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: sample.get(key) for key in _IMAGE_FIELDS}


def replay_review_revision(samples: dict[str, dict[str, Any] | MultiSample], manifest: dict[str, Any],
                           *, reverse: bool = False) -> dict[str, dict[str, Any]]:
    """Replay exact before -> after snapshots, or reconstruct their frozen base.

    The complete dataset hashes and every changed sample's canonical hash are
    checked in both directions. Only annotation/policy edits on the same image
    are permitted; the caller's samples and manifest remain untouched.
    """
    result = _samples(samples)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0"
            or not isinstance(manifest.get("revision_id"), str) or not manifest["revision_id"].strip()
            or not isinstance(manifest.get("revisions"), list)):
        raise ValueError("Invalid annotation review manifest")
    source_key, target_key = (("after_dataset_hash", "before_dataset_hash") if reverse else
                              ("before_dataset_hash", "after_dataset_hash"))
    if _dataset_hash(result) != manifest.get(source_key):
        raise ValueError("Review source dataset fingerprint mismatch")
    seen = set()
    for revision in manifest["revisions"]:
        if not isinstance(revision, dict):
            raise ValueError("Review revision must be an object")
        sid = revision.get("sample_id")
        if not isinstance(sid, str) or sid not in result or sid in seen:
            raise ValueError("Unknown or duplicate review sample: " + str(sid))
        seen.add(sid)
        before, after = _sample(revision.get("before")), _sample(revision.get("after"))
        if (before["sample_id"] != sid or after["sample_id"] != sid
                or _fingerprint(before) != _fingerprint(after)
                or not revision.get("image_sha256")
                or before["image_sha256"] != revision["image_sha256"]):
            raise ValueError("Review image fingerprint mismatch: " + sid)
        if (canonical_sample_hash(before) != revision.get("before_sample_hash")
                or canonical_sample_hash(after) != revision.get("after_sample_hash")):
            raise ValueError("Review before/after sample hash mismatch: " + sid)
        expected, replacement = (after, before) if reverse else (before, after)
        if canonical_sample_hash(result[sid]) != canonical_sample_hash(expected):
            raise ValueError("Review sample does not match source snapshot: " + sid)
        errors = MultiSample.from_dict(replacement).validate(for_run=True)
        if errors:
            raise ValueError("Reviewed annotation is not approved: " + sid + ": " + "; ".join(errors))
        result[sid] = deepcopy(replacement)
    if _dataset_hash(result) != manifest.get(target_key):
        raise ValueError("Review result dataset fingerprint mismatch")
    return result


def _scoring_truth(sample: dict[str, Any]) -> dict[str, Any]:
    people = [{key: person.get(key) for key in
               ("person_id", "annotation_id", "bbox", "character", "character_aliases",
                "identity_status", "review_status")} for person in sample["persons"]]
    return {"image": _fingerprint(sample), "persons": people,
            "ignore_regions": [region["bbox"] for region in sample["ignore_regions"]],
            "review_status": sample["review_status"], "foreground_complete": sample["foreground_complete"],
            "evaluation_policy": sample.get("evaluation_policy", {})}


def validate_frozen_source(source_samples: dict[str, dict[str, Any] | MultiSample],
                           base_samples: dict[str, dict[str, Any] | MultiSample]) -> None:
    """Check already-fingerprint-validated source truth against the review base.

    Callers validate each source run's original stored dataset hash first. For
    the older baseline they apply the separately audited legacy corrections
    first. Evidence/note prose may differ; scoring identities, boxes, status,
    policy, and image fingerprints must agree exactly before using revised GT.
    """
    source, base = _samples(source_samples), _samples(base_samples)
    if set(source) != set(base):
        raise ValueError("Frozen source sample set differs from review base")
    for sid in base:
        if _scoring_truth(source[sid]) != _scoring_truth(base[sid]):
            raise ValueError("Frozen source truth differs from review base: " + sid)
