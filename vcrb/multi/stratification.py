"""Statistics-only person tags, strictly bound to a dataset and frozen per run.

These labels never participate in image generation or model prompts. Existing
runs without a frozen sidecar never borrow labels from the working directory.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import dataset_hash
from .store import atomic_write_json

FACE_ORIENTATIONS = ("front", "three_quarter", "profile", "not_visible")
VISIBILITIES = ("clear", "occluded_or_blurry")
SIDECAR_NAME = "stratification.json"


def _get(value: Any, key: str):
    return value.get(key) if isinstance(value, dict) else getattr(value, key)


def stratification_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_stratification(payload: dict[str, Any], samples: list[Any]) -> dict[str, Any]:
    """Reject stale, incomplete, duplicate or foreign person tags; return a copy."""
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("stratification requires schema_version=1.0")
    if payload.get("dataset_hash") != dataset_hash(samples):
        raise ValueError("stratification dataset_hash does not match the annotation snapshot")
    expected = {}
    for sample in samples:
        for person in _get(sample, "persons"):
            key = (_get(sample, "sample_id"), _get(person, "person_id"))
            if key in expected:
                raise ValueError("Duplicate GT person in stratification scope")
            expected[key] = (_get(sample, "image_sha256"), _get(person, "annotation_id"))
    rows = payload.get("persons")
    if not isinstance(rows, list):
        raise ValueError("stratification persons must be an array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("stratification person must be an object")
        sample_id, person_id = row.get("sample_id"), row.get("person_id")
        if not isinstance(sample_id, str) or type(person_id) is not int or person_id < 1:
            raise ValueError("stratification needs sample_id and a positive integer person_id")
        key = sample_id, person_id
        if key in seen:
            raise ValueError(f"Duplicate stratification person: {key}")
        if key not in expected:
            raise ValueError(f"Foreign stratification person: {key}")
        if (row.get("image_sha256"), row.get("annotation_id")) != expected[key]:
            raise ValueError(f"Stale stratification image/annotation binding: {key}")
        if not row.get("image_sha256") or not row.get("annotation_id"):
            raise ValueError(f"Missing stratification image/annotation binding: {key}")
        if row.get("face_orientation") not in FACE_ORIENTATIONS:
            raise ValueError(f"Invalid face_orientation for {key}")
        if row.get("visibility") not in VISIBILITIES:
            raise ValueError(f"Invalid visibility for {key}")
        seen.add(key)
    if seen != expected.keys():
        raise ValueError(f"Incomplete stratification labels: {len(expected) - len(seen)} persons missing")
    return deepcopy(payload)


def load_stratification(path: Path, samples: list[Any]) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read stratification sidecar: {error}") from error
    return validate_stratification(payload, samples)


def freeze_stratification(payload: dict[str, Any], samples: list[Any], snapshot_dir: Path) -> dict[str, Any]:
    """Freeze selected persons from an already validated full-dataset sidecar."""
    selected_ids = {_get(s, "sample_id") for s in samples}
    frozen = deepcopy(payload)
    frozen["source_dataset_hash"] = payload["dataset_hash"]
    frozen["source_sidecar_hash"] = stratification_hash(payload)
    frozen["dataset_hash"] = dataset_hash(samples)
    frozen["persons"] = [r for r in frozen["persons"] if r["sample_id"] in selected_ids]
    frozen = validate_stratification(frozen, samples)
    path = Path(snapshot_dir) / SIDECAR_NAME
    atomic_write_json(path, frozen)
    return {"schema_version": "1.0", "file": SIDECAR_NAME,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dataset_hash": frozen["dataset_hash"], "person_labels": len(frozen["persons"]),
            "source_dataset_hash": payload["dataset_hash"],
            "source_sidecar_hash": frozen["source_sidecar_hash"],
            "purpose": "statistics_only_never_in_model_inputs"}


def load_frozen_stratification(run: dict[str, Any], samples: list[Any]) -> dict[str, Any] | None:
    metadata = (run.get("fairness_meta") or {}).get("stratification")
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("file") != SIDECAR_NAME:
        raise ValueError("Invalid frozen stratification metadata")
    path = Path(run["dataset_path"]).parent / SIDECAR_NAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("Frozen stratification sidecar is missing") from error
    if hashlib.sha256(raw).hexdigest() != metadata.get("sha256"):
        raise ValueError("Frozen stratification SHA256 mismatch")
    payload = validate_stratification(json.loads(raw), samples)
    if metadata.get("dataset_hash") != payload["dataset_hash"] or metadata.get("person_labels") != len(payload["persons"]):
        raise ValueError("Frozen stratification metadata does not match its labels")
    return payload
