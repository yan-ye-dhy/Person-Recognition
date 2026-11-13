"""Seed v2 drafts from v1 without copying or modifying source images or labels."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..dataset.schema import load_dataset as load_v1_dataset
from .schema import MultiSample, PersonAnnotation, SCHEMA_VERSION
from .store import atomic_write_json, utc_now


def migrate_v1(source: Path, destination: Path) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Refusing to overwrite an existing dataset")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    samples = []
    for original in load_v1_dataset(source):
        image = source.parent / original.image_path
        if not image.is_file():
            raise FileNotFoundError(f"missing source image for {original.sample_id}")
        if original.target_bbox is None:
            raise ValueError(f"{original.sample_id}: cannot inherit a missing target bbox")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        timing = re.search(r"#t=([\d.]+)", original.source_url)
        episode = re.search(r"(?:s\d+e|第)(\d+)", original.source_url, re.I)
        evidence = {"type": "v1_inheritance", "source_sample_id": original.sample_id, "source_dataset_sha256": source_sha, "image_sha256": digest, "note": "Inherited single-target label; not independent identity verification. Remaining foreground people have not been checked."}
        if timing:
            evidence["time_seconds"] = float(timing[1])
        if episode:
            evidence["episode"] = int(episode[1])
        person = PersonAnnotation(person_id=1, bbox=list(original.target_bbox), character=original.target_character, identity_status="needs_review", review_status="pending", character_aliases=list(original.character_aliases), evidence=[dict(evidence)], annotation_id=f"{original.sample_id}_inherited")
        sample = MultiSample(sample_id=original.sample_id, work_slug=original.work_slug, work_name=original.work_name, work_category=original.work_category, image_path=os.path.relpath(image.resolve(), destination.parent).replace(os.sep, "/"), image_width=original.image_width, image_height=original.image_height, image_sha256=digest, persons=[person], foreground_complete=False, review_status="draft", evidence=[evidence], note="v2初始草稿：仅继承v1目标框与待复核身份，尚未补齐其他前景人物。")
        errors = sample.validate()
        if errors:
            raise ValueError(f"{sample.sample_id}: {'; '.join(errors)}")
        samples.append(sample.to_dict())
    payload = {"schema_version": SCHEMA_VERSION, "dataset_id": "multi_person_v2", "created_at": utc_now(), "source": {"kind": "v1_single_target_migration", "dataset_sha256": source_sha, "sample_count": len(samples), "review_policy": "All inherited identities need independent review; no approved samples."}, "samples": samples}
    atomic_write_json(destination, payload)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    return payload


migrate_dataset = migrate_v1
