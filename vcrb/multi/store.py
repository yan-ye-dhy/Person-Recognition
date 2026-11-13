"""Atomic v2 JSON persistence with optimistic concurrency and append-only audits."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import MultiSample, dataset_stats, renumber_persons


class ConcurrentModificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MultiAnnotationStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.audit_path = self.path.parent / "annotation_changes.jsonl"
        self._lock = threading.RLock()
        self.reload()

    def reload(self) -> None:
        raw = self.path.read_bytes()
        self._sha256 = hashlib.sha256(raw).hexdigest()
        self._payload = json.loads(raw)
        if str(self._payload.get("schema_version", "")).split(".")[0] != "2":
            raise ValueError("not a version 2 dataset")

    @property
    def revision(self) -> str:
        return self._sha256

    def samples(self, work_slug: str | None = None) -> list[MultiSample]:
        return [MultiSample.from_dict(copy.deepcopy(s)) for s in self._payload["samples"] if work_slug is None or s["work_slug"] == work_slug]

    def get(self, sample_id: str) -> MultiSample:
        return next((s for s in self.samples() if s.sample_id == sample_id), None) or self._missing(sample_id)

    @staticmethod
    def _missing(sample_id: str):
        raise KeyError(sample_id)

    def stats(self) -> dict[str, Any]:
        return dataset_stats(self.samples())

    @contextmanager
    def _file_lock(self):
        with self._lock, self.path.with_suffix(".lock").open("a+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows
                import msvcrt
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def save_sample(self, sample: MultiSample, *, actor: str, reason: str, evidence: list[dict[str, Any]] | None = None, expected_revision: str | None = None) -> MultiSample:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and modification reason are required")
        sample = MultiSample.from_dict(sample.to_dict())
        sample.persons = renumber_persons(sample.persons)
        errors = sample.validate()
        if errors:
            raise ValueError("; ".join(errors))
        with self._file_lock():
            current_raw = self.path.read_bytes()
            current_sha = hashlib.sha256(current_raw).hexdigest()
            if current_sha != (expected_revision or self._sha256):
                raise ConcurrentModificationError("Dataset changed in another session. Reload before saving; no edits were overwritten.")
            payload = json.loads(current_raw)
            index = next((i for i, s in enumerate(payload["samples"]) if s["sample_id"] == sample.sample_id), None)
            if index is None:
                raise KeyError(sample.sample_id)
            before = payload["samples"][index]
            for name in ("sample_id", "work_slug", "work_name", "work_category", "image_path", "image_width", "image_height", "image_sha256"):
                if before.get(name) != getattr(sample, name):
                    raise ValueError(f"annotation edits cannot change source field {name}")
            after = sample.to_dict()
            if before == after:
                return sample
            event = {"event": "sample_update", "at": utc_now(), "actor": actor, "sample_id": sample.sample_id, "reason": reason, "evidence": evidence or [], "before": before, "after": after, "source_dataset_sha256": current_sha}
            # Write the complete intent first; a failed audit append never commits labels.
            with self.audit_path.open("a", encoding="utf-8") as audit:
                audit.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                audit.flush()
                os.fsync(audit.fileno())
            payload["samples"][index] = after
            payload["updated_at"] = utc_now()
            atomic_write_json(self.path, payload)
            self.reload()
        return sample

    update_sample = save_sample
