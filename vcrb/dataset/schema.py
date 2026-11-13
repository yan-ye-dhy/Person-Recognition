"""正式测试样本 schema 与校验。

三维度独立标注（用户批准的调整 2）：
  character_importance : main / key_supporting / supporting
  visual_difficulty    : normal / side_face / occluded / blurry / costume_change
  is_multi_person      : bool
五组采样配额（默认每部 5×8）由 sampling_group() 从三字段导出，用于分层选择与统计。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import load_works

IMPORTANCE_LEVELS = {"main", "key_supporting", "supporting"}
DIFFICULTY_LEVELS = {"normal", "side_face", "occluded", "blurry", "costume_change"}
SAMPLE_STATUSES = {"pending", "verified", "invalid", "uncertain"}

SAMPLING_GROUPS = [
    "main_character",
    "key_supporting_character",
    "supporting_character",
    "hard_case",
    "multi_person",
]

SOURCE_TYPES = {"official_still", "episode_screenshot", "frame_capture", "web_image", "manual", "unknown"}

REQUIRED_FIELDS = [
    "sample_id",
    "work_slug",
    "work_name",
    "work_category",
    "image_path",
    "target_character",
    "character_importance",
    "visual_difficulty",
    "is_multi_person",
    "target_bbox",
    "annotation_status",
]


def sampling_group(
    character_importance: str, visual_difficulty: str, is_multi_person: bool
) -> str:
    """三维标注映射到互斥采样组：多人优先，其次难度，最后人物重要度。"""
    if is_multi_person:
        return "multi_person"
    if visual_difficulty != "normal":
        return "hard_case"
    return {
        "main": "main_character",
        "key_supporting": "key_supporting_character",
        "supporting": "supporting_character",
    }[character_importance]


@dataclass
class Sample:
    sample_id: str
    work_slug: str
    work_name: str
    work_category: str
    image_path: str  # 相对 dataset.json 所在目录
    target_character: str
    character_importance: str = "supporting"
    visual_difficulty: str = "normal"
    is_multi_person: bool = False
    character_aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    target_bbox: list[float] | None = None
    image_width: int | None = None
    image_height: int | None = None
    annotation_status: str = "verified"
    source_type: str = "unknown"
    source_url: str = ""
    note: str = ""

    # ---- 派生 ----
    @property
    def sampling_group(self) -> str:
        return sampling_group(
            self.character_importance, self.visual_difficulty, self.is_multi_person
        )

    # ---- 校验 ----
    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.sample_id:
            errors.append("sample_id 为空")
        if self.work_slug not in {work.slug for work in load_works()}:
            errors.append(f"未知作品 slug：{self.work_slug}")
        if not self.target_character:
            errors.append("target_character 为空")
        if self.character_importance not in IMPORTANCE_LEVELS:
            errors.append(
                f"character_importance 非法：{self.character_importance}"
                f"（合法值 {sorted(IMPORTANCE_LEVELS)}）"
            )
        if self.visual_difficulty not in DIFFICULTY_LEVELS:
            errors.append(
                f"visual_difficulty 非法：{self.visual_difficulty}"
                f"（合法值 {sorted(DIFFICULTY_LEVELS)}）"
            )
        if self.annotation_status != "verified":
            errors.append(
                f"正式测试集要求 annotation_status=verified，当前 {self.annotation_status}"
            )
        if self.target_bbox is not None:
            bbox = self.target_bbox
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
            ):
                errors.append(f"target_bbox 必须是 4 个数字：{bbox}")
            else:
                x1, y1, x2, y2 = bbox
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    errors.append(f"target_bbox 坐标非法（需 0≤x1<x2≤1, 0≤y1<y2≤1）：{bbox}")
        if self.source_type not in SOURCE_TYPES:
            errors.append(f"source_type 非法：{self.source_type}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


def load_dataset(path: Path) -> list[Sample]:
    """加载 dataset.json（导出器产物）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("samples")
    if entries is None:
        raise ValueError(f"dataset.json 缺少 samples 字段：{path}")
    return [Sample.from_dict(e) for e in entries]


def validate_dataset(
    samples: list[Sample], dataset_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    """全量校验 + 汇总统计。返回 (错误列表, 统计信息)。"""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for s in samples:
        errors.extend(f"{s.sample_id}: {e}" for e in s.validate())
        if s.sample_id in seen_ids:
            errors.append(f"sample_id 重复：{s.sample_id}")
        seen_ids.add(s.sample_id)
        img = dataset_dir / s.image_path
        if not img.is_file():
            errors.append(f"{s.sample_id}: 图片不存在 {img}")
    return errors, dataset_stats(samples)


def dataset_stats(samples: list[Sample]) -> dict[str, Any]:
    by_work: dict[str, dict[str, int]] = {}
    for s in samples:
        bucket = by_work.setdefault(
            s.work_slug,
            {g: 0 for g in SAMPLING_GROUPS}
            | {"total": 0, "multi": 0, "difficulty_normal": 0},
        )
        bucket[s.sampling_group] += 1
        bucket["total"] += 1
        if s.is_multi_person:
            bucket["multi"] += 1
        if s.visual_difficulty == "normal":
            bucket["difficulty_normal"] += 1
    characters: dict[str, set[str]] = {}
    for s in samples:
        characters.setdefault(s.work_slug, set()).add(s.target_character)
    return {
        "total": len(samples),
        "by_work": by_work,
        "characters_per_work": {k: sorted(v) for k, v in characters.items()},
    }
