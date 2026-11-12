"""共享测试夹具：合成图片（PIL 生成，全部离线）+ 数据工厂。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from vcrb.config import Work
from vcrb.dataset.schema import Sample


def make_image_bytes(
    size: tuple[int, int] = (640, 360),
    color: tuple[int, int, int] = (30, 60, 120),
    fmt: str = "JPEG",
) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_image(tmp_path):
    """返回工厂：写一张合成图片到临时目录并返回路径。"""

    counter = {"n": 0}

    def _factory(
        name: str | None = None,
        size: tuple[int, int] = (640, 360),
        color: tuple[int, int, int] = (30, 60, 120),
        fmt: str = "JPEG",
    ):
        counter["n"] += 1
        if name is None:
            ext = fmt.lower()
            name = f"img_{counter['n']:04d}.{ext}"
        path = tmp_path / name
        path.write_bytes(make_image_bytes(size, color, fmt))
        return path

    return _factory


# ---------------------------------------------------------------- 工厂


def make_work(slug: str = "kuangbiao") -> Work:
    meta = {
        "kuangbiao": ("狂飙", "modern_drama", "电视剧", 2023,
                      ("狂飙 电视剧 剧照 高启强", "狂飙 电视剧 截图 角色", "狂飙 安欣 剧照")),
        "langyabang": ("琅琊榜", "costume_drama", "电视剧", 2015,
                       ("琅琊榜 电视剧 剧照 梅长苏", "琅琊榜 靖王 剧照")),
        "gongfunvzu": ("功夫女足", "recent_movie", "电影", None,
                       ("功夫女足 电影 剧照", "功夫女足 电影 截图 角色")),
        "jianlai": ("剑来", "animation", "动画", 2024,
                    ("剑来 动画 剧照 陈平安", "剑来 动漫 截图 角色")),
    }
    name, category, media_type, year, queries = meta[slug]
    return Work(slug=slug, name=name, category=category, media_type=media_type,
                year=year, search_queries=queries)


def make_sample(
    sample_id: str = "kuangbiao_0001",
    work_slug: str = "kuangbiao",
    target_character: str = "高启强",
    image_path: str | None = None,
    **overrides,
) -> Sample:
    work = make_work(work_slug)
    defaults = dict(
        sample_id=sample_id,
        work_slug=work_slug,
        work_name=work.name,
        work_category=work.category,
        image_path=image_path or f"{work_slug}/{sample_id}.jpg",
        target_character=target_character,
        character_importance="main",
        visual_difficulty="normal",
        is_multi_person=False,
        character_aliases=["强哥"],
        tags=["front_face"],
        target_bbox=[0.2, 0.1, 0.6, 0.9],
        annotation_status="verified",
        source_type="official_still",
    )
    defaults.update(overrides)
    return Sample(**defaults)


def build_tiny_dataset(dataset_dir: Path, n_per_work: int = 2) -> tuple[list[Sample], dict[str, Work]]:
    """在 dataset_dir 下生成微型正式集（真实图片文件 + Sample 列表）。"""
    works = {slug: make_work(slug) for slug in ("kuangbiao", "langyabang")}
    samples: list[Sample] = []
    idx = 0
    characters = {
        "kuangbiao": [("高启强", "main", "normal", False), ("陈书婷", "key_supporting", "side_face", True)],
        "langyabang": [("梅长苏", "main", "normal", False), ("靖王", "key_supporting", "normal", True)],
    }
    for slug, chars in characters.items():
        (dataset_dir / slug).mkdir(parents=True, exist_ok=True)
        for (char, importance, difficulty, multi) in chars[:n_per_work]:
            idx += 1
            sample_id = f"{slug}_{idx:04d}"
            rel = f"{slug}/{sample_id}.jpg"
            img = Image.new("RGB", (640, 360), (idx * 40 % 255, 60, 120))
            img.save(dataset_dir / rel, format="JPEG")
            samples.append(
                make_sample(
                    sample_id=sample_id,
                    work_slug=slug,
                    target_character=char,
                    image_path=rel,
                    character_importance=importance,
                    visual_difficulty=difficulty,
                    is_multi_person=multi,
                    character_aliases=[],
                )
            )
    return samples, works


def write_dataset_json(dataset_dir: Path, samples: list[Sample]) -> Path:
    path = dataset_dir / "dataset.json"
    path.write_text(
        json.dumps(
            {"exported_at": "test", "total": len(samples), "samples": [s.to_dict() for s in samples]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path

