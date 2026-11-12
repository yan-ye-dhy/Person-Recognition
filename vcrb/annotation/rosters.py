"""作品角色名单（config/rosters/{slug}.yaml）。

用途：
1. 标注页的角色候选下拉；
2. 评分时的别名匹配与演员名识别（actors 命中 → actor_name，不算正确）。

数据原则：不确定的条目必须标 needs_review: true，等待人工确认，
绝不臆造角色名或演员映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import ConfigError, Work
from ..paths import ROSTERS_DIR

IMPORTANCE_VALUES = {"main", "key_supporting", "supporting"}


@dataclass
class CharacterEntry:
    name: str
    aliases: list[str] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    importance: str = "supporting"  # main / key_supporting / supporting
    notes: str = ""
    needs_review: bool = False      # 名单信息待人工确认

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "actors": self.actors,
            "importance": self.importance,
            "notes": self.notes,
            "needs_review": self.needs_review,
        }


@dataclass
class Roster:
    work_slug: str
    work_name: str
    characters: list[CharacterEntry] = field(default_factory=list)
    source_note: str = ""

    def names(self) -> list[str]:
        return [c.name for c in self.characters]

    def entry(self, name: str) -> CharacterEntry | None:
        for c in self.characters:
            if c.name == name:
                return c
        return None

    def add(self, entry: CharacterEntry) -> None:
        if self.entry(entry.name) is None:
            self.characters.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug,
            "work_name": self.work_name,
            "source_note": self.source_note,
            "characters": [c.to_dict() for c in self.characters],
        }


def roster_path(work_slug: str, rosters_dir: Path | None = None) -> Path:
    base = rosters_dir or ROSTERS_DIR
    return base / f"{work_slug}.yaml"


def load_roster(work: Work, rosters_dir: Path | None = None) -> Roster:
    path = roster_path(work.slug, rosters_dir)
    if not path.exists():
        return Roster(work_slug=work.slug, work_name=work.name)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    characters: list[CharacterEntry] = []
    for item in data.get("characters", []):
        if not item or not item.get("name"):
            continue
        importance = str(item.get("importance", "supporting"))
        if importance not in IMPORTANCE_VALUES:
            raise ConfigError(
                f"roster {work.slug}: 角色 {item.get('name')} importance 非法：{importance}"
            )
        characters.append(
            CharacterEntry(
                name=str(item["name"]),
                aliases=[str(a) for a in item.get("aliases", [])],
                actors=[str(a) for a in item.get("actors", [])],
                importance=importance,
                notes=str(item.get("notes", "") or ""),
                needs_review=bool(item.get("needs_review", False)),
            )
        )
    return Roster(
        work_slug=work.slug,
        work_name=work.name,
        characters=characters,
        source_note=str(data.get("source_note", "") or ""),
    )


def save_roster(roster: Roster, rosters_dir: Path | None = None) -> Path:
    path = roster_path(roster.work_slug, rosters_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(roster.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def build_template(work: Work, rosters_dir: Path | None = None) -> Path:
    """生成空白名单模板（供人工/后续调研填充）。"""
    roster = Roster(
        work_slug=work.slug,
        work_name=work.name,
        source_note=f"{work.name} 角色名单 —— 待人工核实补充（needs_review 条目未经确认不得用于评分别名）",
        characters=[
            CharacterEntry(
                name="示例角色（请替换）",
                aliases=["别名示例"],
                actors=["演员名示例"],
                importance="supporting",
                needs_review=True,
            )
        ],
    )
    return save_roster(roster, rosters_dir)
