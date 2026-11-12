"""配置加载：settings.yaml / works.yaml / 环境变量（API Key）。

支持 VCRB_CONFIG_DIR 环境变量覆盖配置目录（供测试使用）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .paths import CONFIG_DIR, ROOT

_LOADED_DOTENV = False


def _ensure_dotenv() -> None:
    """加载项目根目录 .env（仅一次；不存在则忽略）。"""
    global _LOADED_DOTENV
    if not _LOADED_DOTENV:
        load_dotenv(ROOT / ".env")
        load_dotenv()  # 也尊重用户已在 shell 中导出的变量
        _LOADED_DOTENV = True


class ConfigError(RuntimeError):
    """配置文件缺失或非法。"""


def config_dir() -> Path:
    return Path(os.environ.get("VCRB_CONFIG_DIR", str(CONFIG_DIR)))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - 语法错误路径
        raise ConfigError(f"YAML 解析失败：{path} ({exc})") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return data


# ---------------------------------------------------------------- works

VALID_CATEGORIES = {"modern_drama", "costume_drama", "recent_movie", "animation"}


@dataclass(frozen=True)
class Work:
    slug: str
    name: str
    category: str
    media_type: str
    year: int | None
    search_queries: tuple[str, ...] = ()
    # 作品归属核验（screening）用：
    trusted_domains: tuple[str, ...] = ()   # 可信来源域（官方/权威平台，仍需搭配作品标识才判 verified）
    evidence_aliases: tuple[str, ...] = ()  # 作品标识补充（英文名/别名，用于标题与 URL 匹配）

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "category": self.category,
            "media_type": self.media_type,
            "year": self.year,
            "search_queries": list(self.search_queries),
            "trusted_domains": list(self.trusted_domains),
            "evidence_aliases": list(self.evidence_aliases),
        }


def load_works() -> list[Work]:
    raw = load_yaml(config_dir() / "works.yaml")
    entries = raw.get("works")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("works.yaml 缺少非空 works 列表")
    works: list[Work] = []
    seen: set[str] = set()
    for item in entries:
        missing = {"slug", "name", "category", "media_type"} - set(item or {})
        if missing:
            raise ConfigError(f"works.yaml 条目缺少字段 {sorted(missing)}：{item}")
        slug = str(item["slug"])
        if slug in seen:
            raise ConfigError(f"works.yaml slug 重复：{slug}")
        seen.add(slug)
        category = str(item["category"])
        if category not in VALID_CATEGORIES:
            raise ConfigError(
                f"未知作品类别 {category}（合法值：{sorted(VALID_CATEGORIES)}）"
            )
        year = item.get("year")
        works.append(
            Work(
                slug=slug,
                name=str(item["name"]),
                category=category,
                media_type=str(item["media_type"]),
                year=int(year) if year is not None else None,
                search_queries=tuple(str(q) for q in item.get("search_queries", [])),
                trusted_domains=tuple(str(d).lower() for d in item.get("trusted_domains", [])),
                evidence_aliases=tuple(str(a) for a in item.get("evidence_aliases", [])),
            )
        )
    return works


def get_work(slug: str) -> Work:
    for w in load_works():
        if w.slug == slug:
            return w
    raise ConfigError(f"未知作品 slug：{slug}")


# ---------------------------------------------------------------- settings

_DEFAULT_SETTINGS: dict[str, Any] = {
    "retry": {
        "max_attempts": 4,
        "backoff_base_seconds": 2.0,
        "timeout_seconds": 120,
    },
    "generation": {"temperature": 0.1, "max_tokens": 512},
    "collect": {
        "request_timeout_seconds": 15,
        "max_images_per_work": 150,
        "min_image_bytes": 10240,
        "user_agent": "vcrb/0.1",
    },
    "clean": {
        "min_width": 300,
        "min_height": 200,
        "phash_hamming_threshold": 8,
        "allowed_formats": ["jpg", "jpeg", "png", "webp"],
    },
    "export": {"target_per_group": 20, "target_per_work": 100},
    "scoring": {"fuzzy_review_threshold": 0.9, "refusal_keywords": []},
    "pilot": {"per_model_connectivity": 1, "per_model_small": 10, "per_work_formal": 10},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings() -> dict[str, Any]:
    """settings.yaml 与内置默认值深度合并（yaml 优先）。"""
    path = config_dir() / "settings.yaml"
    user = load_yaml(path) if path.exists() else {}
    return _deep_merge(_DEFAULT_SETTINGS, user)


# ---------------------------------------------------------------- API keys

def get_api_key(env_name: str) -> str | None:
    _ensure_dotenv()
    value = os.environ.get(env_name, "").strip()
    return value or None


def require_api_key(env_name: str) -> str:
    key = get_api_key(env_name)
    if not key:
        raise ConfigError(
            f"缺少 API Key：请在 .env 或环境变量中设置 {env_name}（参考 .env.example）"
        )
    return key
