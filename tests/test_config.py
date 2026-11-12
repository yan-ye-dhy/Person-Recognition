"""配置解析测试：settings/works/env。"""

from __future__ import annotations

import pytest

from vcrb.config import (
    ConfigError,
    load_settings,
    load_works,
    get_api_key,
    get_work,
)


def test_load_works_four_fixed_works():
    works = load_works()
    assert [w.slug for w in works] == [
        "kuangbiao",
        "langyabang",
        "gongfunvzu",
        "jianlai",
    ]
    by_slug = {w.slug: w for w in works}
    assert by_slug["kuangbiao"].name == "狂飙"
    assert by_slug["kuangbiao"].category == "modern_drama"
    assert by_slug["langyabang"].category == "costume_drama"
    assert by_slug["gongfunvzu"].category == "recent_movie"
    assert by_slug["gongfunvzu"].media_type == "电影"
    assert by_slug["jianlai"].category == "animation"
    # 功夫女足年份 2026-11-12 已核实（2026-07-11 上映，导演周星驰）
    assert by_slug["gongfunvzu"].year == 2026


def test_get_work_unknown_slug():
    with pytest.raises(ConfigError):
        get_work("nonexistent")


def test_settings_defaults_and_merge():
    settings = load_settings()
    assert settings["retry"]["max_attempts"] >= 1
    assert settings["generation"]["temperature"] == 0.1
    assert settings["scoring"]["fuzzy_review_threshold"] == 0.9
    # yaml 覆盖过默认值
    assert settings["clean"]["min_width"] == 300


def test_invalid_works_yaml(tmp_path, monkeypatch):
    bad = tmp_path / "works.yaml"
    bad.write_text("works:\n  - slug: x\n    name: X\n", encoding="utf-8")
    monkeypatch.setenv("VCRB_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError):
        load_works()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VCRB_TEST_KEY", "sk-test-123")
    assert get_api_key("VCRB_TEST_KEY") == "sk-test-123"
    monkeypatch.delenv("VCRB_TEST_KEY")
    assert get_api_key("VCRB_TEST_KEY") is None
