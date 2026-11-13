"""角色名匹配测试：别名/演员名/子串/模糊转复核/归一化。"""

from __future__ import annotations

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher, normalize_name, split_role_marker


def _roster() -> Roster:
    return Roster(
        work_slug="kuangbiao",
        work_name="狂飙",
        characters=[
            CharacterEntry(name="高启强", aliases=["强哥", "老高"], actors=["张颂文"], importance="main"),
            CharacterEntry(name="高启盛", aliases=[], actors=["苏小玎"], importance="key_supporting"),
            CharacterEntry(name="安欣", aliases=["安警官"], actors=["张译"], importance="main"),
        ],
    )


def test_normalize_name():
    assert normalize_name(" 高启强 ") == "高启强"
    assert normalize_name("高启强（张颂文饰）") == "高启强"
    assert normalize_name("「安欣」") == "安欣"
    assert normalize_name("安　欣") == "安欣"  # 全角空格
    assert normalize_name("Ｑｗｅ") == "Qwe"  # NFKC 全半角
    assert normalize_name("") == ""


def test_split_role_marker():
    assert split_role_marker("张颂文饰演的高启强") == "高启强"
    assert split_role_marker("张颂文饰高启强") == "高启强"
    assert split_role_marker("高启强") == "高启强"


def test_exact_and_sample_alias():
    m = CharacterMatcher({_roster().work_slug: _roster()})
    assert m.match("高启强", "kuangbiao", "高启强").kind == "exact"
    assert m.match("强哥", "kuangbiao", "高启强", ["强哥"]).kind == "exact"


def test_roster_alias_hit():
    m = CharacterMatcher({"kuangbiao": _roster()})
    r = m.match("老高", "kuangbiao", "高启强", [])  # 别名来自名单而非样本
    assert r.kind == "alias" and r.is_correct


def test_substring_match():
    m = CharacterMatcher({"kuangbiao": _roster()})
    r = m.match("电视剧《狂飙》里的高启强", "kuangbiao", "高启强", [])
    assert r.kind == "substring" and r.is_correct


@pytest.mark.parametrize("predicted,canonical,aliases", [
    ("皇子", "二皇子", []),
    ("皇子", "三皇子", []),
    ("殿下", "太子殿下", []),
    ("院长", "陈萍萍", ["陈院长"]),
    ("高启", "高启强", []),
])
def test_unregistered_name_fragment_is_not_correct(predicted, canonical, aliases):
    matcher = CharacterMatcher({})
    result = matcher.match(predicted, "synthetic_work", canonical, aliases)
    assert not result.is_correct


def test_short_name_must_be_explicitly_confirmed_alias():
    matcher = CharacterMatcher({})
    assert not matcher.match("宝瓶", "synthetic_work", "李宝瓶").is_correct
    assert matcher.match("宝瓶", "synthetic_work", "李宝瓶", ["宝瓶"]).is_correct


@pytest.mark.parametrize("predicted", [
    "电视剧《狂飙》里的高启强",
    "这个角色是高启强，穿着西装。",
    "张颂文饰演的高启强",
    "高启强（张颂文饰）",
])
def test_complete_name_in_single_identification_is_correct(predicted):
    matcher = CharacterMatcher({"kuangbiao": _roster()})
    assert matcher.match(predicted, "kuangbiao", "高启强").is_correct


@pytest.mark.parametrize("predicted", [
    "高启强或安欣",
    "高启强、安欣",
    "高启强/安欣",
    "高启强，安欣",
    "高启强和安欣",
    "可能是高启强，也可能是安欣",
    "高启强（张颂文饰）或安欣（张译饰）",
    "高启强或者未登记的新角色",
    "强哥或安警官",
])
def test_multiple_character_candidates_require_review(predicted):
    matcher = CharacterMatcher({"kuangbiao": _roster()})
    for canonical in ("高启强", "安欣"):
        result = matcher.match(predicted, "kuangbiao", canonical)
        assert result.kind == "ambiguous"
        assert result.needs_review and not result.is_correct


def test_scoring_does_not_count_ambiguous_or_fragment_as_correct(tmp_path):
    """Exercise the actual scoring path with invented responses and a temp DB."""
    import json

    from tests.conftest import make_sample
    from vcrb.evaluation.scoring import score_run
    from vcrb.runner.store import RunStore

    store = RunStore(tmp_path / "synthetic_runs.db")
    store.create_run("synthetic", experiments=["A"], models=["fixture-model"],
                     prompt_version="v1", dataset_hash="fixture", dataset_path="fixture", fairness_meta={})
    answers = ["高启强或安欣", "高启", "电视剧狂飙里的高启强", "张颂文"]
    samples = [make_sample(f"kuangbiao_{i:04d}") for i in range(len(answers))]
    for sample, answer in zip(samples, answers):
        response = json.dumps({"character_name": answer, "is_uncertain": False, "reason": "自造边界测试"}, ensure_ascii=False)
        store.upsert_record(dict(run_id="synthetic", model_id="fixture-model", experiment="A",
            sample_id=sample.sample_id, work_slug="kuangbiao", attempt=1, ok=1, http_status=200,
            latency_ms=1, error_type=None, error_message=None, served_model="fixture-model", usage="{}",
            raw_json=json.dumps({"response": {"raw_text": response}}, ensure_ascii=False), raw_file="",
            created_at="2026-11-13T00:00:00+00:00"))
    result = score_run(store, "synthetic", samples, CharacterMatcher({"kuangbiao": _roster()}),
                       {"scoring": {"refusal_keywords": []}})
    assert [r.outcome for r in result.records] == ["wrong_name", "wrong_name", "correct", "actor_name"]
    assert [r.needs_review for r in result.records] == [True, False, False, False]
    assert "多个角色候选" in result.records[0].detail
    assert result.metrics[("fixture-model", "A")].counts["correct"] == 1
    assert result.metrics[("fixture-model", "A")].accuracy_valid == .25


def test_actor_name_is_not_correct():
    m = CharacterMatcher({"kuangbiao": _roster()})
    r = m.match("张颂文", "kuangbiao", "高启强", [])
    assert r.kind == "actor" and not r.is_correct


def test_actor_of_other_character():
    m = CharacterMatcher({"kuangbiao": _roster()})
    r = m.match("张译", "kuangbiao", "高启强", [])
    assert r.kind == "actor_of_other"


def test_other_character():
    m = CharacterMatcher({"kuangbiao": _roster()})
    r = m.match("高启盛", "kuangbiao", "高启强", [])
    assert r.kind == "other_character" and not r.is_correct


def test_fuzzy_goes_to_review_not_correct():
    m = CharacterMatcher({"kuangbiao": _roster()}, fuzzy_threshold=0.6)
    r = m.match("高歧强", "kuangbiao", "高启强", [])
    assert r.kind == "fuzzy"
    assert r.needs_review and not r.is_correct


def test_fuzzy_below_threshold_is_none():
    m = CharacterMatcher({"kuangbiao": _roster()})  # 默认 0.9
    r = m.match("路人甲乙丙丁", "kuangbiao", "高启强", [])
    assert r.kind == "none"


def test_empty_prediction():
    m = CharacterMatcher({"kuangbiao": _roster()})
    assert m.match(None, "kuangbiao", "高启强", []).kind == "empty"
    assert m.match("", "kuangbiao", "高启强", []).kind == "empty"


def test_needs_review_alias_not_auto_correct():
    """needs_review 条目的别名未经人工确认，不得作为评分别名自动判对。"""
    roster = Roster(
        work_slug="gongfunvzu",
        work_name="功夫女足",
        characters=[
            CharacterEntry(
                name="双双", aliases=["大小姐"], actors=["张小斐"],
                importance="main", needs_review=True,
            ),
        ],
    )
    m = CharacterMatcher({"gongfunvzu": roster})
    # 未确认别名（且与 GT 名无包含关系）：不判对
    r = m.match("大小姐", "gongfunvzu", "双双", [])
    assert not r.is_correct and r.kind == "none"
    # canonical 名与演员名映射仍然生效
    assert m.match("双双", "gongfunvzu", "双双", []).kind == "exact"
    r2 = m.match("张小斐", "gongfunvzu", "双双", [])
    assert r2.kind == "actor" and not r2.is_correct


def test_confirmed_alias_still_matches():
    roster = Roster(
        work_slug="gongfunvzu",
        work_name="功夫女足",
        characters=[
            CharacterEntry(
                name="双双", aliases=["大小姐"], actors=["张小斐"],
                importance="main", needs_review=False,
            ),
        ],
    )
    m = CharacterMatcher({"gongfunvzu": roster})
    assert m.match("大小姐", "gongfunvzu", "双双", []).kind == "alias"
