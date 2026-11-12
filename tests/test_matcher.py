"""角色名匹配测试：别名/演员名/子串/模糊转复核/归一化。"""

from __future__ import annotations

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
