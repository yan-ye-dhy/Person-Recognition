"""Prompt 模板测试：A/B 一致性与防泄露。"""

from __future__ import annotations

import pytest
from tests.conftest import make_work

from vcrb.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    describe_marking,
)


def test_a_b_differ_only_by_work_sentence():
    work = make_work("kuangbiao")
    a = build_user_prompt("A")
    b = build_user_prompt("B", work)
    prefix = f"这张截图来自{work.media_type}《{work.name}》。\n"
    assert b.startswith(prefix)
    assert b[len(prefix):] == a  # 其余逐字相同


def test_a_prompt_contents():
    a = build_user_prompt("A")
    assert "红色方框" in a
    assert "character_name" in a and "is_uncertain" in a and "reason" in a
    assert "演员的真实姓名" in a  # 明确要求角色名而非演员名


def test_a_forbids_work_info():
    with pytest.raises(ValueError):
        build_user_prompt("A", make_work("kuangbiao"))


def test_b_requires_work():
    with pytest.raises(ValueError):
        build_user_prompt("B")


def test_invalid_experiment():
    with pytest.raises(ValueError):
        build_user_prompt("C")


def test_lowercase_experiment_ok():
    assert build_user_prompt("a") == build_user_prompt("A")


def test_build_messages_shape():
    msgs = build_messages("B", make_work("jianlai"))
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert "《剑来》" in msgs[1]["content"]


def test_prompt_version_and_marking_recorded():
    assert PROMPT_VERSION == "v1"
    marking = describe_marking()
    assert "红色" in marking and "同一函数" in marking
