"""响应解析与拒答检测测试。"""

from __future__ import annotations

from vcrb.evaluation.parser import (
    DEFAULT_REFUSAL_KEYWORDS,
    extract_json_text,
    looks_like_refusal,
    parse_response,
)


def test_plain_json():
    p = parse_response('{"character_name": "高启强", "is_uncertain": false, "reason": "外貌判断"}')
    assert p.parse_ok
    assert p.character_name == "高启强"
    assert p.is_uncertain is False
    assert p.reason == "外貌判断"


def test_json_in_code_fence():
    raw = '```json\n{"character_name": "梅长苏", "is_uncertain": false, "reason": "x"}\n```'
    p = parse_response(raw)
    assert p.parse_ok and p.character_name == "梅长苏"


def test_json_with_surrounding_prose():
    raw = '好的，我的判断如下：{"character_name": "安欣", "is_uncertain": false, "reason": "制服"} 以上。'
    p = parse_response(raw)
    assert p.parse_ok and p.character_name == "安欣"


def test_uncertain_null_name():
    p = parse_response('{"character_name": null, "is_uncertain": true, "reason": "看不清"}')
    assert p.parse_ok
    assert p.character_name is None
    assert p.is_uncertain is True


def test_empty_string_name_treated_as_null():
    p = parse_response('{"character_name": "", "is_uncertain": "true", "reason": "不确定"}')
    assert p.parse_ok
    assert p.character_name is None
    assert p.is_uncertain is True  # 字符串 true 也被识别


def test_invalid_json_is_parse_error():
    p = parse_response("我认为这个人是主角。")
    assert not p.parse_ok
    assert "JSON" in p.error


def test_single_quotes_invalid():
    p = parse_response("{'character_name': '高启强'}")
    assert not p.parse_ok


def test_empty_response():
    p = parse_response(None)
    assert not p.parse_ok
    assert "为空" in p.error


def test_tolerates_camel_case_keys():
    p = parse_response('{"characterName": "宁姚", "isUncertain": false, "reason": ""}')
    assert p.parse_ok and p.character_name == "宁姚"


def test_extract_json_text():
    assert extract_json_text('x {"a":1} y') == '{"a":1}'
    assert extract_json_text("no braces") == "no braces"


def test_refusal_detection():
    assert looks_like_refusal("很抱歉，我无法识别人脸身份信息。")
    assert looks_like_refusal("出于隐私考虑，不能提供。")
    assert not looks_like_refusal("无法确定该角色，可能是配角")
    assert not looks_like_refusal("")
    # 自定义词表
    assert looks_like_refusal("自定义拒绝词", ["自定义拒绝词"])
    # 默认表非空且含关键项
    assert "无法识别人脸" in DEFAULT_REFUSAL_KEYWORDS
