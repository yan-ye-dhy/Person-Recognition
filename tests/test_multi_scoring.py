"""Offline v2 protocol, matching, unknown identity and denominator tests."""

from copy import deepcopy
import json

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.config import Work
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.parsing import parse_response
from vcrb.multi.prompts import AB_RULES, SYSTEM_PROMPT, build_user_prompt
from vcrb.multi.schema import IgnoreRegion, MultiSample, PersonAnnotation, dataset_hash
from vcrb.multi.scoring import aggregate, estimated_cost, maximum_iou_matching, score_run, score_sample
from vcrb.runner.store import RunStore


@pytest.fixture
def matcher():
    return CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [
        CharacterEntry("安欣", aliases=["小欣"], actors=["张译"]), CharacterEntry("李响", actors=["李健"]),
    ])})


def sample(persons=None, **kwargs):
    return MultiSample(sample_id="s1", work_slug="kuangbiao", work_name="狂飙",
                       work_category="modern_drama", image_path="s1.jpg",
                       persons=persons if persons is not None else [
                           PersonAnnotation(1, [.05, .1, .4, .9], "安欣", "verified", "approved", evidence=[{"type": "synthetic_test"}]),
                           PersonAnnotation(2, [.55, .1, .9, .9], "李响", "verified", "approved", evidence=[{"type": "synthetic_test"}]),
                       ], foreground_complete=kwargs.pop("foreground_complete", True), **kwargs)


def response(persons):
    return json.dumps({"persons": persons}, ensure_ascii=False)


def test_prompts_ab_have_only_work_prefix_difference_and_c_has_no_gt_count_or_ids():
    work = Work("kuangbiao", "狂飙", "modern_drama", "电视剧", 2023)
    a = build_user_prompt("A", person_ids=[1, 2, 987654])
    b = build_user_prompt("B", work, [1, 2, 987654])
    c = build_user_prompt("C", work, [1, 2, 987654])
    assert b == "这张图片出自电视剧《狂飙》。\n" + a and a == AB_RULES
    assert "狂飙" not in a and "987654" not in c and "person_id" not in c
    assert not any(name in a + b + c + SYSTEM_PROMPT for name in ("安欣", "李响", "张译"))
    with pytest.raises(ValueError):
        build_user_prompt("A", work)
    with pytest.raises(ValueError):
        build_user_prompt("C")


def test_ab_scores_ids_not_response_order_and_accepts_confirmed_alias(matcher):
    s = sample()
    parsed = parse_response(response([{"person_id": 2, "character": "李响"}, {"person_id": 1, "character": "小欣"}]), "B", [1, 2])
    row = score_sample(s, "B", parsed, matcher)
    assert row["identity_correct"] == 2 and row["image_correct"] is True
    swapped = parse_response(response([{"person_id": 2, "character": "安欣"}, {"person_id": 1, "character": "李响"}]), "A", [1, 2])
    assert score_sample(s, "A", swapped, matcher)["identity_correct"] == 0


def test_missing_and_extra_ids_are_explicit_and_block_image_exactness(matcher):
    missing = parse_response(response([{"person_id": 1, "character": "安欣"}]), "A", [1, 2])
    row = score_sample(sample(), "A", missing, matcher)
    assert row["identity_correct"] == 1 and row["format_error"] and row["image_correct"] is False
    assert row["person_details"][1]["outcome"] == "missing_id"
    m = aggregate([row], "A")
    assert m["person_accuracy"]["value"] == .5 and m["person_accuracy"]["denominator"] == 2
    extra = parse_response(response([{"person_id": 1, "character": "安欣"}, {"person_id": 2, "character": "李响"}, {"person_id": 99, "character": None}]), "B", [1, 2])
    row = score_sample(sample(), "B", extra, matcher)
    assert row["identity_correct"] == 2 and row["extra_ids"] == 1 and row["image_correct"] is False


@pytest.mark.parametrize("text", [
    '{"persons":[{"person_id":1,"character":"安欣"},{"person_id":1,"character":"李响"}]}',
    '{"persons":[{"person_id":true,"character":"安欣"}]}',
    '{"persons":[{"person_id":1,"character":["安欣","李响"]}]}',
    '{"persons":[{"person_id":1,"character":"安欣/李响"}]}',
    '{"persons":[{"person_id":1,"character":"安欣或李响"}]}',
    '{"persons":[{"person_id":1,"character":"安欣","alternatives":["李响"]}]}',
    '{"persons":[],"persons":[{"person_id":1,"character":"安欣"}]}',
    '{"persons":[{"person_id":1}]}', '{"persons":[{"person_id":1,"character":""}]}',
    '{"character_name":"安欣"}', 'invalid JSON',
])
def test_malformed_or_v1_output_never_silently_becomes_correct(matcher, text):
    parsed = parse_response(text, "A", [1, 2])
    assert not parsed.parse_ok and parsed.issues
    row = score_sample(sample(), "A", parsed, matcher)
    assert row["identity_correct"] == 0 and row["image_correct"] is False
    assert all(p["outcome"] == "parse_error" for p in row["person_details"])


def test_unknown_and_pending_are_not_ordinary_name_truth(matcher):
    persons = sample().persons
    persons[1] = PersonAnnotation(2, persons[1].bbox, None, "unknown", "approved")
    persons.append(PersonAnnotation(3, [.4, .1, .5, .9], "可能的人物", "needs_review", "pending"))
    s = sample(persons)
    parsed = parse_response(response([{"person_id": 1, "character": "安欣"}, {"person_id": 2, "character": None}, {"person_id": 3, "character": "任意猜测"}]), "A", [1, 2, 3])
    row = score_sample(s, "A", parsed, matcher)
    metric = aggregate([row], "A")
    assert metric["verified_targets"] == 1 and metric["person_accuracy"]["value"] == 1
    assert row["unknown_null"] == 1 and row["pending_count"] == 1 and row["image_correct"] is None
    assert row["person_details"][2]["gt_character"] is None
    assert metric["image_exact_accuracy"]["denominator"] == 0
    guessed = parse_response(response([{"person_id": 1, "character": "安欣"}, {"person_id": 2, "character": "李响"}, {"person_id": 3, "character": None}]), "A", [1, 2, 3])
    assert score_sample(s, "A", guessed, matcher)["unknown_guess"] == 1


def test_actor_name_uncertainty_refusal_and_api_denominators_remain_separate(matcher):
    parsed = parse_response(response([{"person_id": 1, "character": "张译"}, {"person_id": 2, "character": None}]), "A", [1, 2])
    row = score_sample(sample(), "A", parsed, matcher)
    assert [p["outcome"] for p in row["person_details"]] == ["actor_name", "uncertain"]
    refusal = parse_response("抱歉，我不能进行人脸识别", "A", [1, 2])
    refused = score_sample(sample(), "A", refusal, matcher)
    assert refused["refused"] and not refused["format_error"]
    fail = score_sample(sample(), "A", parsed, matcher, api_state="api_failure")
    notrun = score_sample(sample(), "A", parsed, matcher, api_state="not_run")
    metric = aggregate([row, refused, fail, notrun], "A")
    assert metric["person_accuracy"]["denominator"] == 4
    assert metric["person_accuracy_e2e"]["denominator"] == 8
    assert metric["api_failure_rate"]["value"] == .25 and metric["not_run"] == 1


def test_hungarian_matching_maximizes_cardinality_before_iou():
    gt = [[0, 0, .6, 1], [.2, 0, .8, 1]]
    predictions = [[.05, 0, .65, 1], [0, 0, .4, 1]]
    pairs = maximum_iou_matching(gt, predictions, .5)
    assert {(g, p) for g, p, _iou in pairs} == {(0, 1), (1, 0)}


def test_c_matching_is_by_geometry_and_wrong_identity_cannot_use_another_gt(matcher):
    s = sample()
    parsed = parse_response(response([{"bbox": p.bbox, "character": name} for p, name in zip(s.persons, ("李响", "安欣"))]), "C")
    row = score_sample(s, "C", parsed, matcher)
    metric = aggregate([row], "C")
    assert metric["localization"]["f1"] == 1
    assert metric["matched_identity_accuracy"]["value"] == 0
    assert metric["end_to_end"]["f1"] == 0 and row["identity_errors"] == 2
    assert row["image_correct"] is False


def test_c_one_prediction_cannot_match_two_gt_and_same_role_not_deduplicated(matcher):
    s = sample()
    s.persons[1].character = "安欣"
    one = parse_response(response([{"bbox": s.persons[0].bbox, "character": "安欣"}]), "C")
    row = score_sample(s, "C", one, matcher)
    assert row["localization_tp"] == 1 and row["localization_fn"] == 1
    two = parse_response(response([{"bbox": p.bbox, "character": p.character} for p in s.persons]), "C")
    assert two.parse_ok and two.warnings
    assert score_sample(s, "C", two, matcher)["image_correct"] is True


def test_c_ignore_suppresses_only_unmatched_background_and_foreground_matches_first(matcher):
    s = sample(ignore_regions=[IgnoreRegion([.01, .01, .99, .99])])
    parsed = parse_response(response([{"bbox": p.bbox, "character": p.character} for p in s.persons] + [{"bbox": [.42, .2, .5, .8], "character": None}]), "C")
    row = score_sample(s, "C", parsed, matcher)
    assert row["localization_tp"] == 2 and row["ignored_predictions"] == 1
    assert row["localization_fp"] == 0 and row["image_correct"] is True
    assert aggregate([row], "C")["end_to_end"]["f1"] == 1


@pytest.mark.parametrize("boxes", [
    [[-.1, .1, .4, .9]], [[0, .1, 1.1, .9]], [[0, .1, float('nan'), .9]],
    [[.4, .1, .1, .9]], [[.1, .1, .4, .9], [.1, .1, .4, .9]],
])
def test_c_invalid_and_duplicate_positions_are_format_failure_not_correct(matcher, boxes):
    parsed = parse_response(response([{"bbox": box, "character": "安欣"} for box in boxes]), "C")
    assert not parsed.parse_ok
    row = score_sample(sample(), "C", parsed, matcher)
    assert row["localization_tp"] == row["identity_correct"] == 0
    assert row["localization_fn"] == 2 and row["image_correct"] is False


def test_c_unknown_and_pending_matches_excluded_from_identity_name_denominators(matcher):
    persons = sample().persons
    persons[1] = PersonAnnotation(2, persons[1].bbox, None, "unknown", "approved")
    persons.append(PersonAnnotation(3, [.42, .2, .5, .8], None, "needs_review", "pending"))
    s = sample(persons)
    parsed = parse_response(response([{"bbox": p.bbox, "character": name} for p, name in zip(persons, ("安欣", None, "任意猜测"))]), "C")
    row = score_sample(s, "C", parsed, matcher)
    metric = aggregate([row], "C")
    assert metric["localization"]["tp"] == 3 and metric["end_to_end"]["tp"] == 1
    assert metric["end_to_end"]["fp"] == 0 and metric["end_to_end"]["fn"] == 0
    assert metric["matched_identity_accuracy"]["denominator"] == 1 and row["unknown_null"] == 1
    assert row["image_correct"] is None


def test_incomplete_foreground_cannot_claim_precision_from_missing_background(matcher):
    s = sample(foreground_complete=False)
    parsed = parse_response(response([{"bbox": s.persons[0].bbox, "character": "安欣"}, {"bbox": [.42, .2, .5, .8], "character": None}]), "C")
    row = score_sample(s, "C", parsed, matcher)
    metric = aggregate([row], "C")
    assert metric["localization"]["precision"]["value"] is None and metric["localization"]["f1"] is None
    assert metric["diagnostic_localization_counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert metric["unassessed_predictions"] == 1 and metric["matched_identity_accuracy"]["value"] == 1


def make_result(tmp_path, matcher):
    samples = [sample()]
    store = RunStore(tmp_path / "multi_runs.db")
    store.create_run("multi_mock", ["A", "B", "C"], ["synthetic-model"], "multi-v2-abc-1",
                     dataset_hash(samples), "snapshot.json", {"benchmark_protocol": "multi-v2", "is_mock": True})
    for exp in ("A", "B", "C"):
        persons = [{"bbox": p.bbox, "character": p.character} if exp == "C" else {"person_id": p.person_id, "character": p.character} for p in samples[0].persons]
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": .001}
        raw = {"response": {"raw_text": response(persons), "usage": usage}}
        store.upsert_record({"run_id": "multi_mock", "model_id": "synthetic-model", "experiment": exp,
                             "sample_id": "s1", "work_slug": "kuangbiao", "attempt": 1, "ok": 1,
                             "http_status": 200, "latency_ms": 120, "error_type": None, "error_message": None,
                             "served_model": "synthetic-model", "usage": json.dumps(usage), "raw_json": json.dumps(raw),
                             "raw_file": "mock.jsonl", "created_at": "2026-11-13"})
    return score_run(store, "multi_mock", samples, matcher, {}), store, samples


def test_full_run_keeps_abc_and_denominators_latency_usage_cost_quality_distinct(tmp_path, matcher):
    result, store, samples = make_result(tmp_path, matcher)
    assert result.is_mock and len(result.metrics) == 3
    assert result.metrics["synthetic-model|A"]["person_accuracy"]["value"] == 1
    c = result.metrics["synthetic-model|C"]
    assert c["localization"]["f1"] == c["end_to_end"]["f1"] == 1
    assert c["latency_ms_p50"] == c["latency_ms_p95"] == 120
    assert c["usage"]["total_tokens"] == 15 and c["estimated_cost_by_currency"] == {"USD": .001}
    assert result.by_work["synthetic-model|C"]["kuangbiao"]["verified_targets"] == 2
    assert result.by_person_count["synthetic-model|B"]["2"]["person_accuracy"]["denominator"] == 2
    assert result.ab_effects["synthetic-model"]["paired_verified_targets"] == 2
    changed = deepcopy(samples)
    changed[0].persons[0].character = "李响"
    with pytest.raises(ValueError, match="snapshot hash"):
        score_run(store, "multi_mock", changed, matcher, {})
    store.conn.execute("UPDATE runs SET fairness_meta='{}' WHERE run_id='multi_mock'")
    store.conn.commit()
    with pytest.raises(ValueError, match="v1 responses"):
        score_run(store, "multi_mock", samples, matcher, {})
    store.close()


def test_configured_token_cost_requires_prices_and_usage_and_never_calls_missing_free():
    prices = {"input_cny_per_million": 2, "output_cny_per_million": 8}
    cost, currency, method = estimated_cost({"prompt_tokens": 1000, "completion_tokens": 250}, prices)
    assert cost == .004 and currency == "CNY" and "not_full_bill" in method
    assert estimated_cost({}, prices) == (None, None, None)
    assert estimated_cost({"prompt_tokens": 1000, "completion_tokens": 250}, {}) == (None, None, None)
    assert estimated_cost({"prompt_tokens": 0, "completion_tokens": 0}, prices)[0] == 0
    assert estimated_cost({"prompt_tokens": True, "completion_tokens": 0}, prices)[0] is None


def test_optional_none_pricing_metadata_scores_existing_mock_records_without_reexecution(tmp_path, matcher):
    _result, store, samples = make_result(tmp_path, matcher)
    meta = store.get_run("multi_mock")["fairness_meta"]
    meta["pricing"] = None
    store.conn.execute("UPDATE runs SET fairness_meta=? WHERE run_id='multi_mock'", (json.dumps(meta),))
    store.conn.commit()
    raw_before = store.conn.execute("SELECT raw_json FROM records ORDER BY experiment").fetchall()
    result = score_run(store, "multi_mock", samples, matcher, {})
    assert len(result.records) == 3 and result.policy["pricing_sources"] == {}
    assert raw_before == store.conn.execute("SELECT raw_json FROM records ORDER BY experiment").fetchall()
    store.close()
