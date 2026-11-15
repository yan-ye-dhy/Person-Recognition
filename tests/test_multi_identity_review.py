"""Identity-only diagnostic tests protect against format penalties and name inflation."""

from copy import deepcopy
import json

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.identity_review import apply_corrections, score_name_only, summarize_name_rows


@pytest.fixture
def matcher():
    return CharacterMatcher({"work": Roster("work", "测试", [
        CharacterEntry("安欣", aliases=["小欣"], actors=["张译"]),
        CharacterEntry("李响", actors=["李健"]),
    ])})


def sample(names=("安欣", "李响")):
    return {"sample_id":"s1", "work_slug":"work", "image_sha256":"sha", "persons":[
        {"person_id":i+1,"character":name,"identity_status":"verified","review_status":"approved",
         "character_aliases":[],"annotation_id":f"p{i+1}","bbox":[0,0,1,1]}
        for i,name in enumerate(names)
    ]}


def response(people):
    return json.dumps({"persons":people},ensure_ascii=False)


def test_integer_or_mixed_bbox_and_broken_ids_do_not_hide_a_correct_name(matcher):
    people = [{"bbox":[123,.1,900,.9],"person_id":"bad","character":"安欣"},
              {"bbox":None,"character":"李响"}]
    row = score_name_only(sample(),"C",response(people),matcher)
    assert row["correct"] == 2 and row["image_names_complete"]
    assert row["predictions"][0]["original_bbox"] == [123,.1,900,.9]


def test_swapped_ids_are_name_coverage_not_person_correspondence(matcher):
    row = score_name_only(sample(),"B",response([
        {"person_id":2,"character":"安欣"}, {"person_id":1,"character":"李响"}]),matcher)
    assert row["correct"] == 2
    assert row["predictions"][0]["matched_person_id"] == 1


def test_duplicate_correct_name_cannot_rescue_a_second_person(matcher):
    row = score_name_only(sample(),"C",response([{"character":"安欣"},{"character":"安欣"}]),matcher)
    assert row["correct"] == 1 and row["extra_names"] == 1
    assert not row["image_names_complete"]
    assert sum(p["outcome"] == "matched" for p in row["predictions"]) == 1


def test_extra_names_remain_visible_even_when_all_gt_names_covered(matcher):
    row = score_name_only(sample(),"C",response([
        {"character":"安欣"},{"character":"李响"},{"character":"路人甲"}]),matcher)
    aggregate = summarize_name_rows([row])
    assert row["correct"] == 2 and row["image_names_complete"]
    assert aggregate["precision"] == pytest.approx(2/3)
    assert aggregate["images_with_extra_names"] == 1


def test_confirmed_aliases_accepted_and_actor_name_not_correct(matcher):
    row = score_name_only(sample(),"B",response([{"character":"小欣"},{"character":"李健"}]),matcher)
    assert row["correct"] == 1
    assert row["person_details"][0]["match_kind"] == "alias"
    assert row["person_details"][1]["match_kind"] == "actor"


def test_null_and_missing_remain_in_gt_denominator(matcher):
    row = score_name_only(sample(),"C",response([{"character":None}]),matcher)
    assert row["target_count"] == 2 and row["correct"] == 0 and row["named_predictions"] == 0
    assert [p["outcome"] for p in row["person_details"]] == ["uncertain","missing_name"]
    assert summarize_name_rows([row])["accuracy"] == 0


@pytest.mark.parametrize("name", ["安欣/李响","安欣或李响","安欣、李响",["安欣","李响"],"",0])
def test_multiple_candidate_or_non_string_field_cannot_become_correct(matcher,name):
    row = score_name_only(sample(),"B",response([{"character":name}]),matcher)
    assert row["correct"] == 0 and row["invalid_name_predictions"] == 1


def test_bad_json_and_duplicate_keys_are_not_regex_recovered(matcher):
    for raw in ['安欣和李响', '{"persons":[{"character":"安欣"}],"persons":[{"character":"李响"}]}']:
        row = score_name_only(sample(),"C",raw,matcher)
        assert row["correct"] == 0 and row["extraction_error"]


def test_multiple_canonical_names_without_separators_do_not_get_substring_credit(matcher):
    row = score_name_only(sample(),"B",response([{"character":"安欣李响"}]),matcher)
    assert row["correct"] == 0
    assert row["person_details"][0]["match_kind"] == "ambiguous"


def test_matching_uses_augmenting_path_not_greedy(matcher):
    s = sample()
    s["persons"][0]["character_aliases"] = ["李响"]
    row = score_name_only(s,"C",response([{"character":"李响"},{"character":"安欣"}]),matcher)
    assert row["correct"] == 2


def test_name_scoring_does_not_mutate_sample(matcher):
    s = sample(); original=deepcopy(s)
    score_name_only(s,"C",response([{"character":"安欣"}]),matcher)
    assert s == original


def test_correction_fingerprint_and_previous_identity_checked_and_copied():
    s=sample(("范建",)); before=deepcopy(s)
    correction={"sample_id":"s1","person_id":1,"annotation_id":"p1","image_sha256":"sha",
                "old_character":"范建","new_character":"邓子越","new_character_aliases":[]}
    result=apply_corrections({"s1":s},[correction])
    assert s == before and result["s1"]["persons"][0]["character"] == "邓子越"
    assert result["s1"]["persons"][0]["bbox"] == s["persons"][0]["bbox"]
    with pytest.raises(ValueError):apply_corrections({"s1":s},[{**correction,"image_sha256":"other"}])
    with pytest.raises(ValueError):apply_corrections({"s1":s},[{**correction,"old_character":"别的人"}])
