"""User-reviewed name coverage never grants model-specific correctness."""
from copy import deepcopy
import itertools
import json

import pytest

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.identity_review import score_name_only, summarize_name_rows

NAMES = ['安欣', '李响', '陈书婷', '孟钰']


def fixture(names=NAMES):
    sample = {'sample_id':'s','work_slug':'work','persons':[
        {'person_id':i+1,'character':name,'identity_status':'verified','review_status':'approved',
         'character_aliases':[],'bbox':[0,0,1,1],'annotation_id':str(i+1)} for i,name in enumerate(names)]}
    matcher = CharacterMatcher({'work':Roster('work','测试',[CharacterEntry(n) for n in NAMES])})
    return sample, matcher


def score(sample, matcher, names, exp='C'):
    return score_name_only(sample,exp,json.dumps({'persons':[{'character':n} for n in names]},ensure_ascii=False),matcher)


@pytest.mark.parametrize('answer,optional_correct,full_correct,nulls', [
    (['安欣'],0,1,0), (['安欣','李响'],1,2,0), (['安欣',None],0,1,1),
])
def test_known_optional_identity_may_be_named_or_omitted_without_penalty(answer,optional_correct,full_correct,nulls):
    sample, matcher = fixture(NAMES[:2])
    sample['evaluation_policy'] = {'optional_person_ids':[2]}
    row = score(sample,matcher,answer)
    assert row['target_count']==row['correct']==1 and row['image_names_complete']
    assert row['extra_names']==0 and row['optional_correct']==optional_correct
    assert row['full_target_count']==2 and row['full_correct']==full_correct
    assert row['null_predictions']==nulls
    optional = row['person_details'][1]
    assert optional['optional'] and not optional['scored'] and not optional['counts_toward_required']


def test_optional_match_cannot_cover_missing_required_identity():
    sample, matcher = fixture(NAMES[:2]);sample['evaluation_policy']={'optional_person_ids':[2]}
    row=score(sample,matcher,['李响'])
    assert row['correct']==0 and row['target_count']==1 and not row['image_names_complete']
    assert row['optional_correct']==1 and row['extra_names']==0
    assert row['person_details'][0]['outcome']=='missing_name'


def test_required_match_has_priority_even_when_optional_is_first_and_alias_overlaps():
    sample, matcher = fixture(['李响','安欣']);sample['evaluation_policy']={'optional_person_ids':[1]}
    for person in sample['persons']:person['character_aliases']=['共同代号']
    row=score(sample,matcher,['共同代号'])
    assert row['correct']==1 and row['optional_correct']==0 and row['extra_names']==0
    assert row['predictions'][0]['matched_person_id']==2


def test_augmenting_path_can_cover_optional_after_preserving_required_match():
    sample, matcher = fixture(NAMES[:2]);sample['evaluation_policy']={'optional_person_ids':[2]}
    sample['persons'][0]['character_aliases']=['李响']
    row=score(sample,matcher,['李响','安欣'])
    assert row['correct']==1 and row['optional_correct']==1 and row['full_correct']==2
    assert row['extra_names']==0


@pytest.mark.parametrize('answer', [['安欣','李响','路人甲'],['安欣','李响','李响']])
def test_wrong_or_duplicate_extra_names_are_not_blanket_ignored(answer):
    sample, matcher = fixture(NAMES[:2]);sample['evaluation_policy']={'optional_person_ids':[2]}
    row=score(sample,matcher,answer)
    assert row['correct']==1 and row['optional_correct']==1 and row['extra_names']==1


@pytest.mark.parametrize('answer',list(itertools.combinations(NAMES,3)))
def test_any_three_of_four_is_complete_but_full_four_person_coverage_is_separate(answer):
    sample, matcher = fixture();sample['evaluation_policy']={'required_name_count':3}
    before=deepcopy(sample);row=score(sample,matcher,answer)
    assert sample==before
    assert row['target_count']==row['correct']==3 and row['image_names_complete']
    assert row['full_target_count']==4 and row['full_correct']==3 and not row['full_names_complete']
    assert row['extra_names']==0 and len(row['person_details'])==4
    assert sum(p['scored'] for p in row['person_details'])==3
    assert next(p for p in row['person_details'] if not p['scored'])['outcome']=='not_required'


def test_all_four_correct_names_are_accepted_without_precision_or_extra_penalty():
    sample, matcher=fixture();sample['evaluation_policy']={'required_name_count':3}
    row=score(sample,matcher,NAMES);summary=summarize_name_rows([row])
    assert row['correct']==3 and row['full_correct']==4 and row['extra_names']==0
    assert row['accepted_named_predictions']==4 and summary['precision']==1
    assert summary['target_count']==3 and summary['full_target_count']==4


def test_three_name_quota_does_not_forgive_missing_names_or_incorrect_surplus():
    sample, matcher=fixture();sample['evaluation_policy']={'required_name_count':3}
    missed=score(sample,matcher,NAMES[:2]);assert missed['correct']==2 and not missed['image_names_complete']
    wrong=score(sample,matcher,[*NAMES[:3],'路人甲'])
    assert wrong['image_names_complete'] and wrong['extra_names']==1


def test_quota_is_explicit_per_sample_and_not_generalized_to_other_images():
    sample, matcher=fixture();row=score(sample,matcher,NAMES[:3])
    assert row['target_count']==4 and row['correct']==3 and not row['image_names_complete']


@pytest.mark.parametrize('experiment',['A','B','C'])
def test_policy_is_independent_of_model_identity_and_experiment_format(experiment):
    sample, matcher=fixture(NAMES[:2]);sample['evaluation_policy']={'optional_person_ids':[2]}
    row=score(sample,matcher,NAMES[:2],experiment)
    assert row['correct']==1 and row['full_correct']==2 and row['extra_names']==0


@pytest.mark.parametrize('policy',[
    None, {'optional_person_ids':[99]}, {'optional_person_ids':[1,1]},
    {'optional_person_ids':[True]}, {'required_name_count':True},
    {'required_name_count':0}, {'required_name_count':5}, {'model_correct_override':True},
])
def test_invalid_policy_cannot_silently_drop_targets(policy):
    sample, matcher=fixture();sample['evaluation_policy']=policy
    with pytest.raises(ValueError,match='evaluation policy'):score(sample,matcher,NAMES)
