"""Annotation replay verifies historical inputs without rewriting their outputs."""
from copy import deepcopy
import hashlib
import json

import pytest

from vcrb.multi.review_revision import canonical_sample_hash, replay_review_revision, validate_frozen_source
from vcrb.multi.schema import MultiSample, PersonAnnotation, dataset_hash


def person(pid,name):
    return PersonAnnotation(person_id=pid,bbox=[0,0,1,1],character=name,identity_status='verified',
                            review_status='approved',annotation_id=f'person-{pid}',
                            evidence=[{'type':'manual_review','confirmed':True}])


def sample(sid='s1',sha='a'):
    return MultiSample(sample_id=sid,work_slug='work',work_name='测试',work_category='真人剧',
                       image_path=f'images/{sid}.jpg',image_width=100,image_height=100,image_sha256=sha*64,
                       persons=[person(1,'安欣')],review_status='approved',foreground_complete=True,
                       evidence=[{'type':'manual_review','confirmed':True}]).to_dict()


def digest(samples):
    return dataset_hash([MultiSample.from_dict(s) for s in samples.values()])


def reviewed():
    before={'s1':sample(),'s2':sample('s2','b')};after=deepcopy(before)
    after['s1']['persons'].append(person(2,'李响').to_dict())
    after['s1']['evaluation_policy']={'optional_person_ids':[2]}
    manifest={'schema_version':'1.0','revision_id':'explicit-user-review',
              'before_dataset_hash':digest(before),'after_dataset_hash':digest(after),
              'revisions':[{'sample_id':'s1','image_sha256':'a'*64,
                  'before_sample_hash':canonical_sample_hash(before['s1']),
                  'after_sample_hash':canonical_sample_hash(after['s1']),
                  'before':deepcopy(before['s1']),'after':deepcopy(after['s1']),
                  'reason':'Explicit user confirmation of an optional visible identity',
                  'user_evidence':{'confirmed':True}}]}
    return before,after,manifest


def test_empty_policy_preserves_legacy_canonical_serialization_and_hash():
    old=sample();assert 'evaluation_policy' not in old
    expected=hashlib.sha256(json.dumps([old],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    loaded=MultiSample.from_dict(old)
    assert loaded.evaluation_policy=={} and loaded.to_dict()==old
    assert dataset_hash([loaded])==expected
    loaded.evaluation_policy={'required_name_count':1}
    assert loaded.to_dict()['evaluation_policy']=={'required_name_count':1}
    assert dataset_hash([loaded])!=expected


def test_review_round_trip_keeps_before_after_and_source_objects_immutable():
    before,after,manifest=reviewed();saved=(deepcopy(before),deepcopy(after),deepcopy(manifest))
    revised=replay_review_revision(before,manifest)
    assert revised==after and replay_review_revision(revised,manifest,reverse=True)==before
    assert (before,after,manifest)==saved
    revised['s1']['persons'][0]['character']='changed only in returned copy'
    assert before==saved[0] and after==saved[1]


@pytest.mark.parametrize('tamper',[
    'image_sha','image_dimensions','before_hash','after_hash','duplicate','missing_revision','unknown_sample',
    'self_consistent_wrong_before','draft_after','wrong_after_dataset','wrong_before_dataset',
])
def test_bad_fingerprints_partial_replays_or_unapproved_annotations_are_rejected(tamper):
    before,after,manifest=reviewed();entry=manifest['revisions'][0]
    if tamper=='image_sha':entry['image_sha256']='c'*64
    elif tamper=='image_dimensions':
        entry['after']['image_width']=200;entry['after_sample_hash']=canonical_sample_hash(entry['after'])
    elif tamper=='before_hash':entry['before_sample_hash']='wrong'
    elif tamper=='after_hash':entry['after_sample_hash']='wrong'
    elif tamper=='duplicate':manifest['revisions'].append(deepcopy(entry))
    elif tamper=='missing_revision':manifest['revisions']=[]
    elif tamper=='unknown_sample':entry['sample_id']='not-in-dataset'
    elif tamper=='self_consistent_wrong_before':
        entry['before']['persons'][0]['character']='李响'
        entry['before_sample_hash']=canonical_sample_hash(entry['before'])
    elif tamper=='draft_after':
        entry['after']['review_status']='draft'
        entry['after_sample_hash']=canonical_sample_hash(entry['after'])
        changed=deepcopy(after);changed['s1']=entry['after'];manifest['after_dataset_hash']=digest(changed)
    elif tamper=='wrong_after_dataset':manifest['after_dataset_hash']='wrong'
    else:manifest['before_dataset_hash']='wrong'
    saved=deepcopy(before)
    with pytest.raises(ValueError):replay_review_revision(before,manifest)
    assert before==saved


def test_reverse_replay_requires_the_exact_reviewed_source_and_original_dataset_order():
    before,after,manifest=reviewed()
    changed=deepcopy(after);changed['s1']['note']='Unlisted later annotation edit'
    with pytest.raises(ValueError,match='source dataset'):replay_review_revision(changed,manifest,reverse=True)
    with pytest.raises(ValueError,match='source dataset'):
        replay_review_revision(dict(reversed(list(before.items()))),manifest)


def test_source_validation_allows_audited_note_evidence_differences_but_not_scoring_changes():
    before,_after,_manifest=reviewed();source=deepcopy(before)
    source['s1']['note']='Historical evidence wording'
    source['s1']['persons'][0]['evidence']=[{'type':'different_manual_review','confirmed':True}]
    validate_frozen_source(source,before)
    for key,value in [('character','李响'),('bbox',[.1,0,1,1]),('character_aliases',['小欣'])]:
        invalid=deepcopy(source);invalid['s1']['persons'][0][key]=value
        with pytest.raises(ValueError,match='differs'):validate_frozen_source(invalid,before)
    invalid=deepcopy(source);invalid['s1']['evaluation_policy']={'required_name_count':1}
    with pytest.raises(ValueError,match='differs'):validate_frozen_source(invalid,before)


def test_source_must_include_the_entire_review_base_not_only_the_changed_image():
    before,_after,_manifest=reviewed()
    with pytest.raises(ValueError,match='sample set'):validate_frozen_source({'s1':before['s1']},before)


def test_unknown_policy_or_dropped_sample_fields_cannot_escape_fingerprinting():
    s=sample();s['evaluation_policy']={'model_specific_correct':True}
    with pytest.raises(ValueError,match='policy'):canonical_sample_hash(s)
    s=sample();s['evaluation_polciy']={'required_name_count':1}
    with pytest.raises(ValueError,match='Unknown review sample'):canonical_sample_hash(s)
