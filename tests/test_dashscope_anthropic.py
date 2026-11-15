"""Offline HTTP transport checks for capped server-side search and real evidence."""
import base64
from dataclasses import replace
import json

import httpx
import pytest

from tests.conftest import make_image_bytes
from vcrb.models import ProviderSpec, load_registry
from vcrb.providers import GenerationParams
from vcrb.providers.dashscope_anthropic import (
    CLIENT_MARKER, DashScopeAnthropicProvider, anthropic_endpoint, build_native_payload,
    completion_from_native,
)


def native_response(*,content=None,status='end_turn',usage=None):
    return {'id':'message1','type':'message','model':'qwen3.8-max-served','stop_reason':status,
            'content':content if content is not None else [{'type':'text','text':'{"persons":[]}'}],
            'usage':usage if usage is not None else {'input_tokens':10,'output_tokens':5}}


def search_model(**params):
    return replace(load_registry().get('qwen3.8-max'),params={
        'api_mode':'dashscope_anthropic','search_max_uses':1,
        'extra_body':{'enable_search':True,'enable_thinking':False,'search_options':{'forced_search':True}},
        **params,
    })


def provider(monkeypatch,handler,*,attempts=1):
    monkeypatch.setattr('vcrb.providers.dashscope_anthropic.require_api_key',lambda _: 'sk-unit-test-secret012345')
    return DashScopeAnthropicProvider(
        ProviderSpec('dashscope','https://dashscope.aliyuncs.com/compatible-mode/v1','TEST_KEY'),
        settings={'retry':{'max_attempts':attempts,'timeout_seconds':5}},sleeper=lambda _:None,
        http_client_factory=lambda:httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_fifteen_input_images_remain_byte_identical_and_only_one_post_each(monkeypatch):
    received=[]
    def handler(request):
        received.append((request,json.loads(request.content)))
        return httpx.Response(200,json=native_response())
    p=provider(monkeypatch,handler)
    for i in range(15):
        original=make_image_bytes((64,40),(i*10,40,80))
        data_url='data:image/jpeg;base64,'+base64.b64encode(original).decode()
        result=p.recognize(search_model(),data_url,'只输出JSON对象。','识别画面角色。',GenerationParams(.1,4096))
        request,body=received[-1]
        assert result.ok and result.raw_text=='{"persons":[]}'
        assert str(request.url)=='https://dashscope.aliyuncs.com/apps/anthropic/v1/messages'
        assert request.headers['x-api-key']=='sk-unit-test-secret012345'
        assert request.headers['anthropic-version']=='2023-06-01'
        assert body['system']==[{'type':'text','text':CLIENT_MARKER},{'type':'text','text':'只输出JSON对象。'}]
        assert base64.b64decode(body['messages'][0]['content'][0]['source']['data'])==original
        assert body['messages'][0]['content'][1]=={'type':'text','text':'识别画面角色。'}
        assert body['tools']==[{'type':'web_search_20250305','name':'web_search','max_uses':1}]
        assert body['tool_choice']=={'type':'tool','name':'web_search'}
        assert body['thinking']=={'type':'disabled'} and body['stream'] is False
        assert body['temperature']==.1 and body['max_tokens']==4096
        assert 'extra_body' not in body
    assert len(received)==15
    assert p.transport_profile['client_marker']==CLIENT_MARKER
    p.close()


def test_returned_server_results_sources_and_usage_are_evidence_not_final_answer(monkeypatch):
    content=[
        {'type':'server_tool_use','id':'s1','name':'web_search','input':{'queries':['电影角色资料']}},
        {'type':'web_search_tool_result','tool_use_id':'s1','content':[
            {'type':'web_search_result','title':'角色资料','url':'https://example.test/roles','encrypted_content':'opaque'}]},
        {'type':'thinking','thinking':'备选姓名不属于最终回答'},
        {'type':'text','text':'{"persons":[]}','citations':[{'type':'web_search_result_location','url':'https://example.test/roles'}]},
    ]
    p=provider(monkeypatch,lambda _:httpx.Response(200,json=native_response(
        content=content,usage={'input_tokens':10,'output_tokens':5,'cache_read_input_tokens':3,
                               'server_tool_use':{'web_search_requests':1}})))
    result=p.recognize(search_model(),'data:image/jpeg;base64,AAAA','system','user',GenerationParams())
    assert result.raw_text=='{"persons":[]}' and result.reasoning_text=='备选姓名不属于最终回答'
    assert result.usage['prompt_tokens']==13 and result.usage['completion_tokens']==5 and result.usage['total_tokens']==18
    assert result.usage['server_tool_use']['web_search_requests']==1
    info=result.search_metadata['messages'][0]['search_info']
    assert info['web_search_tool_results'][0]['content'][0]['url']=='https://example.test/roles'
    assert info['server_tool_use']['web_search_requests']==1
    assert 'example.test' not in result.raw_text
    assert result.search_metadata['messages'][0]['annotations'][0]['url']=='https://example.test/roles'


@pytest.mark.parametrize('tool_content',[[],{'type':'web_search_tool_result_error','error_code':'max_uses_exceeded'}])
def test_empty_or_error_tool_result_is_preserved_without_claiming_completed_sources(tool_content):
    response=completion_from_native(native_response(content=[
        {'type':'web_search_tool_result','tool_use_id':'s1','content':tool_content},
        {'type':'text','text':'{"persons":[]}'},
    ],usage={'input_tokens':10,'output_tokens':5,'server_tool_use':{'web_search_requests':1}}))
    result=DashScopeAnthropicProvider._parse_response(response,1,1)
    info=result.search_metadata['messages'][0]['search_info']
    assert info['web_search_tool_results'][0]['content']==tool_content
    assert 'completed' not in info


def test_no_result_blocks_does_not_turn_intent_telemetry_into_proof():
    data=native_response();data['x_tools']={'web_search':{'count':1}}
    result=DashScopeAnthropicProvider._parse_response(completion_from_native(data),1,1)
    info=result.search_metadata['messages'][0]['search_info']
    assert info['x_tools']['web_search']['count']==1
    assert 'web_search_tool_results' not in info and 'server_tool_uses' not in info


@pytest.mark.parametrize('status,kind',[(401,'auth_error'),(400,'bad_request'),(404,'model_not_found')])
def test_native_http_errors_are_classified_and_credentials_are_redacted(monkeypatch,status,kind):
    received=[]
    def handler(request):
        received.append(request)
        return httpx.Response(status,json={'error':{'message':'failed sk-unit-test-secret012345','code':'test'}})
    p=provider(monkeypatch,handler,attempts=3)
    result=p.recognize(search_model(),'data:image/jpeg;base64,AAAA','system','user',GenerationParams())
    assert not result.ok and result.error_type==kind and result.http_status==status
    assert result.attempts==len(received)==1
    assert 'sk-unit-test' not in result.error_message


@pytest.mark.parametrize('failure,kind',[(httpx.ReadTimeout('timeout'),'timeout'),(httpx.ConnectError('connect'),'network')])
def test_native_network_failures_retry_as_separate_observed_http_attempts(monkeypatch,failure,kind):
    received=[];events=[]
    def handler(request):
        received.append(request)
        if len(received)==1:
            raise failure
        return httpx.Response(200,json=native_response())
    class Observer:
        def before_attempt(self,*args):
            events.append('before');return None
        def after_attempt(self,receipt,model,*,result,error):
            events.append('after')
            if len(received)==1:
                from vcrb.providers.errors import classify_exception
                assert classify_exception(error)==(kind,True)
    p=provider(monkeypatch,handler,attempts=2)
    result=p.recognize(search_model(),'data:image/jpeg;base64,AAAA','system','user',GenerationParams(),attempt_observer=Observer())
    assert result.ok and result.attempts==len(received)==2
    assert events==['before','after','before','after']


def test_pause_turn_or_truncation_does_not_issue_unobserved_followup_requests(monkeypatch):
    received=[]
    def handler(request):
        received.append(request)
        return httpx.Response(200,json=native_response(content=[],status='pause_turn'))
    p=provider(monkeypatch,handler,attempts=3)
    result=p.recognize(search_model(),'data:image/jpeg;base64,AAAA','system','user',GenerationParams())
    assert result.ok and result.output_issue=='empty_final_content'
    assert result.finish_reason=='pause_turn' and len(received)==1
    converted=completion_from_native(native_response(status='max_tokens'))
    assert DashScopeAnthropicProvider._parse_response(converted,1,1).output_issue=='truncated_output'


@pytest.mark.parametrize('maximum',[0,-1,True,1.5,'1'])
def test_invalid_search_cap_rejected_before_http(maximum):
    p=DashScopeAnthropicProvider(ProviderSpec('dashscope','https://example.test/compatible-mode/v1','TEST'))
    with pytest.raises(ValueError,match='search_max_uses'):
        p.build_request_kwargs(search_model(search_max_uses=maximum),[],GenerationParams())


def test_endpoint_adaptation_and_native_nonstream_request_shape():
    assert anthropic_endpoint('https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')=='https://workspace.cn-beijing.maas.aliyuncs.com/apps/anthropic/v1/messages'
    p=DashScopeAnthropicProvider(ProviderSpec('dashscope','https://example.test/compatible-mode/v1','TEST'))
    kwargs=p.build_request_kwargs(search_model(),p.build_messages('data:image/jpeg;base64,AAAA','sys','user'),GenerationParams())
    assert 'stream' not in kwargs and 'stream_options' not in kwargs
    assert build_native_payload(kwargs)['tools'][0]['max_uses']==1
    with pytest.raises(ValueError):anthropic_endpoint('http://example.test')
    with pytest.raises(ValueError):anthropic_endpoint('https://name:secret@example.test')
    alias=DashScopeAnthropicProvider(ProviderSpec('dashscope','https://example.test','TEST'),max_search_uses=1)
    assert alias.max_uses==1
