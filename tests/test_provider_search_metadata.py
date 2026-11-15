"""Search evidence survives SDK extensions and the full observed HTTP stream."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vcrb.config import load_settings
from vcrb.models import load_registry
from vcrb.providers import GenerationParams, OpenAICompatProvider, ProviderResult
from vcrb.providers.openai_compat import consume_chat_stream, extract_search_metadata


def response(*, usage=None, extras=None, message_extras=None):
    message=NS(content='{"persons":[]}',reasoning_content=None,model_extra=message_extras or {})
    return NS(choices=[NS(message=message,finish_reason='stop')],model='qwen-served',
              usage=usage,model_extra=extras or {})


class DumpUsage:
    def __init__(self, values):
        self.values=values

    def model_dump(self,mode='python'):
        return self.values


def test_absent_search_evidence_stays_empty_without_assuming_execution():
    result=OpenAICompatProvider._parse_response(response(usage=NS(prompt_tokens=10,completion_tokens=5,total_tokens=15)),1,25)
    assert result.search_metadata=={}
    assert result.usage=={'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}
    assert ProviderResult(ok=True).search_metadata=={}


def test_full_sdk_usage_and_plugin_search_count_are_preserved():
    values={'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,
            'prompt_tokens_details':{'cached_tokens':4},'plugins':{'search':{'count':2}}}
    result=OpenAICompatProvider._parse_response(response(usage=DumpUsage(values)),1,25)
    assert result.usage==values
    assert result.search_metadata['usage']=={'plugins':{'search':{'count':2}}}
    assert 'completed' not in result.search_metadata


def test_search_intent_usage_is_explicitly_not_completion_proof():
    result=OpenAICompatProvider._parse_response(response(usage=DumpUsage({'x_tools':{'search':{'count':1}}})),1,25)
    assert result.search_metadata['usage']['x_tools']['search']['count']==1
    assert 'intent_not_completed' in result.search_metadata['count_semantics']['x_tools.search']


def test_top_and_message_sdk_extensions_retain_sources_and_annotations():
    resp=response(extras={'search_info':{'search_results':[{'title':'电影角色','url':'https://example.test/role'}]}},
                  message_extras={'annotations':[{'type':'url_citation','url_citation':{'url':'https://example.test/role'}}],
                                  'tool_calls':[{'type':'web_search','status':'completed'}]})
    evidence=extract_search_metadata(resp)
    assert evidence['response']['search_info']['search_results'][0]['title']=='电影角色'
    assert evidence['messages'][0]['annotations'][0]['type']=='url_citation'
    assert evidence['messages'][0]['tool_calls'][0]['status']=='completed'


def test_selected_evidence_does_not_copy_secrets_headers_or_image_payloads():
    evidence=extract_search_metadata(response(
        extras={'search_info':{'headers':{'Authorization':'Bearer secret'},'api_key':'secret-key',
                               'url':'https://example.test','text':'key sk-test-only-0123456789',
                               'arguments':'{"image":"data:image/jpeg;base64,AAAA"}'},
                'headers':{'Authorization':'Bearer omitted'}}))
    info=evidence['response']['search_info']
    assert 'headers' not in info and 'api_key' not in info
    assert info['url']=='https://example.test'
    assert 'sk-test' not in info['text']
    assert 'AAAA' not in info['arguments']
    assert 'headers' not in evidence


def chunk(text=None,*,reasoning=None,finish=None,usage=None,extra=None):
    choices=[] if text is None and reasoning is None and finish is None else [NS(
        index=0,delta=NS(content=text,reasoning_content=reasoning),finish_reason=finish)]
    return NS(model='qwen-stream-served',choices=choices,usage=usage,model_extra=extra or {})


class FakeStream:
    def __init__(self,events,close_error=None):
        self.events=events;self.closed=False;self.close_error=close_error

    def __iter__(self):
        for event in self.events:
            if isinstance(event,BaseException):
                raise event
            yield event

    def close(self):
        self.closed=True
        if self.close_error:
            raise self.close_error


def test_stream_combines_text_reasoning_usage_and_search_sources():
    stream=FakeStream([
        chunk(reasoning='分析'),chunk('{"persons":',extra={'search_info':{'search_results':[{'url':'https://example.test'}]}}),
        chunk('[]}',finish='stop'),
        chunk(usage=DumpUsage({'prompt_tokens':9,'completion_tokens':6,'total_tokens':15,'x_tools':{'search':{'count':1}}})),
    ])
    result=OpenAICompatProvider._parse_response(consume_chat_stream(stream),1,100)
    assert stream.closed
    assert result.raw_text=='{"persons":[]}' and result.reasoning_text=='分析'
    assert result.served_model=='qwen-stream-served' and result.finish_reason=='stop'
    assert result.usage['total_tokens']==15
    assert result.search_metadata['usage']['x_tools']['search']['count']==1
    assert result.search_metadata['stream_events'][0]['response']['search_info']['search_results'][0]['url']=='https://example.test'


def test_stream_always_closes_and_does_not_mask_original_error():
    stream=FakeStream([chunk('partial'),TimeoutError('stream timed out')],close_error=RuntimeError('cleanup failure'))
    with pytest.raises(TimeoutError,match='stream timed out'):
        consume_chat_stream(stream)
    assert stream.closed


def test_successful_stream_close_failure_does_not_trigger_a_paid_retry():
    stream=FakeStream([chunk('{"persons":[]}',finish='stop')],close_error=RuntimeError('cleanup failure'))
    result=OpenAICompatProvider._parse_response(consume_chat_stream(stream),1,100)
    assert result.ok and result.raw_text=='{"persons":[]}'


class Completions:
    def __init__(self,responses):
        self.responses=list(responses);self.calls=[]

    def create(self,**kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class Observer:
    def __init__(self,streams):
        self.streams=streams;self.events=[]

    def before_attempt(self,*args):
        self.events.append(('before',len(self.events)))
        return len(self.events)

    def after_attempt(self,receipt,model,*,result,error):
        attempt=sum(e[0]=='before' for e in self.events)
        assert self.streams[attempt-1].closed
        self.events.append(('after',result,error))


def test_search_streaming_retry_is_within_each_observed_http_attempt():
    registry=load_registry()
    model=replace(registry.get('qwen3.8-max'),params={'extra_body':{'enable_search':True,'enable_thinking':False}})
    streams=[FakeStream([chunk('partial'),TimeoutError('stream timeout')]),
             FakeStream([chunk('{"persons":[]}',finish='stop'),chunk(usage=DumpUsage({'total_tokens':3}))])]
    completions=Completions(streams)
    settings=load_settings();settings['retry']['max_attempts']=2
    provider=OpenAICompatProvider(registry.providers['dashscope'],settings=settings,
                                  client_factory=lambda:NS(chat=NS(completions=completions)),sleeper=lambda _:None)
    observer=Observer(streams)
    result=provider.recognize(model,'data:image/jpeg;base64,AAAA','system','user',GenerationParams(),attempt_observer=observer)
    assert result.ok and result.attempts==2 and result.raw_text=='{"persons":[]}'
    assert len(completions.calls)==2 and [e[0] for e in observer.events]==['before','after','before','after']
    assert isinstance(observer.events[1][2],TimeoutError)
    assert completions.calls[0]['stream'] is True
    assert completions.calls[0]['stream_options']=={'include_usage':True}
    assert result.search_metadata=={}


def test_explicit_stream_option_and_nonstream_baseline_are_separate():
    registry=load_registry();provider=OpenAICompatProvider(registry.providers['dashscope'])
    model=replace(registry.get('qwen3.8-max'),params={})
    baseline=provider.build_request_kwargs(model,[],GenerationParams())
    assert 'stream' not in baseline and 'stream_options' not in baseline
    streaming=provider.build_request_kwargs(replace(model,params={'stream':True}),[],GenerationParams())
    assert streaming['stream'] is True and streaming['stream_options']['include_usage'] is True


def test_forced_search_and_intent_telemetry_parameters_are_transmitted_verbatim():
    registry=load_registry();provider=OpenAICompatProvider(registry.providers['dashscope'])
    extra={'enable_thinking':False,'enable_search':True,'include_tool_usage':True,
           'search_options':{'forced_search':True,'search_strategy':'turbo'}}
    model=replace(registry.get('qwen3.8-max'),params={'extra_body':extra})
    request=provider.build_request_kwargs(model,[],GenerationParams())
    assert request['extra_body']==extra
    assert request['stream'] is True and request['stream_options']=={'include_usage':True}
    assert 'tools' not in request and 'tool_choice' not in request
