"""Original DashScope multimodal SSE search, no real API requests."""
import base64
from dataclasses import replace
import json

import httpx
import pytest

from tests.conftest import make_image_bytes
from vcrb.models import ProviderSpec, load_registry
from vcrb.providers.base import GenerationParams
from vcrb.providers.dashscope_multimodal_search import (
    DashScopeMultimodalSearchProvider, completion_from_multimodal_events, normalize_multimodal_search,
)


def model():
    return replace(load_registry().get('qwen3.7-plus'), params={'api_mode': 'dashscope_multimodal_search',
                   'extra_body': {'enable_thinking': False, 'enable_search': True}})


def frames():
    return [
        {'search_info': {'queries': ['作品 角色'], 'search_results': [{'url': 'https://example.test/role', 'title': '角色资料'}]},
         'usage': {'input_tokens': 30, 'output_tokens': 0}},
        {'output': {'choices': [{'message': {'role': 'assistant', 'content': [{'text': '{"persons":'}]},
                                 'finish_reason': 'null'}]}, 'usage': {'input_tokens': 30, 'output_tokens': 3}},
        {'output': {'choices': [{'message': {'role': 'assistant', 'content': [{'text': '[]}'}, {'text': '\n'}]},
                                 'finish_reason': 'stop'}]}, 'usage': {'input_tokens': 30, 'output_tokens': 7}},
    ]


def encoded_stream(events):
    return ''.join('id:1\nevent:result\n:HTTP_STATUS/200\ndata:' + json.dumps(x, ensure_ascii=False) + '\n\n' for x in events).encode()


def provider(monkeypatch, handler, attempts=1):
    monkeypatch.setattr('vcrb.providers.dashscope_multimodal_search.require_api_key', lambda _: 'sk-unit-secret0123456789')
    return DashScopeMultimodalSearchProvider(ProviderSpec('dashscope', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'TEST'),
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        settings={'retry': {'max_attempts': attempts}}, sleeper=lambda _: None)


def call(p, observer=None):
    image = make_image_bytes((64, 40), (80, 50, 90))
    image_url = 'data:image/jpeg;base64,' + base64.b64encode(image).decode()
    return p.recognize(model(), image_url, '原始系统提示', '作品：示例作品', GenerationParams(.1, 4096),
                       attempt_observer=observer), image, image_url


def test_wire_preserves_exact_jpeg_and_prompts_search_forced_thinking_off_and_consumes_one_http(monkeypatch):
    received = []
    def handler(request):
        received.append(request)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=encoded_stream(frames()))
    p = provider(monkeypatch, handler)
    result, image, image_url = call(p)
    body = json.loads(received[0].content)
    assert str(received[0].url) == 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
    assert received[0].headers['X-DashScope-SSE'] == 'enable'
    assert body['input']['messages'] == [
        {'role': 'system', 'content': [{'text': '原始系统提示'}]},
        {'role': 'user', 'content': [{'image': image_url}, {'text': '作品：示例作品'}]}]
    assert base64.b64decode(body['input']['messages'][1]['content'][0]['image'].split(',')[1]) == image
    assert body['parameters'] == {'enable_search': True, 'enable_thinking': False, 'incremental_output': True,
        'result_format': 'message', 'max_tokens': 4096, 'temperature': .1,
        'search_options': {'forced_search': True, 'search_strategy': 'max', 'enable_source': True}}
    assert len(received) == 1 and result.ok and result.raw_text == '{"persons":[]}\n'
    assert result.usage['prompt_tokens'] == 30 and result.usage['completion_tokens'] == 7
    assert result.usage['total_tokens'] == 37  # cumulative output snapshots, not 0+3+7
    assert result.search_metadata['retrieved'] is True
    assert result.search_metadata['sources'] == [{'url': 'https://example.test/role', 'title': '角色资料'}]
    assert result.search_metadata['queries'] == ['作品 角色']
    assert result.search_metadata['billing'] == {'units': {}, 'known_units': False}
    assert p.transport_profile['stream'] is True
    p.close()


def test_sources_from_root_and_output_deduplicate_and_invalid_urls_do_not_count():
    metadata = normalize_multimodal_search({'events': [
        {'search_info': {'search_results': [{'url': 'https://example.test/role'}]}},
        {'output': {'search_info': {'search_results': [{'url': 'https://example.test/role'}, {'url': 'https://[bad'},
                                                    {'url': 'file:///secret'}, {'url': 'https://example.test/role2'}]}}}]
    })
    assert len(metadata['sources']) == 2 and metadata['retrieved'] is True


def test_intent_counter_without_sources_is_never_retrieval_or_known_billing():
    metadata = normalize_multimodal_search({'events': [{'usage': {'x_tools': {'search': {'count': 1}}},
        'search_info': {'search_results': []}}]})
    assert metadata['retrieved'] is False and metadata['sources'] == []
    assert metadata['billing']['known_units'] is False


def test_partial_usage_snapshots_merge_without_summing_and_reasoning_not_in_final():
    data = frames()
    data[1]['output']['choices'][0]['message']['reasoning_content'] = '审计思考'
    data[2]['usage'] = {'output_tokens': 7}
    response = completion_from_multimodal_events(data)
    assert response.usage['prompt_tokens'] == 30 and response.usage['completion_tokens'] == 7
    assert response.choices[0].message.content == '{"persons":[]}\n'
    assert response.choices[0].message.reasoning_content == '审计思考'


def test_sse_http200_error_keeps_usage_sources_and_no_unobserved_retry(monkeypatch):
    received, observed = [], []
    def handler(request):
        received.append(request)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=encoded_stream(
            frames()[:1] + [{'code': 'InternalError', 'message': 'failed sk-unit-secret0123456789',
                            'usage': {'input_tokens': 30, 'output_tokens': 2}}]))
    class Observer:
        def before_attempt(self, *args): return 1
        def after_attempt(self, receipt, model, *, result, error):
            observed.append(result)
            assert result.usage['prompt_tokens'] == 30 and error is None
    p = provider(monkeypatch, handler, attempts=3)
    result, _, _ = call(p, Observer())
    assert not result.ok and result.error_type == 'server_error' and len(received) == len(observed) == 1
    assert result.search_metadata['retrieved'] is True
    assert 'sk-unit-secret' not in result.error_message and 'sk-unit-secret' not in json.dumps(result.search_metadata)


def test_sse_without_final_marker_is_incomplete_not_success(monkeypatch):
    p = provider(monkeypatch, lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'},
                 content=encoded_stream(frames()[:2])))
    result, _, _ = call(p)
    assert not result.ok and result.error_type == 'server_error'
    assert result.raw_text == '{"persons":'


class BrokenStream(httpx.SyncByteStream):
    def __iter__(self):
        yield encoded_stream(frames()[:1])
        raise httpx.ReadTimeout('stream interrupted')
    def close(self): pass


def test_midstream_failure_is_one_observed_attempt_then_metered_retry(monkeypatch):
    received, events = [], []
    def handler(request):
        received.append(request)
        return (httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=BrokenStream())
                if len(received) == 1 else httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=encoded_stream(frames())))
    class Observer:
        def before_attempt(self, *args): events.append('before'); return len(received) + 1
        def after_attempt(self, receipt, model, *, result, error):
            events.append('after')
            if receipt == 1:
                assert isinstance(error, TimeoutError)
            else:
                assert result.ok
    p = provider(monkeypatch, handler, attempts=2)
    result, _, _ = call(p, Observer())
    assert result.ok and result.attempts == len(received) == 2
    assert events == ['before', 'after', 'before', 'after']


@pytest.mark.parametrize('status,kind', [(400, 'bad_request'), (401, 'auth_error')])
def test_http_errors_are_read_from_stream_and_classified_without_retry(status, kind, monkeypatch):
    received = []
    def handler(request):
        received.append(request); return httpx.Response(status, json={'code': 'InvalidParameter', 'message': 'failed'})
    p = provider(monkeypatch, handler, attempts=3)
    result, _, _ = call(p)
    assert not result.ok and result.error_type == kind and len(received) == 1


def test_non_sse_json_error_response_is_preserved_not_mistaken_for_empty_stream(monkeypatch):
    p = provider(monkeypatch, lambda _: httpx.Response(200, json={'code': 'BadRequest', 'message': 'bad',
        'usage': {'input_tokens': 10, 'output_tokens': 0}}))
    result, _, _ = call(p)
    assert not result.ok and result.usage['prompt_tokens'] == 10
