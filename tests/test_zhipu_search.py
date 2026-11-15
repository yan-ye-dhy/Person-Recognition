"""GLM native and original-vendor function search, with all HTTP calls metered."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import httpx
import pytest

from vcrb.models import ProviderSpec, load_registry
from vcrb.providers.base import GenerationParams
from vcrb.providers.zhipu_search import ZhipuSearchProvider


class Response(SimpleNamespace):
    def __init__(self, body):
        self.body = body
        self.model = body.get('model')
        self.usage = body.get('usage')
        self.choices = [SimpleNamespace(finish_reason=c.get('finish_reason'),
                         message=SimpleNamespace(**c.get('message', {}))) for c in body.get('choices', [])]
    def model_dump(self, **kwargs):
        return deepcopy(self.body)


def answer(*, calls=None, content='{"persons":[]}', sources=None):
    msg = {'role': 'assistant', 'content': content}
    if calls is not None:
        msg['tool_calls'] = calls
    body = {'model': 'glm-served', 'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25},
            'choices': [{'message': msg, 'finish_reason': 'tool_calls' if calls else 'stop'}]}
    if sources is not None:
        body['web_search'] = sources
    return Response(body)


def tool(number=1, *, args=None, name='web_search'):
    return {'id': f'call-{number}', 'type': 'function', 'function': {'name': name,
            'arguments': args if args is not None else json.dumps({'query': f'作品 角色 {number}'})}}


def make_model(*, native=False, **params):
    return replace(load_registry().get('glm-4.6v'), params={
        'api_mode': 'zhipu_native_search' if native else 'zhipu_function_search',
        'extra_body': {'thinking': {'type': 'disabled'}}, **params})


def provider(monkeypatch, completions, search_handler=None, *, attempts=1):
    received = []
    responses = iter(completions)
    def create(**kwargs):
        received.append(deepcopy(kwargs))
        return next(responses)
    monkeypatch.setattr('vcrb.providers.zhipu_search.require_api_key', lambda _: 'sk-unit-secret0123456789')
    kwargs = {}
    if search_handler is not None:
        kwargs['search_http_client_factory'] = lambda: httpx.Client(transport=httpx.MockTransport(search_handler))
    p = ZhipuSearchProvider(ProviderSpec('zhipu', 'https://open.bigmodel.cn/api/paas/v4', 'TEST'),
        client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        settings={'retry': {'max_attempts': attempts}}, sleeper=lambda _: None, **kwargs)
    return p, received


def call(p, model=None, observer=None):
    return p.recognize(model or make_model(), 'data:image/jpeg;base64,AAAA', '原始系统提示', '作品：示例作品',
                       GenerationParams(.1, 4096), attempt_observer=observer)


class Observer:
    def __init__(self): self.events = []
    def before_attempt(self, *args): self.events.append('before_model'); return 'model'
    def after_attempt(self, receipt, model, *, result, error):
        self.events.append('after_model')
        if result is not None:
            # All function-loop model rounds have zero in-model search fees.
            assert result.search_metadata['billing'] == {'units': {'search': 0}, 'known_units': True}
    def before_search(self, *args): self.events.append('before_search'); return 'search'
    def after_search(self, receipt, model, *, units, error):
        self.events.append('after_search'); assert units == 1


def test_function_rounds_preserve_original_image_prompts_and_meter_model_and_search_separately(monkeypatch):
    searches = []
    def search(request):
        searches.append(request)
        return httpx.Response(200, json={'search_result': [
            {'link': 'https://example.test/role', 'title': '角色资料'},
            {'link': 'https://example.test/role', 'title': '重复资料'},
            {'link': 'https://[broken', 'title': '无效地址'}]})
    p, received = provider(monkeypatch, [answer(calls=[tool()], content=None), answer()], search)
    obs = Observer(); result = call(p, observer=obs)
    assert result.ok and result.attempts == 2
    assert result.usage == {'prompt_tokens': 40, 'completion_tokens': 10, 'total_tokens': 50}
    assert result.search_metadata['sources'] == [{'url': 'https://example.test/role', 'title': '角色资料'}]
    assert result.search_metadata['external_search_requests'] == 1
    assert len(result.search_metadata['search_records']) == 1
    assert result.search_metadata['retrieved'] is True
    assert obs.events == ['before_model', 'after_model', 'before_search', 'after_search', 'before_model', 'after_model']
    assert received[0]['messages'][0] == {'role': 'system', 'content': '原始系统提示'}
    assert received[1]['messages'][:2] == received[0]['messages']
    assert received[0]['messages'][1]['content'][0]['image_url']['url'] == 'data:image/jpeg;base64,AAAA'
    assert received[1]['messages'][2]['tool_calls'][0]['id'] == 'call-1'
    assert received[1]['messages'][3]['role'] == 'tool'
    assert received[1]['messages'][3]['tool_call_id'] == 'call-1'
    body = json.loads(searches[0].content)
    assert body['search_query'] == '作品 角色 1' and body['search_engine'] == 'search_pro'
    assert str(searches[0].url) == 'https://open.bigmodel.cn/api/paas/v4/web_search'


@pytest.mark.parametrize('args', ['[]', 'null', '{bad', '{"query":""}', '{"query":7}'])
def test_invalid_search_arguments_return_tool_error_without_physical_search(args, monkeypatch):
    p, received = provider(monkeypatch, [answer(calls=[tool(args=args)]), answer()])
    monkeypatch.setattr(p, '_search', lambda *args: pytest.fail('Invalid arguments must not reach search'))
    result = call(p)
    assert result.ok and result.search_metadata['external_search_requests'] == 0
    assert result.search_metadata['retrieved'] is False
    assert 'error' in json.loads(received[1]['messages'][3]['content'])


def test_multiple_tool_calls_each_get_response_but_only_eight_physical_search_attempts_including_retries(monkeypatch):
    physical = []
    def search(request):
        physical.append(request)
        if len(physical) % 2:
            raise httpx.ConnectError('temporary')
        return httpx.Response(200, json={'search_result': [{'link': 'https://example.test/role'}]})
    p, received = provider(monkeypatch, [answer(calls=[tool(i) for i in range(1, 10)]), answer()], search, attempts=2)
    obs = Observer(); result = call(p, observer=obs)
    assert result.ok and len(physical) == result.search_metadata['external_search_requests'] == 8
    assert len(result.search_metadata['search_records']) == 8
    assert len(result.search_metadata['queries']) == 4
    assert len([m for m in received[1]['messages'] if m['role'] == 'tool']) == 9
    assert received[1]['tools'] == []
    assert obs.events.count('before_search') == obs.events.count('after_search') == 8


def test_final_model_round_disables_tools_even_if_no_search_has_run(monkeypatch):
    p, received = provider(monkeypatch, [answer(calls=[tool(args='[]')]), answer()])
    result = call(p, make_model(max_model_rounds=2))
    assert result.ok and received[-1]['tools'] == []
    assert received[-1]['messages'][-1]['content'].endswith('不再调用工具。')


def test_model_ignoring_final_tools_off_is_reported_as_loop_limit_without_more_http(monkeypatch):
    p, received = provider(monkeypatch, [answer(calls=[tool()])])
    monkeypatch.setattr(p, '_search', lambda *args: pytest.fail('Tools are disabled on final round'))
    result = call(p, make_model(max_model_rounds=1))
    assert not result.ok and result.error_type == 'tool_loop_limit' and len(received) == 1
    assert received[0]['tools'] == []


@pytest.mark.parametrize('value', [True, 0, -1, 9, '8'])
def test_invalid_physical_cap_rejected_before_model_request(value, monkeypatch):
    p, received = provider(monkeypatch, [])
    with pytest.raises(ValueError, match='including retries'):
        call(p, make_model(max_client_search_calls=value))
    assert received == []


def test_no_results_is_successful_search_but_not_retrieval(monkeypatch):
    p, _ = provider(monkeypatch, [answer(calls=[tool()]), answer()],
                     lambda _: httpx.Response(200, json={'search_result': []}))
    result = call(p)
    assert result.ok and result.search_metadata['attempted'] is True
    assert result.search_metadata['retrieved'] is False and result.search_metadata['sources'] == []
    assert result.search_metadata['search_records'][0]['response'] == {'search_result': []}


def test_search_http200_error_body_is_failure_preserved_in_audit_and_not_final_model_success(monkeypatch):
    p, received = provider(monkeypatch, [answer(calls=[tool()])],
        lambda _: httpx.Response(200, json={'error': {'message': 'failed sk-unit-secret0123456789'}}))
    obs = Observer(); result = call(p, observer=obs)
    assert not result.ok and len(received) == 1 and result.search_metadata['external_search_requests'] == 1
    assert result.search_metadata['search_records'][0]['response']['error']['message'].endswith('[api-key]')
    assert obs.events[-1] == 'after_search'


def test_native_search_uses_native_tools_and_retains_source_evidence(monkeypatch):
    p, received = provider(monkeypatch, [answer(sources=[{'link': 'https://example.test/native', 'title': '资料'}])])
    result = call(p, make_model(native=True))
    assert result.ok and result.search_metadata['retrieved'] is True
    assert received[0]['tools'][0] == {'type': 'web_search', 'web_search': {
        'enable': True, 'search_engine': 'search_pro', 'search_result': True}}
    assert 'stream' not in received[0] and received[0]['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert result.search_metadata['billing']['known_units'] is False


def test_native_http200_error_keeps_usage_and_classifies_without_choices(monkeypatch):
    response = Response({'error': {'code': '1113', 'message': '欠费 sk-unit-secret0123456789'},
                         'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}})
    p, received = provider(monkeypatch, [response], attempts=3)
    result = call(p, make_model(native=True))
    assert not result.ok and result.error_type == 'insufficient_balance' and result.http_status == 200
    assert result.usage['prompt_tokens'] == 20 and len(received) == 1
    assert 'sk-unit-secret' not in result.error_message


@pytest.mark.parametrize('where', ['before_search', 'after_search'])
def test_budget_pause_base_exception_propagates_and_context_is_cleaned(where, monkeypatch):
    class BudgetPause(KeyboardInterrupt): pass
    class PauseObserver(Observer):
        def before_search(self, *args):
            if where == 'before_search': raise BudgetPause()
            return super().before_search(*args)
        def after_search(self, *args, **kwargs):
            if where == 'after_search': raise BudgetPause()
            return super().after_search(*args, **kwargs)
    physical = []
    def search(request):
        physical.append(request); return httpx.Response(200, json={'search_result': []})
    p, _ = provider(monkeypatch, [answer(calls=[tool()]), answer()], search)
    with pytest.raises(BudgetPause): call(p, observer=PauseObserver())
    assert len(physical) == (0 if where == 'before_search' else 1)
    # Same provider can serve another image without inheriting tool history.
    result = call(p)
    assert result.ok and len(result.search_metadata['rounds']) == 1
    assert result.search_metadata['external_search_requests'] == 0


@pytest.mark.parametrize('failure,kind', [(httpx.ReadTimeout('slow'), 'timeout'), (httpx.ConnectError('network'), 'network')])
def test_exhausted_search_retry_keeps_physical_bound_and_error_type(failure, kind, monkeypatch):
    physical = []
    def search(request): physical.append(request); raise failure
    p, _ = provider(monkeypatch, [answer(calls=[tool()])], search, attempts=4)
    result = call(p, make_model(max_client_search_calls=2))
    assert not result.ok and result.error_type == kind and len(physical) == 2
    assert result.search_metadata['external_search_requests'] == 2


@pytest.mark.parametrize('status,body,headers,count,kind', [
    (429, {'error': {'code': '1302', 'message': 'rate limit'}}, {}, 2, None),
    (429, {'error': {'code': '1302', 'message': 'rate limit'}}, {'Retry-After': '121'}, 1, 'rate_limit'),
    (429, {'error': {'code': '1113', 'message': '账户已欠费'}}, {}, 1, 'insufficient_balance'),
    (403, {'error': {'message': 'denied'}}, {}, 1, 'auth_error'),
])
def test_search_http_retry_policy_respects_status_balance_and_retry_after(status, body, headers, count, kind, monkeypatch):
    physical = []
    def search(request):
        physical.append(request)
        if len(physical) == 1:
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(200, json={'search_result': []})
    p, _ = provider(monkeypatch, [answer(calls=[tool()]), answer()], search, attempts=3)
    result = call(p)
    assert len(physical) == count
    assert result.ok is (kind is None)
    if kind: assert result.error_type == kind


def test_context_isolated_between_parallel_images(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    started = Barrier(2)
    def create(**kwargs):
        messages = kwargs['messages']
        original = messages[1]['content'][1]['text']
        if not any(m['role'] == 'tool' for m in messages):
            started.wait(timeout=5)
            return answer(calls=[tool(args=json.dumps({'query': original}))])
        returned = json.loads(next(m['content'] for m in messages if m['role'] == 'tool'))
        assert returned['search_result'][0]['title'] == original
        assert kwargs['tools'] == []  # one physical request per image
        return answer()
    def search(request):
        query = json.loads(request.content)['search_query']
        return httpx.Response(200, json={'search_result': [{'link': 'https://example.test/' + query, 'title': query}]})
    monkeypatch.setattr('vcrb.providers.zhipu_search.require_api_key', lambda _: 'sk-unit-secret0123456789')
    p = ZhipuSearchProvider(ProviderSpec('zhipu', 'https://open.bigmodel.cn/api/paas/v4', 'TEST'),
        client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        search_http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(search)),
        settings={'retry': {'max_attempts': 1}})
    def run(name):
        return p.recognize(make_model(max_client_search_calls=1), 'data:image/jpeg;base64,AAAA', 'sys', name, GenerationParams())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ['imageA', 'imageB']))
    for name, result in zip(['imageA', 'imageB'], results):
        assert result.ok and result.search_metadata['external_search_requests'] == 1
        assert result.search_metadata['queries'] == [name]
        assert result.search_metadata['sources'][0]['title'] == name
