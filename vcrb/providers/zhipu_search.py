"""Original Zhipu model + native search or a metered same-vendor tool loop."""
from __future__ import annotations

from contextvars import ContextVar
import json
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import ProviderResult
from .errors import classify_exception, error_summary
from .native_search_common import APIConnectionError, NativeSearchHTTPError, https_endpoint
from .openai_compat import OpenAICompatProvider, _dump_mapping, _safe_json, _usage_mapping
from .search_evidence import normalize_search_evidence
from .http_audit import response_identity
from ..config import require_api_key

_conversation: ContextVar[Any] = ContextVar('zhipu_search_conversation', default=None)
_tools_enabled: ContextVar[bool] = ContextVar('zhipu_search_tools_enabled', default=True)
_search_state: ContextVar[Any] = ContextVar('zhipu_search_attempts', default=None)
FUNCTION = {'type': 'function', 'function': {'name': 'web_search',
            'description': '检索网页核实当前影视作品的角色身份，返回来源和摘要。',
            'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                           'required': ['query'], 'additionalProperties': False}}}


class ZhipuSearchProvider(OpenAICompatProvider):
    name = 'zhipu_search'

    def __init__(self, provider_spec, *, search_http_client_factory=None, **kwargs):
        super().__init__(provider_spec, **kwargs)
        # Validate the original provider before any authenticated request.
        https_endpoint(provider_spec.base_url, '')
        self._search_http_client_factory = search_http_client_factory

    def build_messages(self, image_data_url, system_prompt, user_prompt):
        return _conversation.get() or super().build_messages(image_data_url, system_prompt, user_prompt)

    def build_request_kwargs(self, model, messages, gen):
        kwargs = super().build_request_kwargs(model, messages, gen)
        if model.params.get('api_mode') == 'zhipu_native_search':
            kwargs['tools'] = [{'type': 'web_search', 'web_search': {
                'enable': True, 'search_engine': 'search_pro', 'search_result': True}}]
        else:
            kwargs['tools'] = [FUNCTION] if _tools_enabled.get() else []
        # Normal chat completions retains top-level web_search source objects.
        kwargs.pop('stream', None)
        kwargs.pop('stream_options', None)
        return kwargs

    @staticmethod
    def _parse_response(resp, attempts, latency_ms):
        raw = _dump_mapping(resp)
        if raw.get('error'):
            error = raw['error']
            code = str(error.get('code', '')) if isinstance(error, dict) else ''
            kind = ('insufficient_balance' if code == '1113' else
                    'rate_limit' if code in {'1302', '1303', '1305'} else 'server_error')
            result = ProviderResult(ok=False, served_model=raw.get('model'),
                usage=_usage_mapping(raw.get('usage')), attempts=attempts, latency_ms=latency_ms,
                http_status=200, error_type=kind, error_message=json.dumps(_safe_json(error), ensure_ascii=False)[:400])
            result.http_metadata = response_identity(resp)
        else:
            result = OpenAICompatProvider._parse_response(resp, attempts, latency_ms)
        result.search_metadata = normalize_search_evidence('zhipu', raw)
        if _conversation.get() is not None:
            result.search_metadata['billing'] = {'units': {'search': 0}, 'known_units': True}
        if raw.get('choices'):
            result.search_metadata['assistant_message'] = _safe_json(raw['choices'][0].get('message') or {})
        return result

    def recognize(self, model, image_data_url, system_prompt, user_prompt, gen, *, attempt_observer=None):
        if model.params.get('api_mode') == 'zhipu_native_search':
            return super().recognize(model, image_data_url, system_prompt, user_prompt, gen,
                                     attempt_observer=attempt_observer)
        max_search = model.params.get('max_client_search_calls', 8)
        max_rounds = model.params.get('max_model_rounds', 10)
        if type(max_search) is not int or not 1 <= max_search <= 8:
            raise ValueError('max_client_search_calls must be an integer from 1 to 8, including retries')
        if type(max_rounds) is not int or not 1 <= max_rounds <= 10:
            raise ValueError('max_model_rounds must be an integer from 1 to 10')
        state = {'requests': 0, 'maximum': max_search, 'latency_ms': 0, 'records': []}
        token = _search_state.set(state)
        try:
            return self._recognize_function(model, image_data_url, system_prompt, user_prompt, gen,
                                            max_rounds, state, attempt_observer)
        finally:
            _search_state.reset(token)

    def _recognize_function(self, model, image_data_url, system_prompt, user_prompt, gen,
                            max_rounds, state, attempt_observer):
        messages = super().build_messages(image_data_url, system_prompt, user_prompt)
        attempts = 0
        latency = 0
        sources, queries, rounds = [], [], []
        seen_sources = set()
        for round_index in range(max_rounds):
            enabled = state['requests'] < state['maximum'] and round_index < max_rounds - 1
            if not enabled:
                messages.append({'role': 'user', 'content': '检索轮数已到上限。请根据已有信息提交最终JSON，不再调用工具。'})
            token = _conversation.set(messages)
            tools_token = _tools_enabled.set(enabled)
            try:
                result = super().recognize(model, image_data_url, system_prompt, user_prompt, gen,
                                           attempt_observer=attempt_observer)
            finally:
                _conversation.reset(token)
                _tools_enabled.reset(tools_token)
            attempts += result.attempts
            latency += result.latency_ms
            rounds.append({'round': round_index + 1, 'raw_text': result.raw_text,
                           'usage': result.usage, 'search_metadata': result.search_metadata})
            if not result.ok:
                break
            assistant = result.search_metadata.get('assistant_message') or {}
            calls = assistant.get('tool_calls') or []
            if not calls:
                break
            if not isinstance(calls, list) or any(not isinstance(c, dict) or not isinstance(c.get('id'), str)
                                                or not c['id'] for c in calls):
                result.ok, result.error_type = False, 'internal_error'
                result.error_message = 'Malformed assistant tool_calls; no valid continuation possible'
                break
            messages.append(assistant)
            for call in calls:
                function = call.get('function') or {}
                response = {'error': '检索次数已到上限'}
                if not isinstance(function, dict) or function.get('name') != 'web_search':
                    response = {'error': 'Unsupported tool; only web_search is available'}
                elif enabled and state['requests'] < state['maximum']:
                    try:
                        args = json.loads(function.get('arguments') or '{}')
                        if not isinstance(args, dict):
                            raise ValueError('Search arguments must be a JSON object')
                        query = args.get('query')
                        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                            raise ValueError('query must be a nonempty string <=2000 characters')
                    except (ValueError, TypeError) as exc:
                        response = {'error': str(exc)}
                    else:
                        queries.append(query)
                        try:
                            response = self._search(model, query, image_data_url, user_prompt, attempt_observer)
                            if not isinstance(response, dict):
                                raise ValueError('Search response must be a JSON object')
                        except Exception as exc:
                            result.ok = False
                            result.error_type = classify_exception(exc)[0]
                            result.error_message = error_summary(exc)
                            response = {'error': result.error_message}
                        for item in response.get('search_result') or []:
                            if isinstance(item, dict) and isinstance(item.get('link'), str):
                                try:
                                    parsed = urlsplit(item['link'])
                                except ValueError:
                                    continue
                                if parsed.scheme in {'http', 'https'} and parsed.netloc and item['link'] not in seen_sources:
                                    seen_sources.add(item['link'])
                                    sources.append({'url': item['link'], 'title': str(item.get('title') or '')})
                messages.append({'role': 'tool', 'tool_call_id': call.get('id'),
                                 'content': json.dumps(_safe_json(response), ensure_ascii=False)})
            if not result.ok:
                break
        else:
            result.ok = False
            result.error_type = 'tool_loop_limit'
            result.error_message = 'Maximum model rounds reached without a final answer'
        if result.ok and (result.search_metadata.get('assistant_message') or {}).get('tool_calls'):
            result.ok = False
            result.error_type = 'tool_loop_limit'
            result.error_message = 'No final answer after bounded tool loop'
        result.attempts, result.latency_ms = attempts, latency + state['latency_ms']
        # Every model exchange is settled separately before the next round.
        # Task-level totals are derived only when every round has valid usage.
        usage_records = [r['usage'] for r in rounds]
        if usage_records and all(isinstance(u, dict) and all(type(u.get(k)) is int and u[k] >= 0
                for k in ('prompt_tokens', 'completion_tokens'))
                and ('total_tokens' not in u or type(u['total_tokens']) is int
                     and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']) for u in usage_records):
            usage = {'prompt_tokens': sum(u['prompt_tokens'] for u in usage_records),
                     'completion_tokens': sum(u['completion_tokens'] for u in usage_records)}
            usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
            result.usage = usage
        result.search_metadata = {'requested': True, 'attempted': state['requests'] > 0,
            'retrieved': bool(sources), 'sources': sources, 'queries': queries,
            'billing': {'units': {'search': 0}, 'known_units': True,
                        'note': 'Search HTTP calls accounted separately by before_search/after_search'},
            'external_search_requests': state['requests'], 'search_records': state['records'], 'rounds': rounds,
            'api_mode': 'zhipu_function_search'}
        return result

    def _search(self, model, query, image_data_url, user_prompt, observer):
        parsed = urlsplit(self.spec.base_url)
        endpoint = urlunsplit((parsed.scheme, parsed.netloc, '/api/paas/v4/web_search', '', ''))
        api_key = require_api_key(self.spec.api_key_env)
        state = _search_state.get()
        if state is None:
            state = {'requests': 0, 'maximum': 8, 'latency_ms': 0, 'records': []}
        for attempt in range(1, self.max_attempts + 1):
            if state['requests'] >= state['maximum']:
                raise ValueError('Maximum physical search requests reached, including retries')
            receipt = observer.before_search(model, image_data_url, user_prompt, attempt) if observer else None
            error = None
            data = None
            response_headers = None
            state['requests'] += 1
            started = time.monotonic()
            try:
                with (self._search_http_client_factory() if self._search_http_client_factory else
                      httpx.Client(timeout=self.timeout, follow_redirects=False)) as client:
                    response = client.post(endpoint,
                        headers={'Authorization': 'Bearer ' + api_key},
                        json={'search_engine': 'search_pro', 'search_query': query,
                              'count': 10, 'content_size': 'medium', 'search_recency_filter': 'noLimit'})
                    response_headers = response.headers
                    if response.status_code >= 400:
                        raise NativeSearchHTTPError(response)
                    data = response.json()
                    if not isinstance(data, dict) or data.get('error'):
                        raise ValueError('Zhipu search returned an error body')
            except httpx.TimeoutException as exc:
                error = TimeoutError('Zhipu search timed out')
            except httpx.RequestError as exc:
                error = APIConnectionError('Zhipu search connection failed')
            except BaseException as exc:
                error = exc
            finally:
                state['latency_ms'] += int((time.monotonic() - started) * 1000)
            state['records'].append({'request': state['requests'], 'attempt': attempt, 'query': query,
                                     'response': _safe_json(data),
                                     'error': error_summary(error) if isinstance(error, Exception) else
                                              type(error).__name__ if error else None})
            if observer:
                capture = getattr(observer, 'observe_search_response', None)
                if callable(capture):
                    capture(receipt, model, response=data, headers=response_headers)
                observer.after_search(receipt, model, units=1, error=error)
            if error is None:
                return data
            if not isinstance(error, Exception):
                raise error
            if attempt < self.max_attempts and state['requests'] < state['maximum'] and classify_exception(error)[1]:
                delay = max(self.backoff_base * 2 ** (attempt - 1), self._retry_after_seconds(error) or 0.)
                if delay > self.MAX_AUTOMATIC_RETRY_WAIT_SECONDS:
                    raise error
                self._sleeper(delay)
                continue
            raise error
        raise RuntimeError('Search retry loop exhausted')
