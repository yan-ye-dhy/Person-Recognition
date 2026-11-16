"""DashScope original multimodal streaming search, with returned source proof.

Official contract: https://help.aliyun.com/zh/model-studio/web-search
The entire SSE exchange stays inside one existing admitted HTTP attempt.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import require_api_key
from .native_search_common import APIConnectionError, NativeSearchHTTPError, NativeSearchProvider, https_endpoint, validate_image_data_url
from .openai_compat import _merge_mapping, _safe_json
from .http_audit import response_identity


NATIVE_PATH = '/api/v1/services/aigc/multimodal-generation/generation'


def build_multimodal_payload(chat_kwargs: dict) -> dict:
    messages = []
    for message in chat_kwargs['messages']:
        role, content = message['role'], message['content']
        if isinstance(content, str):
            blocks = [{'text': content}]
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if block.get('type') == 'image_url':
                    blocks.append({'image': validate_image_data_url((block.get('image_url') or {}).get('url'))})
                elif block.get('type') == 'text' and isinstance(block.get('text'), str):
                    blocks.append({'text': block['text']})
                else:
                    raise ValueError('Unsupported multimodal content block')
        else:
            raise ValueError('Unsupported multimodal message content')
        messages.append({'role': role, 'content': blocks})
    extra = chat_kwargs.get('extra_body') or {}
    if extra.get('enable_thinking', False) is not False:
        raise ValueError('Multimodal search profile must keep thinking disabled')
    limit = chat_kwargs.get('max_tokens', chat_kwargs.get('max_completion_tokens'))
    if type(limit) is not int or limit < 1:
        raise ValueError('A positive output-token limit is required')
    parameters = {'enable_search': True, 'enable_thinking': False, 'incremental_output': True,
                  'result_format': 'message', 'max_tokens': limit,
                  'search_options': {'forced_search': True, 'search_strategy': 'max', 'enable_source': True}}
    if 'temperature' in chat_kwargs:
        parameters['temperature'] = chat_kwargs['temperature']
    return {'model': chat_kwargs['model'], 'input': {'messages': messages}, 'parameters': parameters}


def normalize_multimodal_search(native: dict) -> dict:
    sources, queries, infos = [], [], []
    seen = set()
    frames = native.get('events') if isinstance(native.get('events'), list) else [native]
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        output = frame.get('output') or {}
        candidates = [frame.get('search_info'), output.get('search_info') if isinstance(output, dict) else None]
        for info in candidates:
            if not isinstance(info, dict):
                continue
            infos.append(_safe_json(info))
            for key in ('query', 'queries', 'search_query', 'search_queries'):
                values = info.get(key)
                for value in values if isinstance(values, list) else [values]:
                    if isinstance(value, str) and value.strip() and value not in queries:
                        queries.append(value)
            for item in info.get('search_results') or []:
                if not isinstance(item, dict) or not isinstance(item.get('url'), str):
                    continue
                url = item['url']
                try:
                    parsed = urlsplit(url)
                except ValueError:
                    continue
                if parsed.scheme in {'http', 'https'} and parsed.netloc and url not in seen:
                    sources.append({'url': url, 'title': str(item.get('title') or '')})
                    seen.add(url)
    return {'requested': True, 'attempted': True if infos else None, 'retrieved': bool(sources),
            'sources': sources, 'queries': queries, 'billing': {'units': {}, 'known_units': False},
            'api_mode': 'dashscope_multimodal_search', 'search_info': infos,
            'errors': _safe_json(native.get('errors') or []), 'native': _safe_json(native)}


def completion_from_multimodal_events(events: list[dict]):
    text, reasoning, usage, finish, model, errors = [], [], {}, None, None, []
    for frame in events:
        if not isinstance(frame, dict):
            raise ValueError('Multimodal SSE data must be a JSON object')
        code = frame.get('code')
        if (code not in (None, '', 0, '0', 200, '200')) or frame.get('error'):
            errors.append(_safe_json(frame))
        snapshot = frame.get('usage')
        if isinstance(snapshot, dict):
            _merge_mapping(usage, _safe_json(snapshot))  # cumulative snapshots, never sum
        model = frame.get('model') or model
        output = frame.get('output') or {}
        if not isinstance(output, dict):
            continue
        choices = output.get('choices') or []
        if not choices and isinstance(output.get('text'), str):
            text.append(output['text'])
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                continue
            candidate = choice.get('finish_reason')
            if candidate not in (None, '', 'null'):
                finish = candidate
            message = choice.get('message') or {}
            content = message.get('content')
            if isinstance(content, str):
                text.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get('text'), str):
                        text.append(block['text'])
            thought = message.get('reasoning_content')
            if isinstance(thought, str):
                reasoning.append(thought)
    for source, target in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]:
        if type(usage.get(source)) is int and usage[source] >= 0:
            usage[target] = usage[source]
    if 'total_tokens' not in usage and all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
        usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
    native = {'events': _safe_json(events), 'usage': usage, 'errors': errors}
    error = {'error_type': 'server_error', 'native_errors': errors} if errors else None
    if finish is None and not errors:
        error = {'error_type': 'server_error', 'message': 'SSE ended without a final finish_reason'}
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content=''.join(text), reasoning_content=''.join(reasoning)))],
        model=model, usage=usage, _native_response=native, _native_error=error)


def consume_multimodal_stream(lines):
    events, data_lines = [], []
    def append_event():
        if data_lines:
            payload = '\n'.join(data_lines)
            if payload.strip() != '[DONE]':
                events.append(json.loads(payload))
            data_lines.clear()
    for line in lines:
        if not line:
            append_event()
        elif line.startswith('data:'):
            data_lines.append(line[5:].lstrip())
    append_event()
    return completion_from_multimodal_events(events)


class DashScopeMultimodalSearchProvider(NativeSearchProvider):
    name = 'dashscope_multimodal_search'
    evidence_vendor = 'dashscope'
    native_payload = staticmethod(build_multimodal_payload)

    def __init__(self, provider_spec, **kwargs):
        super().__init__(provider_spec, **kwargs)
        parsed = urlsplit(https_endpoint(provider_spec.base_url, ''))
        self.endpoint = urlunsplit((parsed.scheme, parsed.netloc, NATIVE_PATH, '', ''))

    @property
    def transport_profile(self):
        return {'api_mode': self.name, 'endpoint': self.endpoint, 'stream': True,
                'enable_thinking': False, 'enable_search': True, 'incremental_output': True,
                'search_options': {'forced_search': True, 'search_strategy': 'max', 'enable_source': True},
                'search_limit_semantics': 'Native max strategy; no verified hard search-call cap'}

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                api_key = require_api_key(self.spec.api_key_env)
                self._http_client = (self._http_client_factory() if self._http_client_factory else
                                     httpx.Client(timeout=self.timeout, follow_redirects=False))
                def create(**kwargs):
                    try:
                        with self._http_client.stream('POST', self.endpoint, json=self.native_payload(kwargs), headers={
                            'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json',
                            'X-DashScope-SSE': 'enable'}) as response:
                            if response.status_code >= 400:
                                response.read()
                                raise NativeSearchHTTPError(response)
                            if 'application/json' in response.headers.get('content-type', ''):
                                response.read()
                                completion=completion_from_multimodal_events([response.json()])
                            else:
                                completion=consume_multimodal_stream(response.iter_lines())
                            completion._http_audit_identity=response_identity(completion,headers=response.headers)
                            return completion
                    except httpx.TimeoutException as error:
                        raise TimeoutError('Multimodal search stream timed out') from error
                    except httpx.RequestError as error:
                        raise APIConnectionError('Multimodal search stream connection failed') from error
                self._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return self._client

    def _parse_response(self, response, attempts, latency_ms):
        result = super()._parse_response(response, attempts, latency_ms)
        result.search_metadata = normalize_multimodal_search(getattr(response, '_native_response', {}))
        return result
