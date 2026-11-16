"""DashScope's documented Anthropic server-search interface, one HTTP attempt.

This transport retains web_search_tool_result blocks and a declared max_uses,
instead of inferring retrieval completion from a search-intent usage counter.
It reuses the existing provider's admission, retry and accounting boundaries.
Official contract: https://help.aliyun.com/zh/model-studio/anthropic-api-messages
"""
from __future__ import annotations

import base64
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import require_api_key
from ..models import ModelSpec, ProviderSpec
from .base import GenerationParams
from .http_audit import response_identity
from .openai_compat import OpenAICompatProvider, _safe_json

CLIENT_MARKER = "x-anthropic-billing-header: cc_entrypoint=cli;"
SEARCH_TOOL_TYPE = "web_search_20250305"


def anthropic_endpoint(base_url: str) -> str:
    parsed=urlsplit(base_url)
    if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Anthropic-compatible endpoint requires an HTTPS provider origin')
    return urlunsplit((parsed.scheme,parsed.netloc,'/apps/anthropic/v1/messages','',''))


def _max_uses(value: Any) -> int:
    if type(value) is not int or value<1:
        raise ValueError('search_max_uses must be a positive integer')
    return value


def _image_block(data_url: str) -> dict:
    if not isinstance(data_url,str) or not data_url.startswith('data:image/'):
        raise ValueError('Only the supplied base64 image data URL is supported')
    header,separator,data=data_url.partition(',')
    media_type=header.removeprefix('data:').removesuffix(';base64')
    if not separator or not header.endswith(';base64') or media_type not in {'image/jpeg','image/png','image/gif','image/webp'}:
        raise ValueError('Unsupported image data URL')
    try:
        if not base64.b64decode(data,validate=True):
            raise ValueError('Empty image')
    except (ValueError,TypeError) as error:
        raise ValueError('Invalid image base64') from error
    return {'type':'image','source':{'type':'base64','media_type':media_type,'data':data}}


def build_native_payload(chat_kwargs: dict, *, max_uses: int=1) -> dict:
    """Translate supplied messages without adding any character names or facts.

    The mandatory client marker is explicitly present as a separate system block.
    Search and text inference stay within one server-handled request. A returned
    pause_turn is never silently followed by another unaccounted HTTP request.
    """
    maximum=_max_uses(chat_kwargs.get('_search_max_uses',max_uses))
    system=[{'type':'text','text':CLIENT_MARKER}]
    messages=[]
    for message in chat_kwargs['messages']:
        role,content=message['role'],message['content']
        if role=='system':
            if not isinstance(content,str):
                raise ValueError('Expected a text system message')
            system.append({'type':'text','text':content})
            continue
        if role not in {'user','assistant'}:
            raise ValueError('Unsupported message role')
        if isinstance(content,str):
            blocks=[{'type':'text','text':content}]
        elif isinstance(content,list):
            blocks=[]
            for item in content:
                if item.get('type')=='image_url':
                    blocks.append(_image_block(item.get('image_url',{}).get('url')))
                elif item.get('type')=='text' and isinstance(item.get('text'),str):
                    blocks.append({'type':'text','text':item['text']})
                else:
                    raise ValueError('Unsupported message content block')
        else:
            raise ValueError('Unsupported message content')
        messages.append({'role':role,'content':blocks})
    limit=chat_kwargs.get('max_tokens',chat_kwargs.get('max_completion_tokens'))
    if type(limit) is not int or limit<1:
        raise ValueError('A positive output-token limit is required')
    extra=chat_kwargs.get('extra_body') or {}
    forced=(extra.get('search_options') or {}).get('forced_search',False) is True
    payload={'model':chat_kwargs['model'],'max_tokens':limit,'system':system,'messages':messages,
             'thinking':{'type':'disabled'},'stream':False,
             'tools':[{'type':SEARCH_TOOL_TYPE,'name':'web_search','max_uses':maximum}],
             'tool_choice':{'type':'tool','name':'web_search'} if forced else {'type':'auto'}}
    if 'temperature' in chat_kwargs:
        payload['temperature']=chat_kwargs['temperature']
    return payload


def completion_from_native(data: dict) -> Any:
    """Separate final text, reasoning, sources and server search usage."""
    if not isinstance(data,dict) or not isinstance(data.get('content'),list):
        raise ValueError('Anthropic-compatible response lacks content blocks')
    text=[];reasoning=[];results=[];calls=[];annotations=[]
    for block in data['content']:
        if not isinstance(block,dict):
            continue
        kind=block.get('type')
        if kind=='text':
            if isinstance(block.get('text'),str):
                text.append(block['text'])
            if isinstance(block.get('citations'),list):
                annotations.extend(block['citations'])
        elif kind=='thinking' and isinstance(block.get('thinking'),str):
            reasoning.append(block['thinking'])
        elif kind=='server_tool_use':
            calls.append(deepcopy(block))
        elif kind=='web_search_tool_result':
            # Preserve errors too; an error result is not successful retrieval.
            results.append(deepcopy(block))
    native_usage=data.get('usage') or {}
    if not isinstance(native_usage,dict):
        native_usage={}
    usage=_safe_json(native_usage)
    def token(key):
        value=native_usage.get(key,0)
        return value if type(value) is int and value>=0 else 0
    # Anthropic cache-read/creation tokens are separate from uncached input.
    prompt=token('input_tokens')+token('cache_read_input_tokens')+token('cache_creation_input_tokens')
    completion=token('output_tokens')
    usage.update(prompt_tokens=prompt,completion_tokens=completion,total_tokens=prompt+completion)
    info={'api_mode':'dashscope_anthropic','native_stop_reason':data.get('stop_reason')}
    if calls:
        info['server_tool_uses']=_safe_json(calls)
    if results:
        info['web_search_tool_results']=_safe_json(results)
    if isinstance(native_usage.get('server_tool_use'),dict):
        info['server_tool_use']=_safe_json(native_usage['server_tool_use'])
    if isinstance(data.get('x_tools'),dict):
        info['x_tools']=_safe_json(data['x_tools'])
    message=SimpleNamespace(content='\n'.join(text),reasoning_content='\n'.join(reasoning),
                            search_info=info,annotations=_safe_json(annotations))
    stop_reason=data.get('stop_reason')
    finish={'max_tokens':'length','end_turn':'stop','stop_sequence':'stop','tool_use':'tool_calls'}.get(stop_reason,stop_reason)
    return SimpleNamespace(choices=[SimpleNamespace(message=message,finish_reason=finish)],model=data.get('model'),usage=usage,_native_data=data)


class DashScopeHTTPError(Exception):
    def __init__(self,response: httpx.Response):
        self.status_code=response.status_code
        self.response=response
        try:
            body=response.json()
        except ValueError:
            body={'message':'Provider returned a non-JSON HTTP error'}
        self.body=_safe_json(body)
        super().__init__(f'HTTP {self.status_code}: {self.body}')


class APIConnectionError(ConnectionError):
    """Same classification name as the compatible SDK, without request secrets."""


class DashScopeAnthropicProvider(OpenAICompatProvider):
    name='dashscope_anthropic'

    @staticmethod
    def _parse_response(resp, attempts, latency_ms):
        from .search_evidence import normalize_search_evidence
        result = OpenAICompatProvider._parse_response(resp, attempts, latency_ms)
        data = getattr(resp, '_native_data', None)
        if isinstance(data, dict):
            result.search_metadata.update(normalize_search_evidence('dashscope', data))
        return result

    def __init__(self,provider_spec: ProviderSpec,*,max_uses: int=1,max_search_uses: int | None=None,
                 http_client_factory: Callable[[],httpx.Client] | None=None,**kwargs):
        super().__init__(provider_spec,**kwargs)
        if max_search_uses is not None:
            if max_uses!=1 and max_uses!=max_search_uses:
                raise ValueError('Conflicting server-search caps')
            max_uses=max_search_uses
        self.max_uses=_max_uses(max_uses)
        self.endpoint=anthropic_endpoint(provider_spec.base_url)
        self._http_client_factory=http_client_factory
        self._http_client=None

    @property
    def transport_profile(self) -> dict:
        return {'api_mode':self.name,'endpoint':self.endpoint,'stream':False,
                'client_marker':CLIENT_MARKER,'search_tool_type':SEARCH_TOOL_TYPE,
                'default_search_max_uses':self.max_uses}

    def build_request_kwargs(self,model: ModelSpec,messages: list[dict],gen: GenerationParams) -> dict:
        kwargs=super().build_request_kwargs(model,messages,gen)
        kwargs.pop('stream',None);kwargs.pop('stream_options',None)
        kwargs['_search_max_uses']=_max_uses(model.params.get('search_max_uses',self.max_uses))
        return kwargs

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client=self._client_factory()
            else:
                api_key=require_api_key(self.spec.api_key_env)
                self._http_client=(self._http_client_factory() if self._http_client_factory else
                                   httpx.Client(timeout=self.timeout,follow_redirects=False))
                def create(**kwargs):
                    payload=build_native_payload(kwargs,max_uses=self.max_uses)
                    try:
                        response=self._http_client.post(self.endpoint,json=payload,headers={
                            'x-api-key':api_key,'anthropic-version':'2023-06-01','content-type':'application/json',
                        })
                    except httpx.TimeoutException as error:
                        raise TimeoutError('Anthropic-compatible request timed out') from error
                    except httpx.RequestError as error:
                        raise APIConnectionError('Anthropic-compatible connection failed') from error
                    if response.status_code>=400:
                        raise DashScopeHTTPError(response)
                    data=response.json()
                    completion=completion_from_native(data)
                    completion._http_audit_identity=response_identity(data,headers=response.headers)
                    return completion
                self._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return self._client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
