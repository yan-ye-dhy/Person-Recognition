"""Normalize provider-returned search evidence; configuration is not proof."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from .openai_compat import _safe_json

SEARCH_INSTRUCTION = '\n请使用已提供的联网搜索工具核实角色身份；检索内容仅作参考，以画面为准，最终仍只返回规定JSON。'


def normalize_search_evidence(vendor: str, raw: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, str]] = []
    queries: list[str] = []
    seen: set[str] = set()
    attempted: bool | None = None
    units: dict[str, int] = {}
    errors: list[Any] = []

    def source(item: Any, key: str = 'url') -> None:
        if not isinstance(item, dict):
            return
        url = item.get(key)
        if not isinstance(url, str):
            return
        try:
            parsed = urlsplit(url)
        except ValueError:
            return
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc or url in seen:
            return
        seen.add(url)
        sources.append({'url': url, 'title': str(item.get('title') or '')})

    def query(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in queries:
            queries.append(value)
        elif isinstance(value, list):
            for part in value:
                query(part)

    def count(name: str, value: Any) -> None:
        if type(value) is int and value >= 0:
            units[name] = value

    usage = raw.get('usage') or {}
    if not isinstance(usage, dict):
        usage = {}
    if vendor == 'deepseek':
        for choice in raw.get('choices', []):
            message = choice.get('message', {})
            for batch in message.get('server_tool_calls') or []:
                for call in batch.get('calls') or []:
                    attempted = True
                    query((call.get('action') or {}).get('queries'))
                    if call.get('status') == 'completed':
                        for item in call.get('sources') or []:
                            source(item)
                    if call.get('error'):
                        errors.append(call['error'])
        tool = usage.get('server_tool_use') or {}
        for field, name in [('tool_search', 'search'), ('tool_fetch', 'fetch'), ('tool_image_search', 'image_search')]:
            count(name, tool.get(field))
        # DeepSeek's display result-block count is not its billable query count.
    elif vendor == 'moonshot':
        output = raw.get('output')
        if isinstance(output, list):
            # Moonshot bills each returned web_search_call, including no-result
            # searches; source count is independent of the billable count.
            count('search', sum(isinstance(x, dict) and x.get('type') == 'web_search_call' for x in output))
        for item in raw.get('output') or []:
            if item.get('type') != 'web_search_call':
                continue
            attempted = True
            action = item.get('action') or {}
            query(action.get('query'))
            query(action.get('queries'))
            if item.get('status') == 'completed':
                for entry in action.get('sources') or []:
                    source(entry)
            if item.get('error'):
                errors.append(item['error'])
    elif vendor == 'dashscope':
        for item in raw.get('content') or []:
            if item.get('type') == 'server_tool_use' and item.get('name') == 'web_search':
                attempted = True
                query((item.get('input') or {}).get('query'))
                query((item.get('input') or {}).get('queries'))
            elif item.get('type') == 'web_search_tool_result':
                attempted = True
                content = item.get('content')
                if isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict) and entry.get('type') == 'web_search_result':
                            source(entry)
                elif isinstance(content, dict):
                    errors.append(content)
        count('search', (usage.get('server_tool_use') or {}).get('web_search_requests'))
    elif vendor == 'zhipu':
        for item in raw.get('web_search') or []:
            source(item, 'link')
        if sources:
            attempted = True
        count('search', (usage.get('server_tool_use') or {}).get('web_search_requests'))
    if raw.get('error'):
        errors.append(raw['error'])
    known_units = bool(units)
    if vendor == 'deepseek':
        # A partial counter set is retained, but must not release the nominal
        # allowance for other native operations whose metering is absent.
        tool = usage.get('server_tool_use') or {}
        known_units = all(type(tool.get(k)) is int and tool[k] >= 0
                          for k in ('tool_search', 'tool_fetch', 'tool_image_search'))
    return {'requested': True, 'attempted': attempted, 'retrieved': bool(sources),
            'sources': sources, 'queries': queries, 'errors': _safe_json(errors),
            'billing': {'units': units, 'known_units': known_units},
            'native': _safe_json(raw)}
