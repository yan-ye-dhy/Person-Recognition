from vcrb.providers.search_evidence import normalize_search_evidence


def test_qwen_intent_without_source_is_not_retrieval():
    d = normalize_search_evidence('dashscope', {'content': [{'type':'web_search_tool_result', 'content':{'type':'web_search_tool_result_error'}}], 'usage':{'x_tools':{'search':{'count':1}}}})
    assert not d['retrieved'] and not d['billing']['known_units']


def test_deepseek_uses_billable_units_not_number_of_sources():
    d = normalize_search_evidence('deepseek', {'choices':[{'message':{'server_tool_calls':[{'calls':[{'status':'completed','sources':[{'url':'https://example.com/a'},{'url':'https://example.com/b'}]}]}]}}], 'usage':{'server_tool_use':{'tool_search':9,'web_search_requests':1}}})
    assert d['retrieved'] and len(d['sources']) == 2
    assert d['billing']['units'] == {'search':9}


def test_kimi_search_with_no_results_is_billable_but_not_retrieved():
    d = normalize_search_evidence('moonshot', {'output':[{'type':'web_search_call','status':'completed','action':{'sources':[]}}]})
    assert d['attempted'] and not d['retrieved']
    assert d['billing'] == {'units':{'search':1},'known_units':True}


def test_sources_only_accept_http_and_reject_malformed_urls():
    d = normalize_search_evidence('zhipu', {'web_search':[{'link':'file:///tmp/x'},{'link':'https://[x'},{'link':'https://example.com'}]})
    assert [s['url'] for s in d['sources']] == ['https://example.com']
