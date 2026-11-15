import httpx

from vcrb.providers.errors import classify_exception
from vcrb.providers.dashscope_anthropic import DashScopeHTTPError
from vcrb.providers.native_search_common import NativeSearchHTTPError


def test_dashscope_arrearage_http400_is_account_denial_not_retryable():
    error = DashScopeHTTPError(httpx.Response(400, json={'code':'Arrearage','message':'Access denied'}))
    assert classify_exception(error) == ('insufficient_balance', False)


def test_dashscope_sse_error_retains_business_code():
    response = httpx.Response(400, text='event: error\ndata: {"code":"Arrearage","message":"Access denied"}\n\n')
    error = NativeSearchHTTPError(response)
    assert error.body['code'] == 'Arrearage'
    assert classify_exception(error) == ('insufficient_balance', False)


def test_content_filter_bad_request_is_not_misclassified_as_balance():
    error = NativeSearchHTTPError(httpx.Response(400, json={'code':'DataInspectionFailed','message':'Content filter'}))
    assert classify_exception(error) == ('bad_request', False)


def test_http200_sse_balance_error_is_failure_and_preserves_usage():
    from vcrb.models import ProviderSpec
    from vcrb.providers.dashscope_multimodal_search import DashScopeMultimodalSearchProvider, completion_from_multimodal_events
    provider = DashScopeMultimodalSearchProvider(ProviderSpec('dashscope','https://dashscope.aliyuncs.com/compatible-mode/v1','UNUSED_KEY'))
    response = completion_from_multimodal_events([{'code':'Arrearage','message':'Access denied',
                                                 'usage':{'input_tokens':4,'output_tokens':2}}])
    result = provider._parse_response(response, 1, 1)
    assert not result.ok and result.error_type == 'insufficient_balance'
    assert result.usage['prompt_tokens'] == 4
