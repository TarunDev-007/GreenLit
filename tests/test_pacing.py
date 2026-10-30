import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from greenlit import pacing


def test_shared_slots_are_atomic_and_contain_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv('GEMINI_REQUEST_INTERVAL_SECONDS', '13')
    monkeypatch.setattr(pacing.time, 'time', lambda: 1000)
    settings = {'model': 'gemini-test', 'api_key': 'never-store-this-key'}
    # Separate connections emulate callers in the evaluator and math processes.
    assert pacing.reserve(settings) == 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        delays = list(pool.map(pacing.reserve, [settings] * 3))
    assert sorted(delays) == [13, 26, 39]
    from greenlit.config import data_dir
    content = (data_dir() / 'gemini-pacing.sqlite3').read_bytes()
    assert b'never-store-this-key' not in content
    assert pacing.reserve({'model': 'gemini-other', 'api_key': settings['api_key']}) == 0


async def test_async_pacing_waits_outside_network_request(monkeypatch):
    delays = []
    monkeypatch.setattr(pacing, 'reserve', lambda settings: 13)
    async def sleep(delay): delays.append(delay)
    monkeypatch.setattr(pacing.asyncio, 'sleep', sleep)
    await pacing.wait_turn({})
    assert delays == [13]


@pytest.mark.parametrize('value', ['-1', '61', 'NaN', 'inf', 'bad'])
def test_invalid_intervals_fail_before_sending_requests(monkeypatch, value):
    monkeypatch.setenv('GEMINI_REQUEST_INTERVAL_SECONDS', value)
    with pytest.raises(ValueError, match='GEMINI_REQUEST_INTERVAL_SECONDS'):
        pacing.reserve({'model': 'gemini-test', 'api_key': 'test-key'})


def test_quick_defaults_match_dashboard_and_request_model():
    from greenlit.app import config_template
    from greenlit.models import RunRequest
    template = config_template()
    parsed = RunRequest(purpose='Test a mathematics assistant', target={'url': 'http://localhost/chat'})
    assert template['test_count'] == parsed.test_count == 6
    assert template['repeats'] == parsed.repeats == 1
    assert not template['load_test']['enabled']


@pytest.mark.parametrize('quota_id,value,expected', [
    ('GenerateRequestsPerDay', '20', 'daily quota exceeded'),
    ('GenerateRequestsPerMinute', '0', 'no quota is available'),
    ('GenerateRequestsPerMinute', '5', 'retrying after 26 seconds'),
])
def test_quota_details_are_classified_without_raw_provider_text(quota_id, value, expected):
    response = httpx.Response(429, json={'error': {'message': 'PRIVATE-ERROR-CONTENT', 'details': [
        {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': quota_id, 'quotaValue': value}]},
        {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '25.5s'}]}})
    message = pacing.quota_message(response)
    assert expected in message and 'PRIVATE-ERROR-CONTENT' not in message


def test_quota_guidance_does_not_recommend_changing_models():
    from greenlit.improvements import build_suggestions
    rows = [{'case_id': 'case-001', 'trial': 1, 'category': 'correctness', 'judgment': None,
             'error': None, 'evaluation_error': 'GEMINI HTTP 429: quota or rate limit exceeded'}]
    suggestion = build_suggestions([], rows)[0]
    assert suggestion['category'] == 'quota'
    assert 'Google AI Studio' in suggestion['model_or_system_change']
    assert '6 questions' in suggestion['model_or_system_change']
