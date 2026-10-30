import json

import httpx
import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from greenlit import config, llm


class Answer(BaseModel):
    answer: str = Field(min_length=1)


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'private-test-key')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')


def intercept(monkeypatch, handler):
    actual = httpx.AsyncClient
    monkeypatch.setattr(llm.httpx, 'AsyncClient', lambda **kw: actual(transport=httpx.MockTransport(handler), **kw))


def response(text='{"answer":"4"}', **extra):
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}], **extra}


async def test_math_target_uses_native_gemini_history_schema_and_usage(gemini, monkeypatch):
    from greenlit.math_chatbot import chat, ChatRequest
    def handler(req):
        assert req.url.host == 'generativelanguage.googleapis.com'
        assert req.headers['x-goog-api-key'] == 'private-test-key'
        body = json.loads(req.content)
        assert [c['role'] for c in body['contents']] == ['user', 'model', 'user']
        assert body['contents'][0]['parts'][0]['text'] == 'Remember 4'
        assert body['generationConfig']['responseJsonSchema']['properties']['answer']['type'] == 'string'
        return httpx.Response(200, json=response(usageMetadata={'promptTokenCount': 20, 'candidatesTokenCount': 8, 'totalTokenCount': 30}))
    intercept(monkeypatch, handler)
    result = await chat(ChatRequest(messages=[{'role': 'user', 'content': 'Remember 4'},
                      {'role': 'assistant', 'content': 'OK'}, {'role': 'user', 'content': 'Which number?'}]))
    assert result['provider'] == 'gemini' and result['answer'] == '4'
    assert result['usage'] == {'prompt_tokens': 20, 'completion_tokens': 8, 'total_tokens': 30}


@pytest.mark.parametrize('status', [400, 401, 403, 404, 429, 500])
async def test_provider_errors_are_visible_without_fallback(gemini, monkeypatch, status):
    calls = []
    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(status, json={'error': 'private provider content'})
    intercept(monkeypatch, handler)
    with pytest.raises(ValueError, match=f'GEMINI HTTP {status}') as caught:
        await llm.structured('Answer.', {}, Answer)
    assert len(calls) == 1 and 'generativelanguage.googleapis.com' in calls[0]
    assert 'private provider content' not in str(caught.value)


@pytest.mark.parametrize('body', [
    {'candidates': []}, response('{"wrong":"4"}'), response('{"answer":""}'),
    {'candidates': [{'finishReason': 'MAX_TOKENS'}]},
    {'candidates': [{'content': {'parts': [{'text': '{"answer":"4"}'}]}}]},
    {'candidates': [{'finishReason': 'SAFETY'}]},
])
async def test_incomplete_blocked_or_invalid_answers_never_pass(gemini, monkeypatch, body):
    intercept(monkeypatch, lambda req: httpx.Response(200, json=body))
    with pytest.raises(ValueError, match='GEMINI'):
        await llm.structured('Answer.', {}, Answer)


async def test_bad_response_json_is_sanitized(gemini, monkeypatch):
    intercept(monkeypatch, lambda req: httpx.Response(200, text='not json: private-provider-data'))
    with pytest.raises(ValueError, match='GEMINI returned invalid structured output'):
        await llm.structured('Answer.', {}, Answer)


async def test_network_failure_has_no_local_or_cloud_fallback(gemini, monkeypatch):
    def handler(req): raise httpx.ConnectError('private network detail', request=req)
    intercept(monkeypatch, handler)
    with pytest.raises(ValueError, match='GEMINI network request failed'):
        await llm.structured('Answer.', {}, Answer)


def test_only_gemini_is_used_even_with_legacy_provider_settings(gemini, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('OLLAMA_MODEL', 'llama3.2:3b')
    monkeypatch.setenv('LLM_BASE_URL', 'https://other.example')
    assert llm.provider() == 'gemini'
    assert llm.model_settings()['base_url'] == llm.GEMINI_BASE_URL
    assert llm.model_settings()['model'] == 'gemini-3.5-flash-lite'


@pytest.mark.parametrize('model', ['llama3.2:3b', 'https://other.example', 'gemini-x\nBAD=1'])
def test_non_gemini_or_unsafe_judge_model_rejected(gemini, monkeypatch, model):
    monkeypatch.setenv('JUDGE_MODEL', model)
    with pytest.raises(ValueError, match='Gemini model ID'):
        llm.model_settings(judge=True)


def test_private_configuration_preserves_target_settings_and_updates_judge(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    path = tmp_path / '.env'
    path.write_text('LLM_PROVIDER=ollama\nJUDGE_MODEL=llama3.2:3b\nCHATBOT_API_KEY=target-test-value\nALLOWED_TARGET_HOSTS=localhost\n')
    config.configure_gemini('new-private-test-key', 'gemini-3.5-flash-lite')
    values = dotenv_values(path)
    assert values['LLM_PROVIDER'] == 'gemini' and values['JUDGE_MODEL'] == 'gemini-3.5-flash-lite'
    assert values['GEMINI_API_KEY'] == 'new-private-test-key'
    assert values['CHATBOT_API_KEY'] == 'target-test-value' and values['ALLOWED_TARGET_HOSTS'] == 'localhost'


def test_invalid_configuration_does_not_change_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    path = tmp_path / '.env'
    path.write_text('CHATBOT_API_KEY=target-test-value\n')
    before = path.read_bytes()
    with pytest.raises(ValueError): config.configure_gemini('', 'gemini-3.5-flash-lite')
    with pytest.raises(ValueError): config.configure_gemini('new-private-test-key', 'llama3.2:3b')
    assert path.read_bytes() == before


async def test_math_target_key_required_and_quota_status(gemini, monkeypatch):
    from greenlit.math_chatbot import app
    real = httpx.AsyncClient
    intercept(monkeypatch, lambda req: httpx.Response(429, json={}))
    async with real(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
        result = await client.post('/chat', json={'message': '2+2'})
        assert result.status_code == 429
        monkeypatch.delenv('GEMINI_API_KEY')
        result = await client.post('/chat', json={'message': '2+2'})
        assert result.status_code == 503


async def test_dashboard_missing_key_and_probe(gemini, monkeypatch):
    from greenlit.app import app
    monkeypatch.setenv('MODE', 'llm')
    real = httpx.AsyncClient
    intercept(monkeypatch, lambda req: httpx.Response(200, json=response('{"message":"connected"}')))
    async with real(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
        result = await client.post('/api/check-model')
        assert result.status_code == 200 and result.json()['provider'] == 'gemini'
        monkeypatch.delenv('GEMINI_API_KEY')
        settings = (await client.get('/api/settings')).json()
        assert settings['provider'] == 'gemini' and not settings['llm_configured']
        assert 'configure-gemini' in settings['configuration_error']
