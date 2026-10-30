"""Gemini is the sole inference provider for GreenLit."""
import json
import os
import re

import httpx
from pydantic import BaseModel, ValidationError

GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
DEFAULT_MODEL = 'gemini-3.5-flash-lite'


def provider() -> str:
    return 'gemini'


def validate_model(model: str) -> str:
    if not re.fullmatch(r'gemini-[a-zA-Z0-9._-]+', model):
        raise ValueError('GEMINI_MODEL must be a Gemini model ID, not a URL or local model name')
    return model


def model_settings(judge: bool = False) -> dict:
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key:
        raise ValueError('GEMINI_API_KEY is missing. Run: python -m greenlit.cli configure-gemini')
    model = (os.getenv('JUDGE_MODEL') if judge else '') or os.getenv('GEMINI_MODEL', DEFAULT_MODEL)
    return {'model': validate_model(model), 'base_url': GEMINI_BASE_URL, 'api_key': key,
            'temperature': 1, 'timeout': 60, 'max_retries': 0, 'max_tokens': 16000}


async def gemini_chat(system: str, messages: list[dict], schema: type[BaseModel], judge: bool = False):
    settings = model_settings(judge)
    model = settings['model']
    config = {'temperature': settings['temperature'], 'maxOutputTokens': settings['max_tokens'],
              'responseMimeType': 'application/json', 'responseJsonSchema': schema.model_json_schema()}
    if model.startswith(('gemini-3', 'gemini-4')):
        config['thinkingConfig'] = {'thinkingLevel': 'low'}
    elif model.startswith('gemini-2.5-flash'):
        config['thinkingConfig'] = {'thinkingBudget': 0}
    contents = [{'role': 'model' if m['role'] == 'assistant' else 'user',
                 'parts': [{'text': m['content']}]} for m in messages]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings['timeout'], connect=10),
                                     trust_env=False, follow_redirects=False) as client:
            response = await client.post(f'{GEMINI_BASE_URL}/models/{model}:generateContent',
                headers={'x-goog-api-key': settings['api_key']},
                json={'systemInstruction': {'parts': [{'text': system}]},
                      'contents': contents, 'generationConfig': config})
    except httpx.TimeoutException as exc:
        raise ValueError('GEMINI generation timed out. No questions or judgments were generated; model access and network need checking.') from exc
    except httpx.HTTPError as exc:
        raise ValueError('GEMINI network request failed. Check connectivity to generativelanguage.googleapis.com.') from exc
    if response.status_code != 200:
        explanations = {400: 'request rejected; check model and API key type', 401: 'invalid credentials',
                        403: 'key or project lacks model access', 404: 'model unavailable',
                        429: 'quota or rate limit exceeded'}
        raise ValueError(f"GEMINI HTTP {response.status_code}: {explanations.get(response.status_code, 'provider request failed')}")
    try:
        data = response.json()
        candidates = data.get('candidates', [])
        if not candidates:
            raise ValueError('GEMINI returned no candidate; the provider may have blocked the request')
        candidate = candidates[0]
        if candidate.get('finishReason') != 'STOP':
            raise ValueError('GEMINI response did not complete; evaluation is incomplete')
        text = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                       if not part.get('thought'))
        result = schema.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        location = '.'.join(map(str, first.get('loc', []))) or 'response'
        raise ValueError('GEMINI returned invalid structured output at ' + location + ' (' + first['type'] + '); evaluation is incomplete') from exc
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        if str(exc).startswith('GEMINI '):
            raise
        raise ValueError('GEMINI returned invalid structured output; evaluation is incomplete') from exc
    counters = {}
    usage = data.get('usageMetadata') or {}
    for source, target in [('promptTokenCount', 'prompt_tokens'), ('candidatesTokenCount', 'completion_tokens'),
                           ('totalTokenCount', 'total_tokens')]:
        value = usage.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            counters[target] = value
    return result, counters


async def gemini_structured(system: str, payload: dict, schema: type[BaseModel], judge: bool = False):
    result, _ = await gemini_chat(system, [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], schema, judge)
    return result


async def structured(system: str, payload: dict, schema: type[BaseModel], judge: bool = False):
    return await gemini_structured(system, payload, schema, judge)
