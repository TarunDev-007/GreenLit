import json
import os
import re
from urllib.parse import urlsplit
import httpx

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def provider() -> str:
    value = os.getenv("LLM_PROVIDER", "gemini").lower()
    if value not in {"gemini", "openai", "ollama"}:
        raise ValueError("LLM_PROVIDER must be gemini, openai or ollama")
    return value


def model_settings(judge: bool = False) -> dict:
    if provider() == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip('/')
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'localhost','127.0.0.1','::1'}
                or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path):
            raise ValueError('OLLAMA base URL must be a local http address, e.g. http://127.0.0.1:11434')
        model = (os.getenv('JUDGE_MODEL') if judge else '') or os.getenv('OLLAMA_MODEL','llama3.2:3b')
        if not re.fullmatch(r'[a-zA-Z0-9._:/-]+',model):
            raise ValueError('OLLAMA model name is invalid')
        return {'model':model, 'base_url':base_url, 'temperature':0,
                'timeout':300, 'max_tokens':2048, 'num_ctx':16384}
    if provider() == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is missing. Run: python -m greenlit.cli configure-gemini")
        model = (os.getenv("JUDGE_MODEL") if judge else "") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return {"model": model, "base_url": GEMINI_BASE_URL, "api_key": key,
                "temperature": 1, "timeout": 60, "max_retries": 0, "max_tokens": 16000}
    model = (os.getenv("JUDGE_MODEL") if judge else "") or os.getenv("LLM_MODEL", "")
    if not model:
        raise ValueError("LLM_MODEL is required with MODE=llm")
    return {"model": model, "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("LLM_API_KEY") or "local-no-key", "temperature": 0,
            "timeout": 90, "max_retries": 1, "max_tokens": 12000}


async def gemini_structured(system: str, payload: dict, schema: type[BaseModel], judge: bool = False):
    settings = model_settings(judge)
    model = settings['model']
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
        raise ValueError('GEMINI_MODEL must be a model ID, not a URL')
    config = {'temperature': 1, 'maxOutputTokens': 16000, 'responseMimeType': 'application/json'}
    if model.startswith(('gemini-3', 'gemini-4')):
        config['thinkingConfig'] = {'thinkingLevel': 'low'}
    elif model.startswith('gemini-2.5-flash'):
        config['thinkingConfig'] = {'thinkingBudget': 0}
    instructions = system + '\nReturn ONLY JSON matching this schema:\n' + json.dumps(schema.model_json_schema())
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), trust_env=False, follow_redirects=False) as client:
            response = await client.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                headers={'x-goog-api-key': settings['api_key']},
                json={'systemInstruction': {'parts':[{'text':instructions}]},
                      'contents':[{'role':'user','parts':[{'text':json.dumps(payload,ensure_ascii=False)}]}],
                      'generationConfig':config})
    except httpx.TimeoutException as exc:
        raise ValueError('GEMINI generation timed out. No questions or judgments were generated; model access and network need checking.') from exc
    except httpx.HTTPError as exc:
        raise ValueError('GEMINI network request failed. Check connectivity to generativelanguage.googleapis.com.') from exc
    if response.status_code != 200:
        explanations={400:'request rejected; check model and API key type',401:'invalid credentials',
                      403:'key or project lacks model access',404:'model unavailable',429:'quota or rate limit exceeded'}
        raise ValueError(f"GEMINI HTTP {response.status_code}: {explanations.get(response.status_code,'provider request failed')}")
    data = response.json()
    candidates = data.get('candidates', [])
    if not candidates:
        raise ValueError('GEMINI returned no candidate; the provider may have blocked the request')
    candidate = candidates[0]
    if candidate.get('finishReason') not in {None,'STOP'}:
        raise ValueError('GEMINI response did not complete (finish reason '+str(candidate.get('finishReason'))+'); evaluation is incomplete')
    text = ''.join(part.get('text','') for part in candidate.get('content',{}).get('parts',[]) if not part.get('thought'))
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        first=exc.errors(include_input=False)[0]
        location='.'.join(map(str,first.get('loc',[]))) or 'response'
        raise ValueError('GEMINI returned invalid structured output at '+location+' ('+first['type']+'); evaluation is incomplete') from exc
    except ValueError as exc:
        raise ValueError('GEMINI returned invalid structured output; evaluation is incomplete') from exc


async def ollama_chat(system: str, messages: list[dict], schema: type[BaseModel], judge: bool = False, output_schema: dict | None = None):
    """Native local inference, constrained by JSON schema; never falls back to a cloud provider."""
    settings = model_settings(judge)
    if provider() != 'ollama':
        raise ValueError('OLLAMA provider must be selected before local inference')
    output_schema = output_schema or schema.model_json_schema()
    instructions = system + '\nReturn ONLY JSON matching this schema:\n' + json.dumps(output_schema)
    chat_messages = [{'role':'system','content':instructions}, *messages]
    # UTF-8 bytes conservatively bound byte-level tokens. Reject rather than silently
    # truncating uploaded evidence or chat history when the local context is too small.
    if sum(len(m['content'].encode('utf-8')) for m in chat_messages) > settings['num_ctx'] - settings['max_tokens'] - 1024:
        raise ValueError('OLLAMA input exceeds the local context budget. Use a smaller reference document or shorter history.')
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings['timeout'],connect=10),
                                     trust_env=False,follow_redirects=False) as client:
            response = await client.post(settings['base_url']+'/api/chat',json={
                'model':settings['model'], 'messages':chat_messages,
                'stream':False, 'format':output_schema, 'keep_alive':'10m',
                'options':{'temperature':0,'num_ctx':settings['num_ctx'],'num_predict':settings['max_tokens']}})
    except httpx.TimeoutException as exc:
        raise ValueError('OLLAMA generation timed out after 300 seconds. Check local resources or use a smaller test batch.') from exc
    except httpx.HTTPError as exc:
        raise ValueError('OLLAMA connection failed. Start Ollama with ollama serve, then retry.') from exc
    if response.status_code != 200:
        hint = 'Model not installed. Run ollama pull '+settings['model'] if response.status_code==404 else 'Check Ollama service logs and available memory.'
        raise ValueError(f'OLLAMA HTTP {response.status_code}: {hint}')
    try:
        data = response.json()
        if data.get('done') is not True or data.get('done_reason') not in {None,'stop'}:
            raise ValueError('OLLAMA response was incomplete; no judgment was recorded.')
        result = schema.model_validate_json(data['message']['content'])
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        location = '.'.join(map(str,first.get('loc',[]))) or 'response'
        raise ValueError('OLLAMA returned invalid structured output at '+location+'; evaluation is incomplete.') from exc
    except (KeyError,TypeError,ValueError) as exc:
        if str(exc).startswith('OLLAMA '):
            raise
        raise ValueError('OLLAMA returned invalid structured output; evaluation is incomplete.') from exc
    usage = {}
    for source,target in [('prompt_eval_count','prompt_tokens'),('eval_count','completion_tokens')]:
        value = data.get(source)
        if isinstance(value,int) and not isinstance(value,bool) and value>=0:
            usage[target]=value
    if len(usage)==2:
        usage['total_tokens']=sum(usage.values())
    return result, usage


async def structured(system: str, payload: dict, schema: type[BaseModel], judge: bool = False):
    if provider() == 'ollama':
        from . import risks
        output_schema = schema.model_json_schema()
        if schema.__name__ == 'GeneratedCases':
            items = output_schema['properties']['cases']
            items.update(minItems=payload['count'],maxItems=payload['count'])
            case_schema = output_schema['$defs']['Case']
            # These fields are calculated by Python after source validation.
            for field in ['reference_evidence_verified','validation_notes']:
                case_schema['properties'].pop(field,None)
            tags = case_schema['properties']['risk_tags']
            tags.update(minItems=1,items={'type':'string','enum':risks.ALL})
            case_schema['required'] = list(dict.fromkeys([*case_schema['required'],'risk_tags','turns','expected_passages','reference_evidence']))
            if payload.get('category'):
                case_schema['properties']['category']={'type':'string','const':payload['category']}
        if schema.__name__ == 'Judgment':
            tags = list(dict.fromkeys(t for t in payload.get('case',{}).get('risk_tags',[]) if t in risks.ALL))
            output_schema['properties']['risk_metrics'] = {
                'type':'object', 'properties':{tag:{'$ref':'#/$defs/RiskMetric'} for tag in tags},
                'required':tags, 'additionalProperties':False}
            output_schema['required'] = list(dict.fromkeys([*output_schema['required'],'risk_metrics']))
            kind=payload.get('application_type','chatbot')
            if kind!='rag': output_schema['properties']['retrieval']={'type':'null'}
            if kind!='agent': output_schema['properties']['agent']={'type':'null'}
        result, _ = await ollama_chat(system,[{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],schema,judge,output_schema)
        if schema.__name__ == 'Judgment' and set(result.risk_metrics)!=set(tags):
            raise ValueError('OLLAMA returned invalid structured output for risk metrics; evaluation is incomplete.')
        return result
    if provider() == 'gemini':
        return await gemini_structured(system,payload,schema,judge)
    llm = ChatOpenAI(**model_settings(judge))
    instructions = system + "\nReturn ONLY valid JSON matching this schema:\n" + json.dumps(schema.model_json_schema())
    response = await llm.ainvoke([('system',instructions),('human',json.dumps(payload,ensure_ascii=False))])
    text = response.content
    if not isinstance(text,str):
        raise ValueError('LLM returned unsupported response content')
    text=text.strip()
    if text.startswith('```'):
        text='\n'.join(text.splitlines()[1:-1])
    return schema.model_validate_json(text)
