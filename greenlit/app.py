import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import demo, storage
from .config import allowed_hosts, mode
from .models import RunRequest, OverrideRequest, ResetOverrideRequest, ApprovalRequest
from .llm import model_settings, provider
from .reporting import markdown
from .security import safe_error, validate_target
from .workflow import run_evaluation

jobs = set()
lock = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app):
    storage.recover_interrupted()
    yield
    for task in jobs:
        task.cancel()
    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)


app = FastAPI(title='GreenLit', version='0.5.0', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])
static = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=static), name='static')


@app.middleware('http')
async def local_origin(request: Request, call_next):
    origin = request.headers.get('origin')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin:
        parsed = urlsplit(origin)
        if parsed.netloc != request.headers.get('host') or parsed.scheme not in {'http', 'https'}:
            return JSONResponse({'detail': 'Cross-origin requests are disabled'}, status_code=403)
    # Keep documents/configs within the local MVP input budget.
    if request.method == 'POST':
        try:
            limit=11000000 if request.url.path == '/api/reference' else 250000
            if int(request.headers.get('content-length', '0')) > limit:
                return JSONResponse({'detail': 'Request too large'}, status_code=413)
        except ValueError:
            return JSONResponse({'detail': 'Invalid content length'}, status_code=400)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response


@app.get('/')
def index():
    return FileResponse(static / 'index.html')


@app.get('/api/settings')
def settings():
    import os
    error = None
    try:
        config = model_settings()
    except ValueError as exc:
        config = {}
        error = str(exc)
    return {'mode': mode(), 'allowed_hosts': sorted(allowed_hosts()),
            'provider': provider(), 'model': config.get('model'),
            'configuration_error': error, 'llm_configured': bool(config), 'version': '0.5.0'}


@app.post('/api/reload-settings')
def reload_settings():
    from dotenv import load_dotenv
    from .config import ROOT
    if jobs:
        raise HTTPException(409, 'Wait for active evaluations before reloading settings')
    load_dotenv(ROOT / '.env', override=True)
    return settings()


def require_test_fixtures():
    import os
    if os.getenv('GREENLIT_TEST_FIXTURES') != '1':
        raise HTTPException(404,'Test fixtures are disabled')


@app.get('/api/config-template')
def config_template():
    return {'name':'My application evaluation','purpose':'','reference':'',
            'target':{'url':'','application_type':'rag','contexts_path':'retrieved_contexts',
                      'tool_trace_path':'tool_calls','request_template':{'message':'{{question}}'},
                      'response_path':'answer','api_key_env':'','auth_header':'Authorization','timeout_seconds':120 if provider()=='ollama' else 20},
            'load_test':{'enabled':False,'requests':6,'concurrency':2},'test_count':6 if provider()=='ollama' else 24,'repeats':1 if provider()=='ollama' else 2,'rules':{'require_full_coverage':True,'go_score':.8,'no_go_score':.5,'max_p95_ms':5000,
                'max_regression_drop':0,'min_judge_confidence':.7},'compare_previous':True,'baseline_id':None}


@app.post('/api/reference')
async def reference_file(file: UploadFile = File(...)):
    from io import BytesIO
    content=await file.read(10000001)
    await file.close()
    if len(content)>10000000:
        raise HTTPException(413,'Reference file must be smaller than 10 MB')
    suffix=Path(file.filename or '').suffix.lower()
    try:
        if suffix=='.pdf':
            from pypdf import PdfReader
            reader=PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs are not supported')
            if len(reader.pages)>100:
                raise ValueError('PDF is limited to 100 pages')
            parts=[]
            for page in reader.pages:
                parts.append(page.extract_text() or '')
                if sum(map(len,parts))>50000:
                    raise ValueError('Reference exceeds 50000 characters; upload a smaller document')
            text='\n'.join(parts)
        elif suffix in {'.txt','.md'}:
            text=content.decode('utf-8-sig')
        else:
            raise ValueError('Upload PDF, TXT or Markdown')
        if len(text)>50000:
            raise ValueError('Reference exceeds 50000 characters')
        if len(text.strip())<20:
            raise ValueError('Not enough readable text. Scanned PDFs need OCR before upload.')
    except Exception as exc:
        message=str(exc) if isinstance(exc,(ValueError,UnicodeError)) else 'Unable to extract text from this file'
        raise HTTPException(422,message) from exc
    return {'text':text,'characters':len(text),'filename':file.filename}


@app.post('/api/check-model')
async def check_model():
    from .llm import structured
    from pydantic import BaseModel
    class Probe(BaseModel):
        message: Literal['connected']
    try:
        result=await structured('Return JSON with message exactly connected.',{},Probe)
        if result.message != 'connected':
            raise ValueError('Model generation probe returned unexpected content')
        return {'generation_verified':True,'provider':provider()}
    except Exception as exc:
        raise HTTPException(502,safe_error(exc)) from exc


@app.get('/api/demo-config')
def demo_config(request: Request):
    require_test_fixtures()
    return RunRequest(purpose=demo.PURPOSE, reference=demo.REFERENCE,
                      target={'url': str(request.base_url).rstrip('/') + '/demo/chat'},rules={'require_full_coverage':False}).model_dump()


@app.post('/demo/chat')
async def demo_chat(request: Request):
    require_test_fixtures()
    body = await request.json()
    question = body.get('message', '')
    if not isinstance(question, str):
        raise HTTPException(422, 'message must be text')
    return {'answer': demo.answer(question)}


@app.post('/demo/chat-faulty')
async def faulty_chat(request: Request):
    require_test_fixtures()
    body = await request.json()
    question = body.get('message', '')
    if not isinstance(question, str):
        raise HTTPException(422, 'message must be text')
    return {'answer': demo.answer(question, faulty=True)}


async def execute(run_id, config):
    try:
        async with lock:
            await run_evaluation(run_id, config)
    except asyncio.CancelledError:
        storage.update(run_id, status='failed', stage='interrupted', error='Run interrupted by shutdown.')
        raise
    except Exception as exc:
        row=storage.get(run_id)
        stage=(row or {}).get('stage','unknown')
        storage.update(run_id, status='failed', stage='failed:'+stage, error=safe_error(exc))


@app.post('/api/runs', status_code=202)
async def start(config: RunRequest):
    try:
        validate_target(config.target.url)
        if '/demo/' in urlsplit(config.target.url).path:
            require_test_fixtures()
        if mode() == "llm":
            model_settings()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(jobs) >= 3:
        raise HTTPException(429, 'Evaluation queue is full. Wait for an active run to finish.')
    run_id = uuid.uuid4().hex
    storage.create(run_id, config.model_dump())
    task = asyncio.create_task(execute(run_id, config))
    jobs.add(task)
    task.add_done_callback(jobs.discard)
    return {'id': run_id}


@app.get('/api/runs')
def list_runs():
    return storage.history()


def require_run(run_id):
    row = storage.get(run_id)
    if not row:
        raise HTTPException(404, 'Evaluation not found')
    if row.get('report'):
        from .review import initialize
        from .improvements import build_suggestions
        if 'review' not in row['report']:initialize(row['report'])
        if 'improvements' not in row['report']:
            row['report']['improvements']=build_suggestions(row['report']['dataset'].get('cases',[]),row['report']['results'])
    return row


@app.get('/api/runs/{run_id}')
def get_run(run_id: str):
    return require_run(run_id)


@app.post('/api/runs/{run_id}/baseline')
def approve_baseline(run_id: str):
    try:
        storage.approve_baseline(run_id)
    except LookupError as exc:raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc
    return {'approved':True}


def save_review(run_id,action,data):
    try:
        return storage.append_review(run_id,action,data.model_dump())
    except LookupError as exc:raise HTTPException(404,str(exc)) from exc
    except storage.RevisionConflict as exc:raise HTTPException(409,str(exc)) from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.post('/api/runs/{run_id}/overrides')
def override_judgment(run_id: str, data: OverrideRequest):
    return save_review(run_id,'override',data)


@app.post('/api/runs/{run_id}/overrides/reset')
def reset_override(run_id: str, data: ResetOverrideRequest):
    return save_review(run_id,'reset',data)


@app.post('/api/runs/{run_id}/approval')
def approve_release(run_id: str, data: ApprovalRequest):
    return save_review(run_id,'approve',data)


@app.get('/api/runs/{run_id}/report.json')
def json_report(run_id: str):
    row = require_run(run_id)
    if not row['report']:
        raise HTTPException(409, 'Report is not ready')
    return Response(json.dumps(row['report'], indent=2, ensure_ascii=False), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="greenlit-{row["id"]}.json"'})


@app.get('/api/runs/{run_id}/report.md')
def md_report(run_id: str):
    row = require_run(run_id)
    if not row['report']:
        raise HTTPException(409, 'Report is not ready')
    return Response(markdown(row['report']), media_type='text/markdown',
                    headers={'Content-Disposition': f'attachment; filename="greenlit-{row["id"]}.md"'})
