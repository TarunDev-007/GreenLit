import copy
import json
import time
import uuid

import httpx
import pytest

from greenlit import demo, storage, tools
from greenlit.mcp_client import connect_tools
from greenlit.models import Rules, RunRequest, Target
from greenlit.security import validate_target
from greenlit.workflow import run_evaluation


def request(url):
    return RunRequest(purpose=demo.PURPOSE, reference=demo.REFERENCE, target={'url': url + '/demo/chat'}, test_count=12,rules={'require_full_coverage':False})


async def execute(config):
    run_id = uuid.uuid4().hex
    storage.create(run_id, config.model_dump())
    return await run_evaluation(run_id, config)


async def test_mcp_discovers_and_executes_tools():
    async with connect_tools() as client:
        assert set(client.available) == {'plan_evaluation', 'generate_test_questions', 'run_chatbot_tests', 'evaluate_answers', 'compare_baseline', 'generate_report','measure_load'}
        dataset = await client.call('generate_test_questions', {'purpose': demo.PURPOSE, 'reference': demo.REFERENCE, 'count': 12})
        assert len(dataset['cases']) == 12
        assert dataset['dataset_hash'] == tools.digest(dataset['cases'])
        assert client.trace[0]['transport'] == 'stdio'


async def test_graph_good_faulty_baseline_and_persistence(service):
    config = request(service)
    good = await execute(config)
    assert good['decision']['verdict'] == 'GO'
    assert good['summary']['score'] == 1
    assert len(good['results']) == 24
    assert len(good['mcp']['calls']) == 6
    assert good['summary']['estimated_target_cost_usd'] is None
    storage.update(good['id'], approved=1)
    changed = config.model_copy(deep=True)
    changed.target.url = service + '/demo/chat-faulty'
    changed.baseline_id = good['id']
    bad = await execute(changed)
    assert bad['decision']['verdict'] == 'NO-GO'
    assert bad['summary']['hard_failures'] == 4
    assert bad['regression']['status'] == 'compared'
    assert bad['regression']['score_drop'] > 0
    assert bad['regression']['regressed_cases']
    assert bad['dataset'] == good['dataset']
    assert storage.get(bad['id'])['status'] == 'completed'


async def test_ai_mode_with_mock_provider(service, monkeypatch):
    monkeypatch.setenv('MODE', 'llm')
    monkeypatch.setenv('LLM_BASE_URL', service + '/v1')
    monkeypatch.setenv('LLM_MODEL', 'mock')
    monkeypatch.setenv('LLM_API_KEY', 'test-secret-never-store')
    monkeypatch.setenv('JUDGE_MODEL', 'mock')
    result = await execute(request(service))
    assert result['decision']['verdict'] == 'GO'
    assert result['mode'] == 'llm'
    assert 'test-secret-never-store' not in json.dumps(result)
    assert result['dataset']['source'].startswith('LLM-generated')


@pytest.mark.parametrize('path', ['/broken', '/malformed', '/redirect'])
async def test_bad_target_is_not_go(service, path):
    config = request(service)
    config.target.url = service + path
    report = await execute(config)
    assert report['summary']['errors'] == 24
    assert report['summary']['judged'] == 0
    assert report['decision']['verdict'] == 'REVIEW'
    if path == '/redirect':
        assert report['results'][0]['status_code'] == 307


async def test_judge_failure_is_review(service, monkeypatch):
    observations = (await tools.run_chatbot_tests({'url':service+'/demo/chat'}, demo.generate(12), 2))['observations']
    monkeypatch.setenv('MODE', 'llm')
    monkeypatch.delenv('LLM_MODEL', raising=False)
    results = (await tools.evaluate_answers(demo.PURPOSE, demo.REFERENCE, demo.generate(12), observations))['results']
    report = tools.generate_report(request(service).model_dump(), {'dataset_hash':'test'}, results)
    assert report['decision']['verdict'] == 'REVIEW'
    assert report['summary']['errors'] == 24


async def test_demo_rejects_other_reference():
    with pytest.raises(ValueError, match='MODE=demo'):
        await tools.generate_test_questions('An unrelated bot', 'This is a different reference document.', 12)


@pytest.mark.parametrize('url', ['http://example.com/chat', 'file:///tmp/secret', 'http://localhost/chat?key=secret', 'http://u:p@localhost/chat', 'http://169.254.169.254/latest'])
def test_target_restrictions(url):
    with pytest.raises(ValueError):
        validate_target(url)


def sample_results():
    return [{ 'case_id': case['id'], 'category': case['category'], 'trial': trial,
              'answer':demo.answer(case['question']), 'error':None,'evaluation_error':None,
              'latency_ms':10, 'estimated_cost_usd':None,
              'judgment':demo.judge(case,demo.answer(case['question']))}
            for case in demo.generate(12) for trial in [1,2]]


def test_uncertain_instability_and_hard_gate_precedence():
    rows = sample_results()
    rows[0]['judgment']['confidence'] = .2
    rows[1]['judgment']['verdict'] = 'fail'
    summary = tools.summarize(rows, Rules().model_dump())
    assert summary['uncertain'] == 1
    assert summary['unstable_cases']
    decision = tools.make_decision(summary, {'status':'not_requested'}, Rules().model_dump())
    assert decision['verdict'] == 'REVIEW'
    rows[-3]['judgment']['verdict'] = 'fail'  # harmless failure is not a safety gate
    rows[16]['judgment']['verdict'] = 'fail'  # safety test
    summary = tools.summarize(rows, Rules().model_dump())
    assert tools.make_decision(summary, {'status':'not_requested'}, Rules().model_dump())['verdict'] == 'NO-GO'


def test_missing_repeatability_requires_review():
    rows = [r for r in sample_results() if r['trial'] == 1]
    summary = tools.summarize(rows, Rules().model_dump())
    assert tools.make_decision(summary, {'status':'not_requested'}, Rules().model_dump())['verdict'] == 'REVIEW'


def test_unknown_cost_is_not_zero(monkeypatch):
    assert tools.usage_cost({'prompt_tokens':100,'completion_tokens':200}) is None
    monkeypatch.setenv('TARGET_INPUT_PRICE_PER_MILLION','1')
    monkeypatch.setenv('TARGET_OUTPUT_PRICE_PER_MILLION','2')
    assert tools.usage_cost({'prompt_tokens':100,'completion_tokens':200}) == pytest.approx(.0005)
    assert tools.usage_cost({'prompt_tokens':100}) is None


def test_template_preserves_quotes_and_nested_fields():
    question='"quoted" \\ text\nnew line'
    rendered=tools.render_template({'messages':[{'content':'{{question}}'}]},question)
    assert rendered['messages'][0]['content'] == question
    assert tools.extract_path({'choices':[{'message':{'content':'answer'}}]},'choices.0.message.content') == 'answer'
    with pytest.raises(ValueError):
        Target(url='http://localhost:9000',request_template={'message':'no variable'})


async def test_baseline_configuration_changes_rejected(service):
    config=request(service)
    report=await execute(config)
    storage.update(report['id'],approved=1)
    config.baseline_id=report['id']
    config.reference += '\nNew policy.'
    with pytest.raises(ValueError,match='Baseline purpose/reference changed'):
        await execute(config)


def test_http_dashboard_job_download_baseline_and_origin(service):
    with httpx.Client(base_url=service,trust_env=False,timeout=30) as client:
        page=client.get('/')
        assert page.status_code==200
        assert 'GreenLit' in page.text
        assert 'frame-ancestors' in page.headers['content-security-policy']
        config=client.get('/api/demo-config').json()
        config['test_count']=12
        started=client.post('/api/runs',json=config)
        assert started.status_code==202
        run_id=started.json()['id']
        for _ in range(200):
            row=client.get('/api/runs/'+run_id).json()
            if row['status'] in {'completed','failed'}:
                break
            time.sleep(.1)
        assert row['status']=='completed',row
        assert row['report']['decision']['verdict']=='GO'
        approval=client.post('/api/runs/'+run_id+'/approval',json={'reviewer':'Test reviewer','reason':'Reviewed all test evidence.','revision':0,'verdict':'GO'})
        assert approval.status_code==200
        assert client.post('/api/runs/'+run_id+'/baseline').status_code==200
        assert client.get('/api/runs/'+run_id+'/report.json').json()['id']==run_id
        assert '# GreenLit' in client.get('/api/runs/'+run_id+'/report.md').text
        assert client.post('/api/runs',json=config,headers={'Origin':'https://untrusted.example'}).status_code==403
        config['target']['url']='http://untrusted.example/chat'
        assert client.post('/api/runs',json=config).status_code==422


def test_incomplete_coverage_cannot_pass():
    config = RunRequest(purpose=demo.PURPOSE, reference=demo.REFERENCE, target={'url':'http://localhost/chat'}, test_count=12).model_dump()
    dataset = {'cases':demo.generate(12),'dataset_hash':'test'}
    report = tools.generate_report(config, dataset, sample_results()[:-1])
    assert report['summary']['coverage_complete'] is False
    assert report['decision']['verdict'] == 'REVIEW'


def test_latency_hard_gate():
    rows=sample_results()
    rows[0]['latency_ms']=10000
    rows[1]['latency_ms']=10000
    summary=tools.summarize(rows, Rules().model_dump())
    assert tools.make_decision(summary, {'status':'not_requested'}, Rules().model_dump())['verdict']=='NO-GO'



def test_missing_gemini_key_has_actionable_error(monkeypatch):
    from greenlit.llm import model_settings
    monkeypatch.setenv('LLM_PROVIDER','gemini')
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='configure-gemini'):
        model_settings()



async def test_native_gemini_request_and_response(monkeypatch):
    from greenlit import llm
    from greenlit.models import Judgment
    monkeypatch.setenv('LLM_PROVIDER','gemini')
    monkeypatch.setenv('GEMINI_API_KEY','private-fake-key')
    monkeypatch.setenv('GEMINI_MODEL','gemini-3.8-flash')
    captured={}
    judgment={'verdict':'pass','correctness':1,'relevance':1,'faithfulness':1,'confidence':.9,'reason':'Supported by the reference.'}
    def handler(request):
        captured.update(url=str(request.url),header=request.headers.get('x-goog-api-key'),body=json.loads(request.content))
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(judgment)}]}}]})
    actual=httpx.AsyncClient
    monkeypatch.setattr(llm.httpx,'AsyncClient',lambda **kwargs:actual(transport=httpx.MockTransport(handler),**kwargs))
    result=await llm.structured('Evaluate response.',{'answer':'8 PM'},Judgment)
    assert result.verdict=='pass'
    assert captured['url'].endswith('/models/gemini-3.8-flash:generateContent')
    assert captured['header']=='private-fake-key'
    assert captured['body']['generationConfig']['responseMimeType']=='application/json'


async def test_native_gemini_timeout_is_actionable(monkeypatch):
    from greenlit import llm
    from greenlit.models import Judgment
    monkeypatch.setenv('LLM_PROVIDER','gemini')
    monkeypatch.setenv('GEMINI_API_KEY','private-fake-key')
    def handler(request):
        raise httpx.ReadTimeout('timeout')
    actual=httpx.AsyncClient
    monkeypatch.setattr(llm.httpx,'AsyncClient',lambda **kwargs:actual(transport=httpx.MockTransport(handler),**kwargs))
    with pytest.raises(ValueError,match='generation timed out'):
        await llm.structured('Evaluate.',{},Judgment)


@pytest.mark.parametrize('kind,present,passed', [('rag',True,True),('rag',False,False),('agent',True,True),('agent',False,False)])
async def test_rag_and_agent_evidence_controls_verdict(monkeypatch,kind,present,passed):
    monkeypatch.setenv('MODE','llm')
    async def judge(*args,**kwargs):
        from greenlit.models import Judgment
        return Judgment(verdict='pass',correctness=1,relevance=1,faithfulness=1,confidence=.95,reason='Answer is correct.',
                        retrieval={'context_relevance':1,'answer_supported_by_context':True,'reason':'Supported by the retrieved text.'} if kind=='rag' else None,
                        agent={'tool_use_correctness':1,'task_success':'pass','reason':'Tool result confirms the requested action.'} if kind=='agent' else None)
    monkeypatch.setattr(tools,'structured',judge)
    row={'case_id':'case-001','category':'correctness','trial':1,'answer':'8 PM','error':None,'latency_ms':1,
         'application_type':kind,'retrieval_observed':present,'trace_observed':present,
         'retrieved_contexts':['Library closes at 8 PM.'] if present else [],
         'tool_calls':[{'name':'lookup','result':'8 PM'}] if present else []}
    result=(await tools.evaluate_answers(demo.PURPOSE,demo.REFERENCE,demo.generate(12),[row]))['results'][0]
    assert result['judgment']['verdict']==('pass' if passed else 'review')


async def test_unsupported_rag_answer_fails(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    from greenlit.models import Judgment
    async def judge(*args,**kwargs):
        return Judgment(verdict='pass',correctness=1,relevance=1,faithfulness=1,confidence=1,reason='Correct against general reference.',
                        retrieval={'context_relevance':0,'answer_supported_by_context':False,'reason':'Answer is not in retrieved passages.'})
    monkeypatch.setattr(tools,'structured',judge)
    row={'case_id':'case-001','category':'correctness','trial':1,'answer':'8 PM','error':None,'latency_ms':1,
         'application_type':'rag','retrieval_observed':True,'retrieved_contexts':['Other information.']}
    result=(await tools.evaluate_answers(demo.PURPOSE,demo.REFERENCE,demo.generate(12),[row]))['results'][0]
    assert result['judgment']['verdict']=='fail'
    assert result['judgment']['faithfulness']==0


def test_production_demo_disabled(monkeypatch):
    from greenlit.config import mode
    monkeypatch.delenv('GREENLIT_TEST_FIXTURES',raising=False)
    with pytest.raises(ValueError,match='disabled'):
        mode()


def test_reference_upload_and_real_config_template(service):
    with httpx.Client(base_url=service,trust_env=False) as client:
        template=client.get('/api/config-template').json()
        assert template['target']['url']==''
        assert template['reference']==''
        r=client.post('/api/reference',files={'file':('source.txt',b'The college library closes at 8 PM.','text/plain')})
        assert r.status_code==200
        assert '8 PM' in r.json()['text']
        assert client.post('/api/reference',files={'file':('bad.exe',b'not a document')}).status_code==422


def test_trace_secrets_redacted():
    assert tools.redact_trace([{'tool':'lookup','authorization':'private'}])[0]['authorization']=='[REDACTED]'


@pytest.mark.parametrize('kind,path',[('rag','/rag'),('agent','/agent')])
async def test_actual_connector_captures_rag_or_agent_evidence(service,kind,path):
    observed=(await tools.run_chatbot_tests({'url':service+path,'application_type':kind},demo.generate(12)[:1],1))['observations'][0]
    assert observed['error'] is None
    if kind=='rag':
        assert observed['retrieval_observed'] is True
        assert observed['retrieved_contexts']==['The library closes at 8 PM.']
    else:
        assert observed['trace_observed'] is True
        assert observed['tool_calls'][0]['authorization']=='[REDACTED]'


async def test_multiturn_sends_actual_history(service):
    case=demo.generate(12)[0]
    case['turns']=['Remember that my name is Riya.','What is my name?']
    case['risk_tags']=['context_retention']
    target={'url':service+'/conversation','conversation_request_template':{'messages':'{{messages}}'}}
    row=(await tools.run_chatbot_tests(target,[case],1))['observations'][0]
    assert row['error'] is None
    assert len(row['conversation'])==4
    assert row['answer']=='Your name is Riya.'


async def test_multiturn_missing_adapter_is_not_success(service):
    case=demo.generate(12)[0]
    case['turns']=['Remember my name.','What is it?']
    row=(await tools.run_chatbot_tests({'url':service+'/demo/chat'},[case],1))['observations'][0]
    assert row['error']


async def test_bounded_load_measurement(service):
    report=await tools.measure_load({'url':service+'/demo/chat'},demo.generate(12)[:1],requests=4,concurrency=2)
    assert report['status']=='measured'
    assert report['requests']==4
    assert report['failure_rate']==0
    with pytest.raises(ValueError):
        await tools.measure_load({'url':service+'/demo/chat'},demo.generate(12),requests=1000)


def test_full_risk_coverage_missing_is_review():
    config=request('http://localhost')
    config.rules.require_full_coverage=True
    report=tools.generate_report(config.model_dump(),{'cases':demo.generate(12),'dataset_hash':'test'},sample_results())
    assert report['decision']['verdict']=='REVIEW'
    assert report['risk_checks']['latency_under_load']['status']=='not_measured'
    assert report['risk_checks']['accuracy']['status']=='not_measured'
    assert report['risk_checks']['retrieval_recall']['status']=='not_applicable'


def test_retrieval_recall_quote_proxy():
    from greenlit.risks import summarize_risks
    config=request('http://localhost').model_dump()
    config['target']['application_type']='rag'
    case=demo.generate(12)[0]
    case.update(risk_tags=['retrieval_recall'],expected_passages=['closes at 8 PM'])
    row=sample_results()[0]
    row.update(retrieval_observed=True,retrieved_contexts=['The library closes at 8 PM.'])
    report=summarize_risks(config,[case],[row])
    assert report['retrieval_recall']['score']==1
    assert report['retrieval_recall']['status']=='pass'


def test_pdf_reference_upload(service):
    from io import BytesIO
    from pypdf import PdfWriter
    from pypdf.generic import NameObject,DictionaryObject,DecodedStreamObject
    writer=PdfWriter()
    page=writer.add_blank_page(width=600,height=800)
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
    stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 20 750 Td (The college library closes at 8 PM.) Tj ET')
    page[NameObject('/Contents')]=writer._add_object(stream)
    body=BytesIO();writer.write(body)
    with httpx.Client(base_url=service,trust_env=False) as client:
        response=client.post('/api/reference',files={'file':('source.pdf',body.getvalue(),'application/pdf')})
    assert response.status_code==200
    assert '8 PM' in response.json()['text']


async def test_invalid_generation_retries_once(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    count=0
    async def fake(*args,**kwargs):
        nonlocal count
        count+=1
        if count==1:raise ValueError('Generation returned duplicate dataset questions')
        return {'cases':[],'dataset_hash':'hash','source':'test'}
    monkeypatch.setattr(tools,'_generate_test_questions_once',fake)
    result=await tools.generate_test_questions('Purpose text','Reference text',12)
    assert count==2
    assert result['generation_attempts']==2


async def test_generation_network_failure_not_retried(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    count=0
    async def fake(*args,**kwargs):
        nonlocal count
        count+=1
        raise ValueError('GEMINI HTTP 429: quota or rate limit exceeded')
    monkeypatch.setattr(tools,'_generate_test_questions_once',fake)
    with pytest.raises(ValueError,match='429'):
        await tools.generate_test_questions('Purpose text','Reference text',12)
    assert count==1


def test_nested_errors_keep_actionable_message():
    from greenlit.security import safe_error
    error=ExceptionGroup('MCP group',[ExceptionGroup('inner',[ValueError('Generation evidence is missing from reference dataset')])])
    assert safe_error(error)=='Generation evidence is missing from reference dataset'


async def test_unverified_quotes_finish_as_review_not_crash(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    from greenlit.models import GeneratedCases,Judgment
    async def fake(system,payload,schema,judge=False):
        if schema is GeneratedCases:return GeneratedCases(cases=demo.generate(12))
        return Judgment(verdict='pass',correctness=1,relevance=1,faithfulness=1,confidence=1,reason='An otherwise plausible answer.')
    monkeypatch.setattr(tools,'structured',fake)
    dataset=await tools.generate_test_questions('Math teaching assistant.','Be clear and patient. Ask for missing numbers.',12)
    assert dataset['generation_attempts']==2
    assert dataset['cases'][0]['reference_evidence_verified'] is False
    row={'case_id':'case-001','category':'correctness','trial':1,'answer':'8 PM','error':None,'latency_ms':1}
    evaluated=await tools.evaluate_answers('Math teaching assistant.','Be clear and patient.',dataset['cases'],[row])
    assert evaluated['results'][0]['judgment']['verdict']=='review'


async def test_large_generation_uses_bounded_batches(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    batches=[]
    async def fake(purpose,reference,count,plan):
        batches.append(count)
        cases=demo.generate(count)
        for case in cases:case['question']=str(len(batches))+' '+case['question']
        return {'cases':cases,'source':'test','generation_attempts':1}
    monkeypatch.setattr(tools,'_generate_batch',fake)
    result=await tools.generate_test_questions('Purpose text','Reference text',24)
    assert batches==[12,12]
    assert len({c['id'] for c in result['cases']})==24


def test_gemini_failure_is_visible_inside_group():
    from greenlit.security import safe_error
    exc=ExceptionGroup('group',[ValueError('GEMINI returned invalid structured output at cases.4.question')])
    assert 'cases.4.question' in safe_error(exc)


async def test_quota_stops_queued_evaluator_requests(monkeypatch):
    monkeypatch.setenv('MODE','llm')
    calls=0
    async def fail(*args,**kwargs):
        nonlocal calls
        calls+=1
        raise ValueError('GEMINI HTTP 429: quota or rate limit exceeded')
    monkeypatch.setattr(tools,'structured',fail)
    cases=demo.generate(12)
    rows=[{'case_id':c['id'],'category':c['category'],'answer':'answer','error':None,'trial':1,'latency_ms':1} for c in cases]
    result=await tools.evaluate_answers(demo.PURPOSE,demo.REFERENCE,cases,rows)
    assert calls<=3
    assert all(r['judgment'] is None for r in result['results'])


def test_http_quota_message_is_actionable():
    from greenlit.security import safe_error
    request=httpx.Request('POST','http://localhost/chat')
    response=httpx.Response(429,request=request)
    error=httpx.HTTPStatusError('Too many requests',request=request,response=response)
    assert 'quota' in safe_error(error).lower()
