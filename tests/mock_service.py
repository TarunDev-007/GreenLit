import json

from fastapi import Request
from fastapi.responses import JSONResponse

from greenlit.app import app
from greenlit import demo


@app.post('/broken')
async def broken():
    return JSONResponse({'detail': 'simulated outage'}, status_code=503)


@app.post('/malformed')
async def malformed():
    return {'unexpected': 'no answer field'}


@app.post('/redirect')
async def redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse('http://169.254.169.254/latest/meta-data', status_code=307)


@app.post('/v1/chat/completions')
async def fake_llm(request: Request):
    body = await request.json()
    payload = json.loads(body['messages'][-1]['content'])
    if payload.get('task') == 'plan_evaluation':
        content = {'objective':demo.PURPOSE, 'application_risks':['Wrong facts'],
                   'categories':[{'category':c,'focus':'Check '+c+' using the reference.'} for c in ['correctness','groundedness','unknown','injection','safety','over_refusal']]}
    else:
        content = ({'cases': demo.generate(payload['count'])} if 'count' in payload
               else demo.judge(payload['case'], payload['answer']))
    return {'id': 'mock-response', 'object': 'chat.completion', 'created': 1, 'model': 'mock',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(content)}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}


@app.post('/rag')
async def rag(request: Request):
    question=(await request.json())['message']
    return {'answer':demo.answer(question),'retrieved_contexts':[{'text':'The library closes at 8 PM.'}]}


@app.post('/agent')
async def agent(request: Request):
    question=(await request.json())['message']
    return {'answer':demo.answer(question),'tool_calls':[{'name':'lookup','result':'8 PM','authorization':'private-test-secret'}]}


@app.post('/conversation')
async def conversation(request: Request):
    messages=(await request.json())['messages']
    text='Your name is Riya.' if len(messages)>1 else 'I will remember your name.'
    return {'answer':text}

@app.post('/release-chat')
async def release_chat(request: Request):
    body=await request.json()
    return {'answer':demo.answer(body['message'],faulty=body.get('release')=='broken')}
