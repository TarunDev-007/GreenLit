from datetime import datetime, timezone
from importlib.metadata import version
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import storage
from .config import mode
from .llm import model_settings
from .mcp_client import connect_tools
from .models import RunRequest
from .security import validate_target
from .tools import digest
from .regression import incompatibility, snapshot, contextualize, apply_comparison_concern
from .review import initialize


class State(TypedDict, total=False):
    request: dict
    plan: dict
    dataset: dict
    observations: list[dict]
    results: list[dict]
    report: dict


async def run_evaluation(run_id: str, request: RunRequest) -> dict:
    validate_target(request.target.url)
    payload = request.model_dump()
    current_models = {'generator': 'demo fixtures', 'judge': 'demo fixture checks'}
    if mode() == 'llm':
        current_models = {key: {k: v for k, v in model_settings(judge).items() if k != 'api_key'}
                          for key, judge in [('generator', False), ('judge', True)]}
    baseline=None
    baseline_revision=0
    comparison_context={'status':'first_run','selection':'previous','baseline_id':None,
                        'baseline_revision':None,'reason':'No earlier completed run for this application.'}
    saved=None
    if request.baseline_id:
        saved=storage.get(request.baseline_id)
        if not saved or not saved['approved'] or saved['status']!='completed':
            raise ValueError('Baseline must be a completed run marked as an approved baseline')
    elif request.compare_previous:
        saved=storage.previous_run(payload,run_id)
    else:
        comparison_context.update(status='disabled',reason='Automatic comparison was disabled for this run.')
    if saved:
        problem=incompatibility(saved['report'],payload,current_models,mode())
        if problem and request.baseline_id:raise ValueError(problem)
        comparison_context={'status':'not_comparable' if problem else 'compared',
            'selection':'baseline' if request.baseline_id else 'previous','baseline_id':saved['id'],
            'baseline_revision':saved['revision'],'reason':problem or 'Saved questions reused for a fair comparison.'}
        if not problem:
            baseline=saved['report']
            baseline_revision=saved['revision']
    comparison_snapshot=snapshot(baseline,baseline_revision)
    storage.update(run_id, status='running', stage='connecting_tools')
    async with connect_tools() as tools:
        async def plan(state):
            storage.update(run_id, stage='planning_tests')
            if baseline:
                return {'plan': baseline['plan']}
            return {'plan': await tools.call('plan_evaluation', {
                'purpose': request.purpose, 'reference': request.reference})}

        async def generate(state):
            storage.update(run_id, stage='generating_questions')
            if baseline:
                return {'dataset': baseline['dataset']}
            return {'dataset': await tools.call('generate_test_questions', {
                'purpose': request.purpose, 'reference': request.reference, 'count': request.test_count, 'plan': {**state['plan'],'application_type':request.target.application_type}})}

        async def execute(state):
            storage.update(run_id, stage='running_tests')
            return await tools.call('run_chatbot_tests', {'target': payload['target'],
                'cases': state['dataset']['cases'], 'repeats': request.repeats})

        async def evaluate(state):
            storage.update(run_id, stage='evaluating_answers')
            return await tools.call('evaluate_answers', {'purpose': request.purpose, 'reference': request.reference,
                'cases': state['dataset']['cases'], 'observations': state['observations']})

        async def report(state):
            storage.update(run_id, stage='building_report')
            regression = await tools.call('compare_baseline', {'results': state['results'], 'baseline': baseline,
                'dataset_hash': state['dataset']['dataset_hash']})
            load=None
            if request.load_test.enabled:
                storage.update(run_id,stage='checking_load')
                load=await tools.call('measure_load',{'target':payload['target'],'cases':state['dataset']['cases'],
                    'requests':request.load_test.requests,'concurrency':request.load_test.concurrency})
            summary = await tools.call('generate_report' , {'request': payload, 'dataset': state['dataset'],
                'results': state['results'], 'baseline': baseline, 'load':load})
            summary['regression'] = contextualize(regression,comparison_context)
            apply_comparison_concern(summary)
            return {'report': summary}

        graph = StateGraph(State)
        for name, function in [('plan', plan), ('generate', generate), ('execute', execute), ('evaluate', evaluate), ('report', report)]:
            graph.add_node(name, function)
        for a, b in [(START, 'plan'), ('plan', 'generate'), ('generate', 'execute'), ('execute', 'evaluate'), ('evaluate', 'report'), ('report', END)]:
            graph.add_edge(a, b)
        state = await graph.compile().ainvoke({'request': payload})
        models = current_models
        result = {**state['report'], 'id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'plan': state['plan'], 'request': payload, 'dataset': state['dataset'], 'results': state['results'],
                  'mode': mode(), 'models': models, 'config_hash': digest({'request': payload, 'models': models}),
                  'mcp': {'transport': 'stdio', 'tools_discovered': tools.available, 'calls': tools.trace},
                  'versions': {name: version(name) for name in ['mcp', 'langgraph', 'httpx']},
                  'limitations': ['Automated judgments are fallible and not calibrated against human labels.',
                     'Passing this suite is evidence for these tests, not a safety certification.',
                     'Generated questions and their expected behaviours may share model bias.',
                     'RAG evidence and agent traces are supplied by the target API; they are not independently instrumented.',
                     'Target costs require returned token usage and configured prices; evaluator costs are not included.',
                     'Each test request is independent; trace inspection does not verify external side effects or multi-turn state.']}
        initialize(result)
        storage.update(run_id, status='completed', stage='completed', report=result,baseline_snapshot=comparison_snapshot)
        return result
