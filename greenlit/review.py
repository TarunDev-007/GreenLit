"""Human review overlay. Automated evidence is retained; no model calls are needed."""
from copy import deepcopy

from . import risks
from .models import Judgment
from .regression import contextualize, apply_comparison_concern
from .tools import generate_report


def pending_approval():
    return {'status':'pending','verdict':None,'reviewer':None,'reason':None,'created':None,'revision':None}


def initialize(report):
    report['automated_decision']=deepcopy(report['decision'])
    report['automated_summary']=deepcopy(report['summary'])
    report['review']={'revision':0,'overrides_count':0,'events':[]}
    report['approval']=pending_approval()
    return report


def validate_action(report, action, data):
    if action=='approve':
        if data['verdict']=='GO' and report['decision']['verdict']!='GO':
            raise ValueError('GO approval requires a calculated GO decision. Correct the judgments or resolve the blocking/incomplete checks first.')
        return
    row=next((r for r in report['results'] if r['case_id']==data['case_id'] and r['trial']==data['trial']),None)
    if row is None:raise ValueError('Case/trial not found in this report')
    if action=='reset':return
    if row.get('error') or not row.get('answer'):
        raise ValueError('Cannot override a failed target request. Rerun it to obtain an actual answer.')
    judgment=Judgment.model_validate(data['judgment'])
    old_metrics=set((row.get('judgment') or {}).get('risk_metrics',{})) & set(risks.ALL)
    if not old_metrics<=set(judgment.risk_metrics):
        raise ValueError('Keep all existing risk checks and explicitly review their statuses; do not remove them.')
    if set(judgment.risk_metrics)-set(risks.ALL):raise ValueError('Unknown risk metric in reviewer judgment')
    if judgment.verdict=='pass':
        kind=row.get('application_type',report['request']['target'].get('application_type','chatbot'))
        if kind=='rag' and (not row.get('retrieval_observed') or not judgment.retrieval):
            raise ValueError('A passing RAG judgment requires actual retrieval evidence and a retrieval assessment')
        if kind=='rag' and row['category'] in {'correctness','groundedness','over_refusal'} and (not row.get('retrieved_contexts') or not judgment.retrieval.answer_supported_by_context):
            raise ValueError('A passing grounded answer requires actual supporting retrieved passages')
        if kind=='agent' and (not row.get('trace_observed') or not judgment.agent or judgment.agent.task_success!='pass'):
            raise ValueError('A passing agent judgment requires actual tool traces and confirmed task success')


def rebuild(machine, events, baseline):
    report=deepcopy(machine)
    originals={(r['case_id'],r['trial']):deepcopy(r) for r in machine['results']}
    rows={key:deepcopy(value) for key,value in originals.items()}
    for event in events:
        if event['action'] not in {'override','reset'}:continue
        payload=event['payload'];key=(payload['case_id'],payload['trial'])
        if event['action']=='reset':rows[key]=deepcopy(originals[key]);continue
        row=rows[key]
        row['automated_judgment']=deepcopy(originals[key].get('judgment'))
        row['automated_evaluation_error']=originals[key].get('evaluation_error')
        row['judgment']=deepcopy(payload['judgment'])
        row['evaluation_error']=None
        row['human_override']={k:event[k] for k in ['revision','reviewer','reason','created']}
    results=list(rows.values())
    calculated=generate_report(report['request'],report['dataset'],results,baseline,report.get('load'))
    calculated['regression']=contextualize(calculated['regression'],{
        key:value for key,value in machine['regression'].items()
        if key in {'selection','baseline_id','baseline_revision','reason','status'} and (key!='status' or baseline is None)})
    report.update(calculated,results=results)
    apply_comparison_concern(report)
    report['automated_decision']=deepcopy(machine.get('automated_decision',machine['decision']))
    report['automated_summary']=deepcopy(machine.get('automated_summary',machine['summary']))
    report['review']={'revision':events[-1]['revision'] if events else 0,
        'overrides_count':sum('human_override' in r for r in results),'events':deepcopy(events)}
    report['approval']=pending_approval()
    if events and events[-1]['action']=='approve':
        last=events[-1]
        report['approval']={'status':'approved','verdict':last['payload']['verdict'],
            **{key:last[key] for key in ['revision','created','reviewer','reason']}}
    return report
