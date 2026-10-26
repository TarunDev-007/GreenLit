"""Comparable-run selection and immutable comparison inputs."""
from copy import deepcopy


def incompatibility(baseline, request, models, mode):
    previous=baseline['request']
    if 'plan' not in baseline:
        return 'Baseline predates the planning agent; start a new comparison series'
    if previous['purpose']!=request['purpose'] or previous.get('reference','')!=request.get('reference',''):
        return 'Baseline purpose/reference changed; a new dataset is required'
    if baseline.get('models')!=models or baseline.get('mode')!=mode:
        return 'Baseline evaluator settings changed; scores are not directly comparable'
    if previous['test_count']!=request['test_count'] or previous['repeats']!=request['repeats']:
        return 'Baseline question count or trials changed; a new comparison series is required'
    if {k:v for k,v in previous['rules'].items() if k!='max_regression_drop'}!={k:v for k,v in request['rules'].items() if k!='max_regression_drop'} or previous.get('load_test',{})!=request.get('load_test',{}):
        return 'Baseline release criteria or load profile changed'
    for key in ['application_type','contexts_path','tool_trace_path','citations_path','conversation_request_template']:
        default='chatbot' if key=='application_type' else None if key=='conversation_request_template' else {'contexts_path':'retrieved_contexts','tool_trace_path':'tool_calls','citations_path':'citations'}[key]
        if previous['target'].get(key,default)!=request['target'].get(key,default):
            return 'Baseline application type or evidence mapping changed'
    return None


def snapshot(report, revision=0):
    if report is None:return None
    return deepcopy({'id':report.get('id'),'revision':revision,
        'dataset':{'dataset_hash':report['dataset']['dataset_hash']},
        'summary':report['summary'],'request':{'rules':report['request']['rules']},
        'results':[{'case_id':r['case_id'],'trial':r['trial'],'judgment':r.get('judgment')} for r in report['results']]})


def contextualize(comparison, context):
    result={**comparison,**context}
    if comparison['status']=='compared':result['status']='compared'
    return result


def apply_comparison_concern(report):
    if report['regression']['status']=='not_comparable':
        reason='Previous run cannot be fairly compared: '+report['regression']['reason']
        report['decision']['concerns'].append(reason)
        if report['decision']['verdict']=='GO':report['decision']['verdict']='REVIEW'
        report['decision']['reasons']=report['decision']['blockers']+report['decision']['concerns']
