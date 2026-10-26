import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone

import pytest
from fastapi.testclient import TestClient

from greenlit import storage,tools,review,demo
from greenlit.app import app
from greenlit.models import OverrideRequest,ApprovalRequest
from greenlit.regression import snapshot
from tests.test_greenlit import request,execute,sample_results


def saved(rows=None,baseline=None,load=None):
    config=request('http://localhost').model_dump()
    rows=copy.deepcopy(rows if rows is not None else sample_results())
    dataset={'cases':demo.generate(12),'dataset_hash':tools.digest(demo.generate(12)),'source':'test fixtures'}
    report=tools.generate_report(config,dataset,rows,baseline,load)
    report.update(id=uuid.uuid4().hex,created_at=datetime.now(timezone.utc).isoformat(),request=config,
        dataset=dataset,results=rows,mode='demo',models={'generator':'demo fixtures','judge':'demo fixture checks'},
        plan={'objective':'Fixture plan','categories':[]},mcp={'calls':[]},limitations=[],config_hash='fixture-hash')
    review.initialize(report)
    storage.create(report['id'],config)
    storage.update(report['id'],status='completed',stage='completed',report=report,baseline_snapshot=baseline)
    return report


def identity(revision=0):return {'revision':revision,'reviewer':'College reviewer','reason':'Checked against the approved reference.'}


def override(report,index,judgment,revision=0):
    row=report['results'][index]
    return OverrideRequest(**identity(revision),case_id=row['case_id'],trial=row['trial'],judgment=judgment).model_dump()


def test_any_score_drop_is_no_go_even_below_old_tolerance():
    baseline=saved()
    rows=sample_results()
    rows[0]['judgment']['correctness']=.99
    calculated=tools.generate_report(baseline['request'],baseline['dataset'],rows,snapshot(baseline))
    assert 0<calculated['regression']['score_drop']<.001
    assert calculated['decision']['verdict']=='NO-GO'
    assert any('declined' in r for r in calculated['decision']['blockers'])


async def test_automatically_compare_latest_unapproved_run_and_reuse_questions(service):
    config=request(service)
    config.target.url=service+'/release-chat'
    config.target.request_template={'message':'{{question}}','release':'good'}
    previous=await execute(config)
    assert previous['regression']['status']=='first_run'
    assert previous['approval']['status']=='pending'
    assert not storage.get(previous['id'])['approved']
    config.target.request_template['release']='broken'
    current=await execute(config)
    assert current['regression']['status']=='compared'
    assert current['regression']['baseline_id']==previous['id']
    assert current['regression']['score_drop']>0
    assert current['decision']['verdict']=='NO-GO'
    assert current['dataset']==previous['dataset']
    assert not any(c['tool']=='generate_test_questions' for c in current['mcp']['calls'])
    assert storage.get(current['id'])['baseline_snapshot']['id']==previous['id']


async def test_changed_count_starts_new_series_and_explains_incomparability(service):
    config=request(service)
    config.target.url=service+'/release-chat'
    previous=await execute(config)
    config.test_count=18
    current=await execute(config)
    assert current['regression']['status']=='not_comparable'
    assert current['regression']['baseline_id']==previous['id']
    assert len(current['dataset']['cases'])==18
    assert current['decision']['verdict']=='REVIEW'


def test_previous_selection_is_scoped_to_application():
    first=saved()
    config=request('http://localhost').model_dump()
    config['target']['url']='http://another-local-app/chat'
    other_id=uuid.uuid4().hex
    other=copy.deepcopy(first);other.update(id=other_id,request=config)
    storage.create(other_id,config);storage.update(other_id,status='completed',report=other)
    current_id=uuid.uuid4().hex
    storage.create(current_id,first['request'])
    selected=storage.previous_run(first['request'],current_id)
    assert selected['id']==first['id']



def test_override_recomputes_scores_preserves_ai_and_invalidates_approval():
    original=sample_results()
    original[16]['judgment'].update(verdict='fail',correctness=0,confidence=1,reason='The AI judge marked a safety failure.')
    report=saved(original)
    assert report['decision']['verdict']=='NO-GO'
    # A reviewer first accepts the NO-GO recommendation.
    storage.append_review(report['id'],'approve',{**identity(),'verdict':'NO-GO'})
    corrected=sample_results()[16]['judgment']
    result=storage.append_review(report['id'],'override',override(report,16,corrected,1))
    edited=result['report']
    assert edited['decision']['verdict']=='GO' and edited['summary']['score']==1
    assert edited['approval']['status']=='pending'
    assert edited['automated_decision']['verdict']=='NO-GO'
    assert edited['results'][16]['automated_judgment']['verdict']=='fail'
    assert edited['review']['overrides_count']==1
    assert len(edited['review']['events'])==2
    stored=storage.get(report['id'])
    assert stored['machine_report']['results'][16]['judgment']['verdict']=='fail'
    assert not stored['approved']


def test_restore_ai_judgment_keeps_audit_and_resets_approval():
    report=saved()
    bad=copy.deepcopy(report['results'][0]['judgment']);bad.update(verdict='fail',correctness=0)
    storage.append_review(report['id'],'override',override(report,0,bad))
    storage.append_review(report['id'],'approve',{**identity(1),'verdict':'NO-GO'})
    result=storage.append_review(report['id'],'reset',{**identity(2),'case_id':report['results'][0]['case_id'],'trial':1})['report']
    assert result['results'][0]['judgment']==report['results'][0]['judgment']
    assert result['review']['overrides_count']==0 and len(result['review']['events'])==3
    assert result['approval']['status']=='pending'


def test_no_go_cannot_be_approved_go_and_human_can_hold_a_go():
    rows=sample_results();rows[16]['judgment']['verdict']='fail'
    report=saved(rows)
    with pytest.raises(ValueError,match='GO approval requires'):
        storage.append_review(report['id'],'approve',{**identity(),'verdict':'GO'})
    good=saved()
    held=storage.append_review(good['id'],'approve',{**identity(),'verdict':'NO-GO'})['report']
    assert held['decision']['verdict']=='GO'
    assert held['approval']['verdict']=='NO-GO'
    with pytest.raises(ValueError,match='human GO approval'):storage.approve_baseline(good['id'])


def test_target_errors_cannot_be_overridden_into_success():
    rows=sample_results();rows[0].update(answer='',error='Connection failed',judgment=None)
    report=saved(rows)
    with pytest.raises(ValueError,match='failed target request'):
        storage.append_review(report['id'],'override',override(report,0,sample_results()[0]['judgment']))
    assert storage.get(report['id'])['revision']==0


def test_evaluator_error_can_be_graded_by_a_human_with_an_actual_answer():
    rows=sample_results();rows[0].update(judgment=None,evaluation_error='Judge unavailable')
    report=saved(rows)
    edited=storage.append_review(report['id'],'override',override(report,0,sample_results()[0]['judgment']))['report']
    assert edited['results'][0]['evaluation_error'] is None
    assert edited['results'][0]['automated_evaluation_error']=='Judge unavailable'
    assert edited['summary']['errors']==0


def test_stale_concurrent_approvals_have_one_atomic_winner():
    report=saved()
    def approve(_):
        try:return storage.append_review(report['id'],'approve',{**identity(),'verdict':'GO'})['revision']
        except storage.RevisionConflict:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(approve,range(2)))
    assert sorted(map(str,results))==['1','conflict']
    stored=storage.get(report['id'])
    assert stored['revision']==1 and len(stored['report']['review']['events'])==1


def test_comparison_snapshot_does_not_change_when_previous_report_is_reviewed():
    previous=saved()
    frozen=snapshot(previous)
    current=saved(baseline=frozen)
    bad=copy.deepcopy(previous['results'][0]['judgment']);bad.update(verdict='fail',correctness=0)
    storage.append_review(previous['id'],'override',override(previous,0,bad))
    almost=copy.deepcopy(current['results'][0]['judgment']);almost['correctness']=.99
    edited=storage.append_review(current['id'],'override',override(current,0,almost))['report']
    assert edited['regression']['previous_score']==1
    assert edited['regression']['score_drop']>0
    assert edited['decision']['verdict']=='NO-GO'


def test_failed_cases_get_evidence_linked_prompt_and_model_suggestions():
    rows=sample_results();rows[16]['judgment'].update(verdict='fail',reason='Unsafe answer was returned.')
    report=saved(rows)
    suggestion=next(s for s in report['improvements'] if s['category']=='safety')
    assert rows[16]['case_id'] in suggestion['case_ids']
    assert suggestion['prompt_change'] and suggestion['model_or_system_change']
    assert suggestion['evidence'][0]['finding']=='Unsafe answer was returned.'
    assert 'not an LLM' in suggestion['source']


def test_review_api_validation_approval_and_downloads():
    report=saved();run_id=report['id']
    with TestClient(app) as client:
        assert client.post('/api/runs/'+run_id+'/baseline').status_code==409
        assert client.post('/api/runs/'+run_id+'/approval',json={**identity(),'reason':' ','verdict':'GO'}).status_code==422
        response=client.post('/api/runs/'+run_id+'/approval',json={**identity(),'verdict':'GO'})
        assert response.status_code==200 and response.json()['report']['approval']['verdict']=='GO'
        assert client.post('/api/runs/'+run_id+'/baseline').status_code==200
        assert client.post('/api/runs/'+run_id+'/approval',json={**identity(),'verdict':'GO'}).status_code==409
        exported=client.get('/api/runs/'+run_id+'/report.json').json()
        assert exported['approval']['reviewer']=='College reviewer'
        markdown=client.get('/api/runs/'+run_id+'/report.md').text
        assert '## Human approval' in markdown and '## Review audit history' in markdown
        assert client.post('/api/runs/'+run_id+'/approval',json={**identity(1),'verdict':'NO-GO'},headers={'Origin':'https://untrusted.example'}).status_code==403


def test_override_cannot_clear_observed_load_latency_failure():
    report=saved(load={'status':'measured','p95_ms':10000,'failure_rate':0})
    edited=storage.append_review(report['id'],'override',override(report,0,report['results'][0]['judgment']))['report']
    assert edited['decision']['verdict']=='NO-GO'
    assert edited['load']['p95_ms']==10000
    assert any('Latency under load' in r for r in edited['decision']['blockers'])


def test_legacy_database_migration_preserves_saved_records():
    import sqlite3,json
    from greenlit.config import data_dir
    path=data_dir()/'greenlit.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE runs (id TEXT PRIMARY KEY,created TEXT,status TEXT,stage TEXT,request TEXT,report TEXT,error TEXT,approved INTEGER DEFAULT 0)')
        config=request('http://localhost').model_dump()
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',('legacy-id','2026-10-23','failed','failed',json.dumps(config),None,'Old error',1))
    migrated=storage.get('legacy-id')
    assert migrated['request']==config and migrated['error']=='Old error' and migrated['approved']==1
    assert migrated['revision']==0 and migrated['machine_report'] is None


def test_missing_rag_evidence_cannot_be_overridden_into_pass():
    rows=sample_results();rows[0].update(application_type='rag',retrieval_observed=False,retrieved_contexts=[])
    report=saved(rows)
    proposed=copy.deepcopy(rows[0]['judgment'])
    proposed['retrieval']={'context_relevance':1,'answer_supported_by_context':True,'reason':'Reviewer assertion without trace.'}
    with pytest.raises(ValueError,match='actual retrieval evidence'):
        storage.append_review(report['id'],'override',override(report,0,proposed))


def test_legacy_compared_report_requires_a_fresh_frozen_comparison():
    report=saved()
    with storage.database() as db:
        report['regression'].update(status='compared',score_drop=.1)
        import json
        db.execute('UPDATE runs SET report=?,machine_report=NULL WHERE id=?',(json.dumps(report),report['id']))
    with pytest.raises(ValueError,match='Legacy comparison report'):
        storage.append_review(report['id'],'approve',{**identity(),'verdict':'GO'})
