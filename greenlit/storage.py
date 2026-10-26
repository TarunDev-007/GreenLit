import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import data_dir


class RevisionConflict(ValueError):
    pass


def connect():
    connection=sqlite3.connect(data_dir()/'greenlit.sqlite3',timeout=30)
    connection.row_factory=sqlite3.Row
    try:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('''CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, created TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, request TEXT NOT NULL, report TEXT, error TEXT,
            approved INTEGER NOT NULL DEFAULT 0)''')
        columns={row['name'] for row in connection.execute('PRAGMA table_info(runs)')}
        for name,definition in [('revision','INTEGER NOT NULL DEFAULT 0'),('machine_report','TEXT'),('baseline_snapshot','TEXT')]:
            if name not in columns:connection.execute(f'ALTER TABLE runs ADD COLUMN {name} {definition}')
        connection.execute('''CREATE TABLE IF NOT EXISTS review_events (
            run_id TEXT NOT NULL, revision INTEGER NOT NULL, created TEXT NOT NULL,
            action TEXT NOT NULL, reviewer TEXT NOT NULL, reason TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(run_id,revision))''')
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


@contextmanager
def database():
    db=connect()
    try:
        with db:yield db
    finally:db.close()


def decode(row):
    if row is None:return None
    result=dict(row)
    for field in ['request','report','machine_report','baseline_snapshot']:
        result[field]=json.loads(result[field]) if result.get(field) else None
    return result


def create(run_id, request):
    with database() as db:
        db.execute('INSERT INTO runs(id,created,status,stage,request) VALUES(?,?,?,?,?)',
            (run_id,datetime.now(timezone.utc).isoformat(),'queued','queued',json.dumps(request)))


def update(run_id, **fields):
    allowed={'status','stage','report','error','approved','baseline_snapshot'}
    if not fields or not fields.keys()<=allowed:raise ValueError('Invalid update')
    original=fields.get('report')
    for name in ['report','baseline_snapshot']:
        if name in fields:fields[name]=json.dumps(fields[name],ensure_ascii=False)
    with database() as db:
        assignments=','.join(name+'=?' for name in fields)
        values=list(fields.values())
        if original is not None:
            assignments+=',machine_report=COALESCE(machine_report,?)'
            values.append(json.dumps(original,ensure_ascii=False))
        db.execute(f'UPDATE runs SET {assignments} WHERE id=?',[*values,run_id])


def get(run_id):
    with database() as db:
        return decode(db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone())


def previous_run(request, run_id):
    with database() as db:
        current=db.execute('SELECT created FROM runs WHERE id=?',(run_id,)).fetchone()
        if not current:return None
        rows=db.execute("SELECT * FROM runs WHERE status='completed' AND id!=? AND created<=? ORDER BY created DESC,rowid DESC",(run_id,current['created']))
        for row in rows:
            previous=decode(row)
            target=previous['request']['target']
            if target['url']==request['target']['url'] and target.get('application_type','chatbot')==request['target'].get('application_type','chatbot'):
                return previous
    return None


def history():
    with database() as db:
        rows=db.execute('SELECT id,created,status,stage,approved,request,report,revision FROM runs ORDER BY created DESC,rowid DESC LIMIT 100').fetchall()
    history=[]
    for row in rows:
        report=json.loads(row['report']) if row['report'] else None
        history.append({'id':row['id'],'created':row['created'],'status':row['status'],'stage':row['stage'],
            'approved':bool(row['approved']),'name':json.loads(row['request'])['name'],'revision':row['revision'],
            'verdict':report['decision']['verdict'] if report else None,
            'approval_status':(report or {}).get('approval',{}).get('status','pending'),
            'final_verdict':(report or {}).get('approval',{}).get('verdict')})
    return history


def append_review(run_id, action, data):
    from .review import validate_action, rebuild
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        row=decode(db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone())
        if row is None:raise LookupError('Evaluation not found')
        if row['status']!='completed' or not row['report']:raise ValueError('Only completed evaluations can be reviewed')
        if data['revision']!=row['revision']:raise RevisionConflict('This report changed. Reload it before reviewing again.')
        if row['report'].get('regression',{}).get('status')=='compared' and not row['baseline_snapshot']:
            raise ValueError('Legacy comparison report has no frozen baseline. Run a fresh evaluation before reviewing it.')
        validate_action(row['report'],action,data)
        revision=row['revision']+1
        event={'revision':revision,'created':datetime.now(timezone.utc).isoformat(),'action':action,
            'reviewer':data['reviewer'],'reason':data['reason'],
            'payload':{key:value for key,value in data.items() if key not in {'revision','reviewer','reason'}}}
        if action=='override':
            before=next(r for r in row['report']['results'] if r['case_id']==data['case_id'] and r['trial']==data['trial'])
            event['payload']['previous_judgment']=before.get('judgment')
        events=[]
        for saved in db.execute('SELECT * FROM review_events WHERE run_id=? ORDER BY revision',(run_id,)):
            item=dict(saved);item['payload']=json.loads(item['payload']);item.pop('run_id');events.append(item)
        events.append(event)
        machine=row['machine_report'] or row['report']
        report=rebuild(machine,events,row['baseline_snapshot'])
        db.execute('INSERT INTO review_events(run_id,revision,created,action,reviewer,reason,payload) VALUES(?,?,?,?,?,?,?)',
            (run_id,revision,event['created'],action,event['reviewer'],event['reason'],json.dumps(event['payload'],ensure_ascii=False)))
        db.execute('UPDATE runs SET report=?,machine_report=COALESCE(machine_report,?),revision=?,approved=0 WHERE id=?',
            (json.dumps(report,ensure_ascii=False),json.dumps(machine,ensure_ascii=False),revision,run_id))
    return {'revision':revision,'report':report}


def approve_baseline(run_id):
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        row=decode(db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone())
        if row is None:raise LookupError('Evaluation not found')
        report=row.get('report') or {}
        approval=report.get('approval',{})
        if row['status']!='completed' or report.get('decision',{}).get('verdict')!='GO' or approval.get('status')!='approved' or approval.get('verdict')!='GO':
            raise ValueError('A completed GO report needs human GO approval before it can become a baseline')
        db.execute('UPDATE runs SET approved=1 WHERE id=?',(run_id,))


def recover_interrupted():
    with database() as db:
        db.execute("UPDATE runs SET status='failed',stage='interrupted',error='Application stopped during this run. Start a new evaluation.' WHERE status IN ('queued','running')")
