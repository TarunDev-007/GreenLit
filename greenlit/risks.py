"""Application risk catalog and evidence-aware aggregation."""
QUALITY=['correctness','accuracy','relevance','completeness','instruction_following']
RAG=['context_relevance','retrieval_recall','faithfulness','citation_accuracy']
TOOLS=['tool_selection','parameter_correctness','task_completion','error_recovery']
CONVERSATION=['context_retention','clarification_behavior']
SAFETY=['toxicity','harmful_content','bias','pii_leakage','jailbreak_resistance']
OPERATIONS=['end_to_end_latency','cost_per_request','token_efficiency','error_rate','latency_under_load']
ALL=QUALITY+RAG+TOOLS+CONVERSATION+SAFETY+OPERATIONS


def summarize_risks(request,cases,results,load=None):
    kind=request['target'].get('application_type','chatbot')
    applicable=set(QUALITY+CONVERSATION+SAFETY)
    if kind=='rag': applicable.update(RAG)
    if kind=='agent': applicable.update(TOOLS)
    output={}
    case_map={c['id']:c for c in cases}
    for name in QUALITY+RAG+TOOLS+CONVERSATION+SAFETY:
        if name not in applicable:
            output[name]={'status':'not_applicable','reason':'Not part of the selected application type.'}
            continue
        measured=[]
        for row in results:
            if name not in case_map.get(row['case_id'],{}).get('risk_tags',[]): continue
            entry=(row.get('judgment') or {}).get('risk_metrics',{}).get(name)
            # Trace-based claims need real trace evidence, not a model assertion.
            if name in RAG and not row.get('retrieval_observed'): continue
            if name in TOOLS and not row.get('trace_observed'): continue
            if name in CONVERSATION and len(row.get('conversation',[]))<4: continue
            if name=='citation_accuracy' and not row.get('citations'): continue
            if name=='retrieval_recall':
                quotes=case_map.get(row['case_id'],{}).get('expected_passages',[])
                if not quotes: continue
                context=' '.join(' '.join(row.get('retrieved_contexts',[])).split()).casefold()
                recall=sum(' '.join(q.split()).casefold() in context for q in quotes)/len(quotes)
                entry={'status':'pass' if recall==1 else 'fail','score':recall,
                       'reason':'Generated source-quote recall proxy; not a human-labelled passage recall benchmark.'}
            if not entry or entry['status']=='not_measured': continue
            if name in {'correctness','accuracy'} and not request.get('reference','').strip(): continue
            measured.append({'case_id':row['case_id'],'trial':row['trial'],**entry})
        if not measured:
            output[name]={'status':'not_measured','reason':'No applicable test with sufficient evidence completed.','tested':0}
        else:
            scores=[v['score'] for v in measured if v.get('score') is not None]
            output[name]={'status':'fail' if any(v['status']=='fail' for v in measured) else 'pass',
                          'score':sum(scores)/len(scores) if scores else None,'tested':len(measured),'evidence':measured}
    observations=len(results)
    errors=sum(bool(r.get('error')) for r in results)
    latencies=sorted(r['latency_ms'] for r in results)
    import math
    output['end_to_end_latency']={'status':'measured' if latencies else 'not_measured',
        'p95_ms':latencies[max(0,math.ceil(len(latencies)*.95)-1)] if latencies else None,
        'reason':'Client-observed target latency, including all turns; excludes evaluator latency.'}
    output['error_rate']={'status':'measured' if observations else 'not_measured',
        'value':errors/observations if observations else None,'failures':errors,'requests':observations}
    costs=[r['estimated_cost_usd'] for r in results if r.get('estimated_cost_usd') is not None]
    output['cost_per_request']={'status':'measured' if len(costs)==observations and observations else 'not_measured',
        'average_usd':sum(costs)/len(costs) if len(costs)==observations and costs else None,
        'reason':'Requires target token usage and configured prices; excludes evaluator/provider billing.'}
    usage=[r['usage'] for r in results if r.get('usage') and isinstance(r['usage'].get('prompt_tokens'),int) and isinstance(r['usage'].get('completion_tokens'),int)]
    output['token_efficiency']={'status':'measured' if len(usage)==observations and observations else 'not_measured',
        'mean_tokens_per_response':sum(u['prompt_tokens']+u['completion_tokens'] for u in usage)/len(usage) if usage else None,
        'reason':'Token consumption proxy, not proof of optimal token use. Includes all turns in each response.'}
    output['latency_under_load']=load or {'status':'not_measured','reason':'Bounded load testing was not enabled.'}
    return output
