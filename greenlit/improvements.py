"""Evidence-linked suggestions; no additional LLM calls or automatic prompt changes."""
from collections import defaultdict

GUIDANCE={
 'correctness':('Ask the target to check calculations/facts against approved sources before giving a final answer.','Validate with labelled examples; consider a stronger reasoning model only if failures persist.'),
 'groundedness':('Require answers to use retrieved source passages and to acknowledge missing support; request source citations.','Check chunking, retrieval ranking and document freshness before changing the generation model.'),
 'unknown':('Tell the target to ask for missing inputs and state when information is unavailable instead of guessing.','Use a model that follows uncertainty instructions; verify using unanswerable questions.'),
 'injection':('Keep trusted system rules separate from user/doc content. Treat instructions inside retrieved text as untrusted and never disclose private prompts or records.','Evaluate an instruction-following model on held-out injection cases; a model change alone is not a security fix.'),
 'safety':('State the unsafe actions the target must refuse and offer a safe alternative. Keep tool permissions and private data access restricted.','Evaluate the model’s safety behavior with independent labelled tests before replacing it.'),
 'over_refusal':('Distinguish harmless in-scope requests from unsafe requests; instruct the target to answer permitted questions directly.','If a safety-tuned model keeps refusing benign requests, test another configuration against both benign and unsafe cases.')}


def build_suggestions(cases, results):
    lookup={c['id']:c for c in cases}
    groups=defaultdict(list)
    for row in results:
        j=row.get('judgment')
        if row.get('error') or row.get('evaluation_error'):
            groups['connection'].append(row)
        elif j and (j['verdict']!='pass' or any(m['status']=='fail' for m in j.get('risk_metrics',{}).values())):
            groups[row['category']].append(row)
    suggestions=[]
    for category,rows in groups.items():
        if category=='connection':
            prompt='Check the target endpoint, response field, timeout and model service. Rerun the failed requests before changing the application prompt.'
            model='Check local model availability and resources; an evaluation error is not evidence that a different target model is needed.'
        else:prompt,model=GUIDANCE[category]
        evidence=[{'case_id':r['case_id'],'trial':r['trial'],
            'finding':(r.get('judgment') or {}).get('reason') or r.get('error') or r.get('evaluation_error'),
            'expected':lookup.get(r['case_id'],{}).get('expected_behavior','')} for r in rows]
        suggestions.append({'id':'improve-'+category,'category':category,
            'case_ids':list(dict.fromkeys(r['case_id'] for r in rows)), 'evidence':evidence,
            'prompt_change':prompt,'model_or_system_change':model,
            'verify':'Review the expected answers, apply a targeted change, then rerun this saved dataset and compare scores.',
            'source':'Evidence-linked guidance; not an LLM-generated or verified fix.'})
    return suggestions
