import json

def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False)

def markdown(report: dict) -> str:
    decision, summary = report['decision'], report['summary']
    lines = [f"# GreenLit: {report['request']['name']}", '', f"Calculated recommendation: **{decision['verdict']}**", '',
             f"Mode: {report['mode']} | Run: {report['id']}", f"Created: {report['created_at']}", '',
             '## Decision evidence', *['- ' + reason for reason in decision['reasons']], '',
             '## Metrics', f"- Responses: {summary['total']}", f"- Errors: {summary['errors']}",
             f"- Quality score: {summary['score']}", f"- Pass rate: {summary['pass_rate']}",
             f"- p95 latency: {summary['p95_ms']} ms", f"- Safety/injection gate failures: {summary['hard_failures']}",
             f"- Unstable cases: {len(summary['unstable_cases'])}",
             f"- Target cost estimate (USD): {summary['estimated_target_cost_usd'] if summary['estimated_target_cost_usd'] is not None else 'Unavailable'}", '',
             '## Reproducibility', f"Dataset SHA-256: {report['dataset']['dataset_hash']}",
             f"Configuration SHA-256: {report['config_hash']}", f"Dataset source: {report['dataset']['source']}",
             f"Regression: {report['regression']}", '', '## Risk coverage', json_text(report.get('risk_checks',{})), '', '## AI test plan', json_text(report.get('plan', {})), '', '## Test evidence']
    approval=report.get('approval',{})
    lines.extend(['', '## Human approval', 'Status: '+approval.get('status','pending'),
        'Final verdict: '+str(approval.get('verdict') or 'Awaiting approval'),
        'Reviewer: '+str(approval.get('reviewer') or ''),'Reason: '+str(approval.get('reason') or ''),
        '', '## Improvement suggestions'])
    for suggestion in report.get('improvements',[]):
        lines.extend(['', '### '+suggestion['category'], 'Cases: '+', '.join(suggestion['case_ids']),
            'Prompt: '+suggestion['prompt_change'],'Model/system: '+suggestion['model_or_system_change'],
            'Verify: '+suggestion['verify'],'Source: '+suggestion['source']])
    lines.extend(['', '## Original automated recommendation',json_text(report.get('automated_decision',decision)),
        '', '## Review audit history',json_text(report.get('review',{}))])
    lookup = {case['id']: case for case in report['dataset']['cases']}
    for row in report['results']:
        judgment = row.get('judgment')
        lines.extend(['', f"### {row['case_id']} / trial {row['trial']} / {row['category']}",
                      'Question: ' + lookup[row['case_id']]['question'], '',
                      'Expected: ' + lookup[row['case_id']]['expected_behavior'], '',
                      'Answer: ' + row['answer'], '',
                      'Result: ' + (judgment['verdict'] + ' — ' + judgment['reason'] if judgment else 'ERROR — ' + str(row.get('error') or row.get('evaluation_error')))])
        if row.get('human_override'):
            lines.extend(['Reviewer override: '+json_text(row['human_override']),
                'Original automated judgment: '+json_text(row.get('automated_judgment'))])
    lines.extend(['', '## Limitations', *['- ' + item for item in report['limitations']]])
    return '\n'.join(lines)
