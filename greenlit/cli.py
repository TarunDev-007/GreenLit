import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Literal


def main():
    parser = argparse.ArgumentParser(description='GreenLit automated chatbot evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve', help='Start the local evaluation dashboard')
    serve.add_argument('--port', type=int, default=8765)
    run = commands.add_parser('run', help='Run a JSON configuration through LangGraph and MCP')
    run.add_argument('config', type=Path)
    run.add_argument('--output', type=Path, default=Path('report.json'))
    configure = commands.add_parser('configure-gemini', help='Enter Gemini key privately and enable AI mode')
    configure.add_argument('--model', default='gemini-3.5-flash-lite')
    commands.add_parser('check-model', help='Make a small live request to verify model credentials')
    commands.add_parser('mcp', help='Run the stdio MCP server')
    args = parser.parse_args()
    if args.command == 'configure-gemini':
        import getpass
        from .config import configure_gemini
        from .llm import validate_model
        try:
            validate_model(args.model)
            key = getpass.getpass('Gemini API key (hidden): ').strip()
            configure_gemini(key, args.model)
        except ValueError as exc:
            parser.exit(1, str(exc) + '\n')
        # Reload the local dashboard without sending the key over HTTP.
        import httpx
        try:
            with httpx.Client(trust_env=False, timeout=3) as client:
                response = client.post('http://127.0.0.1:8765/api/reload-settings')
            if response.status_code == 200:
                print('Gemini configured and dashboard reloaded. Refresh your browser.')
            else:
                print('Gemini configured. Restart the dashboard after active runs finish.')
        except httpx.HTTPError:
            print('Gemini configured. Start GreenLit when ready.')
    elif args.command == 'check-model':
        from pydantic import BaseModel
        from .llm import structured
        from .security import safe_error
        class Probe(BaseModel):
            message: Literal['connected']
        try:
            result = asyncio.run(structured('Return a JSON object with message set to connected.', {}, Probe))
            if result.message != 'connected':
                raise ValueError('Unexpected model probe response')
            print('Model connection verified successfully.')
        except Exception as exc:
            parser.exit(1, safe_error(exc) + '\n')
    elif args.command == 'serve':
        import uvicorn
        uvicorn.run('greenlit.app:app', host='127.0.0.1', port=args.port, workers=1)
    elif args.command == 'mcp':
        from .mcp_server import main as serve_mcp
        serve_mcp()
    else:
        from . import storage
        from .models import RunRequest
        from .security import safe_error
        from .workflow import run_evaluation
        request = RunRequest.model_validate_json(args.config.read_text(encoding='utf-8'))
        run_id = uuid.uuid4().hex
        storage.create(run_id, request.model_dump())
        try:
            report = asyncio.run(run_evaluation(run_id, request))
        except Exception as exc:
            error = safe_error(exc)
            storage.update(run_id, status='failed', stage='failed', error=error)
            parser.exit(1, error + '\n')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Recommendation {report['decision']['verdict']}: report saved to {args.output.resolve()}. Human approval is pending in the dashboard.")
        raise SystemExit({'GO': 2, 'REVIEW': 2, 'NO-GO': 3}[report['decision']['verdict']])


if __name__ == '__main__':
    main()
