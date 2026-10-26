"""Real stdio MCP server. Never print to stdout: it carries protocol messages."""
from mcp.server.fastmcp import FastMCP

from . import tools
from .security import safe_error
from pydantic import ValidationError

server = FastMCP('GreenLit testing tools')


@server.tool()
async def plan_evaluation(purpose: str, reference: str) -> dict:
    """Ask the configured central LLM to identify risks and plan all required test categories."""
    try:
        return await tools.plan_evaluation(purpose,reference)
    except ValueError as exc:
        return {'_greenlit_error':safe_error(exc)}


@server.tool()
async def generate_test_questions(purpose: str, reference: str, count: int = 24, plan: dict | None = None) -> dict:
    """Generate balanced application-specific test cases with reference evidence."""
    try:
        return await tools.generate_test_questions(purpose,reference,count,plan)
    except ValidationError:
        return {'_greenlit_error':'Generation returned invalid test case fields after validation retry.'}
    except ValueError as exc:
        return {'_greenlit_error':safe_error(exc)}


@server.tool()
async def run_chatbot_tests(target: dict, cases: list[dict], repeats: int = 2) -> dict:
    """Call an explicitly allowlisted chatbot endpoint and measure its responses."""
    return await tools.run_chatbot_tests(target, cases, repeats)


@server.tool()
async def evaluate_answers(purpose: str, reference: str, cases: list[dict], observations: list[dict]) -> dict:
    """Evaluate correctness, relevance, groundedness, safety and refusal behaviour."""
    return await tools.evaluate_answers(purpose, reference, cases, observations)


@server.tool()
def compare_baseline(results: list[dict], baseline: dict | None, dataset_hash: str) -> dict:
    """Compare identical dataset cases with a previous approved run."""
    return tools.compare_baseline(results, baseline, dataset_hash)


@server.tool()
def generate_report(request: dict, dataset: dict, results: list[dict], baseline: dict | None = None, load: dict | None = None) -> dict:
    """Calculate operational metrics, repeatability and deterministic release gates."""
    return tools.generate_report(request, dataset, results, baseline, load)


@server.tool()
async def measure_load(target: dict, cases: list[dict], requests: int = 6, concurrency: int = 2) -> dict:
    """Run a user-enabled bounded concurrent latency test, maximum 20 requests / 5 parallel calls."""
    return await tools.measure_load(target,cases,requests,concurrency)


def main():
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
