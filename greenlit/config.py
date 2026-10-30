import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def data_dir() -> Path:
    path = Path(os.getenv("GREENLIT_DATA_DIR", str(ROOT / "data"))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def mode() -> str:
    value=os.getenv("MODE", "llm").lower()
    if value == 'demo' and os.getenv('GREENLIT_TEST_FIXTURES') != '1':
        raise ValueError('MODE=demo is disabled. Real evaluations require MODE=llm.')
    if value not in {'llm','demo'}:
        raise ValueError('MODE= must be llm')
    return value


def allowed_hosts() -> set[str]:
    return {h.strip().lower() for h in os.getenv(
        "ALLOWED_TARGET_HOSTS", "127.0.0.1,localhost,::1"
    ).split(",") if h.strip()}


def configure_gemini(key: str, model: str) -> None:
    """Save local settings in place; OneDrive can prevent dotenv's rename operation."""
    import re
    from .llm import validate_model
    validate_model(model)
    if not key.strip() or any(c in key for c in "\r\n'\\"):
        raise ValueError('GEMINI_API_KEY must be a nonempty single-line API key')
    path = ROOT / '.env'
    contents = path.read_text(encoding='utf-8') if path.exists() else ''
    changes = {'MODE': 'llm', 'LLM_PROVIDER': 'gemini', 'GEMINI_MODEL': model,
               'GEMINI_API_KEY': key.strip(), 'JUDGE_MODEL': model}
    for name, value in changes.items():
        pattern = rf'^\s*(?:export\s+)?{name}\s*=.*$'
        line = f"{name}='{value}'"
        contents = re.sub(pattern, lambda match: line, contents, flags=re.MULTILINE) if re.search(pattern, contents, re.MULTILINE) else contents.rstrip() + '\n' + line + '\n'
    path.write_text(contents, encoding='utf-8')
    os.environ.update(changes)
