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
