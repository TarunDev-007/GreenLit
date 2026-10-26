import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('MODE', 'demo')
    monkeypatch.setenv('GREENLIT_TEST_FIXTURES','1')
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.setenv('GREENLIT_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('ALLOWED_TARGET_HOSTS', '127.0.0.1,localhost,::1')
    monkeypatch.delenv('TARGET_INPUT_PRICE_PER_MILLION', raising=False)
    monkeypatch.delenv('TARGET_OUTPUT_PRICE_PER_MILLION', raising=False)


@pytest.fixture(scope='session')
def service(tmp_path_factory):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    directory = tmp_path_factory.mktemp('service')
    env = {**os.environ, 'LLM_PROVIDER':'openai', 'MODE': 'demo', 'GREENLIT_TEST_FIXTURES':'1', 'GREENLIT_DATA_DIR': str(directory / 'data'),
           'ALLOWED_TARGET_HOSTS': '127.0.0.1,localhost,::1', 'PYTHONPATH': str(PROJECT)}
    log = (directory / 'server.log').open('w', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'tests.mock_service:app',
                               '--host', '127.0.0.1', '--port', str(port)],
                              cwd=PROJECT, env=env, stdout=log, stderr=log)
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(150):
            if process.poll() is not None:
                raise RuntimeError('Test server failed: ' + (directory / 'server.log').read_text())
            try:
                if httpx.get(url + '/api/settings', timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError('Timed out starting test server')
        yield url
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()
