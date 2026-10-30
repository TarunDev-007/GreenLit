"""Coordinate Gemini request starts across this installation's evaluator and math API."""
import asyncio
import hashlib
import math
import os
import sqlite3
import time
from contextlib import closing

from .config import data_dir


def interval_seconds() -> float:
    try:
        value = float(os.getenv('GEMINI_REQUEST_INTERVAL_SECONDS', '13'))
    except ValueError as exc:
        raise ValueError('GEMINI_REQUEST_INTERVAL_SECONDS must be between 0 and 60') from exc
    if not math.isfinite(value) or not 0 <= value <= 60:
        raise ValueError('GEMINI_REQUEST_INTERVAL_SECONDS must be between 0 and 60')
    return value


def reserve(settings: dict) -> float:
    interval = interval_seconds()
    if not interval:
        return 0
    # Store only a digest, never credentials. Independent models get independent slots.
    scope = hashlib.sha256((settings['model'] + '\0' + settings['api_key']).encode()).hexdigest()
    with closing(sqlite3.connect(data_dir() / 'gemini-pacing.sqlite3', timeout=5)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS slots (scope TEXT PRIMARY KEY, next_at REAL NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        now = time.time()
        row = db.execute('SELECT next_at FROM slots WHERE scope=?', (scope,)).fetchone()
        start = max(now, row[0] if row else now)
        db.execute('INSERT OR REPLACE INTO slots VALUES (?, ?)', (scope, start + interval))
        db.commit()
    return max(0, start - now)


async def wait_turn(settings: dict) -> None:
    delay = reserve(settings)
    if delay:
        await asyncio.sleep(delay)


def quota_message(response) -> str:
    """Expose only quota classifications and bounded retry time, never raw provider text."""
    daily = False
    zero = False
    retry = None
    try:
        details = response.json().get('error', {}).get('details', [])
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for violation in detail.get('violations', []):
                metric = (str(violation.get('quotaId', '')) + ' ' + str(violation.get('quotaMetric', ''))).lower()
                daily |= any(marker in metric for marker in ['perday', 'per_day', 'per day'])
                zero |= str(violation.get('quotaValue', '')) == '0'
            if str(detail.get('@type', '')).endswith('RetryInfo'):
                value = str(detail.get('retryDelay', ''))
                if value.endswith('s'):
                    seconds = float(value[:-1])
                    if math.isfinite(seconds) and 0 <= seconds <= 86400:
                        retry = math.ceil(seconds)
    except (ValueError, TypeError, AttributeError):
        pass
    prefix = 'GEMINI HTTP 429: '
    if zero:
        return prefix + 'no quota is available for this model/project. Check active limits and model access in Google AI Studio; slowing requests cannot add quota.'
    if daily:
        return prefix + 'daily quota exceeded. Check Google AI Studio and wait for the daily reset or use available project quota; immediate retries will not help.'
    advice = f' Google suggests retrying after {retry} seconds.' if retry is not None else ''
    return prefix + 'quota or rate limit exceeded.' + advice + ' Check active limits in Google AI Studio. Reduce requests and retry only after quota is available.'
