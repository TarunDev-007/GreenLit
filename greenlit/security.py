import ipaddress
import re
from urllib.parse import urlsplit

from .config import allowed_hosts


def validate_target(url: str) -> str:
    """Operator-managed exact allowlist; no redirects, wildcards or embedded credentials."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Target must be an http(s) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Credentials in URLs and URL fragments are not allowed")
    # Never put API secrets in persisted query strings.
    if parsed.query:
        raise ValueError("Use headers for credentials; target query strings are not supported")
    host = parsed.hostname.lower()
    if host not in allowed_hosts():
        raise ValueError(f"Target host {host!r} is not in ALLOWED_TARGET_HOSTS")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host in {"metadata.google.internal", "instance-data"}:
            raise ValueError("Cloud metadata endpoints are forbidden")
    else:
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Metadata/link-local, multicast and unspecified addresses are forbidden")
    # Fail on invalid port before starting a run.
    _ = parsed.port
    return url


def safe_error(exc: Exception) -> str:
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            message = safe_error(child)
            if not message.endswith('service availability.'):
                return message
    status = getattr(exc, 'status_code', None) or getattr(getattr(exc,'response',None),'status_code',None)
    if status in {401, 403}:
        return 'Model/API authentication failed. Check your key and model access.'
    if status == 429:
        return 'Model/API quota or rate limit reached. Check provider quota and try later.'
    if isinstance(exc,RuntimeError) and str(exc).startswith('MCP tool '):
        return str(exc)[:400]
    # Exceptions from HTTP/LLM SDKs can contain request content or credentials.
    if isinstance(exc,ValueError) and str(exc).startswith(("GEMINI ",)):
        return str(exc)[:400]
    if isinstance(exc, ValueError):
        message = str(exc)
        if any(word in message.lower() for word in ["target", "template", "baseline", "dataset", "mode=", "llm_", "gemini_", "chatbot_", "generation"]):
            return message[:400]
    return f"{type(exc).__name__}: operation failed; check the local configuration and service availability."


def secret_env_name(value: str) -> bool:
    return bool(re.fullmatch(r"CHATBOT_[A-Z0-9_]+", value))
