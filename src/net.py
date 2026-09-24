"""
Shared HTTPS fetch with explicit certificate handling.

Two environments break in opposite ways:

  * macOS Python.org builds do not use the system keychain. Unless
    "Install Certificates.command" has been run, every HTTPS request fails
    with CERTIFICATE_VERIFY_FAILED. certifi's bundle fixes this.
  * Machines behind a TLS-inspecting proxy present a locally-trusted root
    that certifi does not know about. There the system store works and
    certifi fails.

So this tries certifi first and falls back to the system store, rather than
assuming either. Whichever context succeeds is pinned for the rest of the
run.

Verification is never disabled. The one-line ssl._create_unverified_context
"fix" would make both failures disappear, and has no place in the codebase
of a dissertation about web security.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

_SYSTEM = ssl.create_default_context()
try:
    import certifi
    _CERTIFI = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _CERTIFI = None

_CONTEXTS = [c for c in (_CERTIFI, _SYSTEM) if c is not None]
_PINNED: ssl.SSLContext | None = None


def _open(url: str, timeout: int) -> str:
    global _PINNED
    req = urllib.request.Request(url, headers={"User-Agent": "fedxss/1.0"})
    candidates = [_PINNED] if _PINNED else _CONTEXTS
    last = None
    for ctx in candidates:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                _PINNED = ctx
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            if not isinstance(getattr(exc, "reason", None), ssl.SSLError):
                raise
            last = exc
    raise last if last else RuntimeError("no TLS context available")


def fetch(cache_path: Path, url: str, timeout: int = 120) -> str:
    """Download once, cache on disk, return the decoded text."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    text = _open(url, timeout)
    cache_path.write_text(text, encoding="utf-8")
    return text
