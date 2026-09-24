"""
Phase 1a — collect, normalise, de-duplicate and label XSS payloads.

Sources (all public, all fetched at run time so the corpus is reproducible):
  * PayloadsAllTheThings  — filter-bypass, polyglot, WAF-bypass, Angular
  * PayloadsAllTheThings  — Intruders wordlists
  * OWASP Juice Shop      — challenge payloads (added locally, see JUICE_SHOP)

Chapter 3 originally named the XSSed archive; that site has been offline
for several years, so it is replaced here. Record this substitution in the
revised Section 3.3.

Output: data/processed/malicious.csv with columns
  payload, source, category, label
where category in {reflected, stored, dom} and label == 1.
"""

from __future__ import annotations

import csv
import hashlib
import html
import re
import sys
import urllib.parse
from pathlib import Path

from net import fetch as _fetch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "malicious.csv"

BASE = "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/XSS%20Injection"

MARKDOWN_SOURCES = {
    "patt_filter_bypass": f"{BASE}/1%20-%20XSS%20Filter%20Bypass.md",
    "patt_polyglot": f"{BASE}/2%20-%20XSS%20Polyglot.md",
    "patt_waf_bypass": f"{BASE}/3%20-%20XSS%20Common%20WAF%20Bypass.md",
    "patt_angular": f"{BASE}/5%20-%20XSS%20in%20Angular.md",
    "patt_readme": f"{BASE}/README.md",
}

SECLISTS = (
    "https://raw.githubusercontent.com/danielmiessler/SecLists/master"
    "/Fuzzing/XSS/robot-friendly"
)

WORDLIST_SOURCES = {
    "patt_quick": f"{BASE}/Intruders/xss_payloads_quick.txt",
    "patt_polyglots_list": f"{BASE}/Intruders/XSS_Polyglots.txt",
    # PortSwigger XSS cheat sheet — the corpus named in Section 3.3.
    "portswigger": f"{SECLISTS}/XSS-Cheat-Sheet-PortSwigger.txt",
    "brutelogic": f"{SECLISTS}/XSS-BruteLogic.txt",
    "brutelogic_bypass": f"{SECLISTS}/XSS-Bypass-Strings-BruteLogic.txt",
    "rsnake": f"{SECLISTS}/XSS-RSNAKE.txt",
    "somdev": f"{SECLISTS}/XSS-Somdev.txt",
    "jhaddix": (
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master"
        "/Fuzzing/XSS/human-friendly/XSS-Jhaddix.txt"
    ),
}

# Juice Shop challenge payloads. These are the documented solutions to the
# XSS challenges; add any further ones you exercise yourself against a local
# Juice Shop instance and note the challenge name in the comment.
JUICE_SHOP = [
    '<iframe src="javascript:alert(`xss`)">',                    # DOM XSS (search bar)
    '<script>alert("xss")</script>',                             # Bonus / reflected
    '<img src="a" onerror="alert(1)">',                          # Reflected (track order)
    '<<script>Foo</script>iframe src="javascript:alert(`xss`)">', # Sanitiser bypass
]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def fetch(name: str, url: str) -> str:
    """Download once, cache in data/raw, return text."""
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".txt"
    return _fetch(RAW / f"{name}{suffix}", url, timeout=30)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)

# A line only counts as a payload if it looks like injectable script/markup.
LOOKS_INJECTABLE = re.compile(
    r"""(<\s*/?\s*(script|img|svg|iframe|body|input|details|video|audio|object
        |embed|marquee|math|form|a|div|style|link|meta|table|select|textarea)\b)
        | (\bon[a-z]{3,15}\s*=)
        | (javascript\s*:)
        | (\balert\s*\()
        | (\bprompt\s*\()
        | (\bconfirm\s*\()
        | (\beval\s*\()
        | (document\s*\.\s*(cookie|write|location|domain))
        | (\bfetch\s*\()
        | (\{\{.*\}\})
        """,
    re.IGNORECASE | re.VERBOSE,
)

# Lines that are prose, headings, shell commands or URLs, not payloads.
NOISE = re.compile(
    r"""^\s*(\#{1,6}\s
        | [-*]\s+\[
        | >\s
        | \|
        | https?://\S+$
        | \$\s
        | (curl|git|python|pip|npm|node)\b
        )""",
    re.IGNORECASE | re.VERBOSE,
)


def extract_from_markdown(text: str) -> list[str]:
    """Pull payloads out of fenced code blocks and inline-code spans."""
    out: list[str] = []
    for block in FENCE.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if not line or NOISE.match(line):
                continue
            if LOOKS_INJECTABLE.search(line):
                out.append(line)
    # inline `code` spans often hold one-liner payloads
    for span in re.findall(r"`([^`\n]{8,400})`", text):
        span = span.strip()
        if LOOKS_INJECTABLE.search(span) and not NOISE.match(span):
            out.append(span)
    return out


def extract_from_wordlist(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and LOOKS_INJECTABLE.search(line):
            out.append(line)
    return out


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def normalise(payload: str) -> str:
    """
    Encoding normalisation, as promised in Section 3.3.

    Decodes percent-encoding and HTML entities up to two passes (attackers
    routinely double-encode), collapses whitespace, and strips wrapping
    quotes. Deliberately does NOT lowercase: case variation is itself an
    evasion signal the detector should be able to learn from.
    """
    s = payload
    for _ in range(2):
        try:
            unq = urllib.parse.unquote(s)
        except Exception:
            unq = s
        ent = html.unescape(unq)
        if ent == s:
            break
        s = ent
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def canonical_key(payload: str) -> str:
    """
    De-duplication key. Case-folded and stripped of whitespace so that
    trivially-permuted duplicates collapse, while the stored payload keeps
    its original form.
    """
    return hashlib.sha1(
        re.sub(r"\s+", "", payload).lower().encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------
# sub-category labelling
# --------------------------------------------------------------------------

DOM_SINKS = re.compile(
    r"(document\s*\.\s*(write|writeln|cookie|location|domain|URL|referrer)"
    r"|innerHTML|outerHTML|insertAdjacentHTML|\beval\s*\(|setTimeout\s*\("
    r"|setInterval\s*\(|location\s*\.\s*(hash|search|href)"
    r"|window\s*\.\s*name|localStorage|sessionStorage|postMessage)",
    re.IGNORECASE,
)

STORED_HINTS = re.compile(
    r"(<\s*(script|iframe|object|embed|form)\b|srcdoc\s*=|<\s*style\b)",
    re.IGNORECASE,
)


def categorise(payload: str) -> str:
    """
    Heuristic sub-category assignment (reflected / stored / dom).

    NOTE: superseded by synth.py, which assigns category by INJECTION
    CONTEXT rather than by payload string. Retained only so the raw
    payload file carries a rough hint; nothing downstream depends on it.
    """
    if DOM_SINKS.search(payload):
        return "dom"
    if STORED_HINTS.search(payload):
        return "stored"
    return "reflected"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    rows: dict[str, dict] = {}

    for name, url in MARKDOWN_SOURCES.items():
        try:
            payloads = extract_from_markdown(fetch(name, url))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}", file=sys.stderr)
            continue
        print(f"  {name}: {len(payloads)} candidates")
        for p in payloads:
            add(rows, p, name)

    for name, url in WORDLIST_SOURCES.items():
        try:
            payloads = extract_from_wordlist(fetch(name, url))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}", file=sys.stderr)
            continue
        print(f"  {name}: {len(payloads)} candidates")
        for p in payloads:
            add(rows, p, name)

    print(f"  juice_shop: {len(JUICE_SHOP)} candidates")
    for p in JUICE_SHOP:
        add(rows, p, "juice_shop")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["payload", "source", "category", "label"])
        w.writeheader()
        for r in rows.values():
            w.writerow(r)

    dist: dict[str, int] = {}
    for r in rows.values():
        dist[r["category"]] = dist.get(r["category"], 0) + 1

    print(f"\n  unique payloads after de-duplication: {len(rows)}")
    print(f"  category distribution: {dist}")
    print(f"  written to {OUT}")
    return 0


def add(rows: dict, payload: str, source: str) -> None:
    p = normalise(payload)
    if not (8 <= len(p) <= 600):
        return
    key = canonical_key(p)
    if key in rows:
        return
    rows[key] = {
        "payload": p,
        "source": source,
        "category": categorise(p),
        "label": 1,
    }


if __name__ == "__main__":
    raise SystemExit(main())
