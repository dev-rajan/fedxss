"""
Phase 2a — fixed-length feature vector from one observation record.

Design constraint (Section 3.2): this must run inside the browser on every
navigation and DOM mutation. So the extractor is STATELESS — no fitted
vocabulary, no IDF table, nothing to ship or keep in sync with the server.
Character n-grams go through the hashing trick, which is a few lines of
arithmetic and ports to JavaScript directly.

A TF-IDF vectoriser would score slightly better here but would require
shipping and versioning a vocabulary to every client, and a client whose
vocabulary drifted from the server's would silently produce garbage
features. Record this trade-off in Section 3.4 — it is a deployability
argument, not a modelling one.

Vector layout (DIM = 320):
    0                     hand-crafted URL features      (18)
    18                    hand-crafted script features   (16)
    34                    sink one-hot                   (10)
    44                    origin one-hot                  (2)
    46 .. 46+N_HASH       hashed char 3-grams           (274)
"""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import Counter

import numpy as np

from schema import SINKS

N_HASH = 274
OFFSET_HASH = 46
DIM = OFFSET_HASH + N_HASH  # 320

DANGEROUS_TAGS = re.compile(
    r"<\s*/?\s*(script|img|svg|iframe|body|input|details|video|audio|object"
    r"|embed|marquee|math|form|style|link|meta|base|applet|isindex)\b",
    re.IGNORECASE,
)
EVENT_HANDLER = re.compile(r"\bon[a-z]{3,15}\s*=", re.IGNORECASE)
JS_URI = re.compile(r"javascript\s*:", re.IGNORECASE)
SINK_CALL = re.compile(
    r"(innerHTML|outerHTML|document\s*\.\s*write|insertAdjacentHTML"
    r"|\beval\s*\(|setAttribute|\bFunction\s*\()",
    re.IGNORECASE,
)
SOURCE_READ = re.compile(
    r"(location\s*\.\s*(hash|search|href)|window\s*\.\s*name"
    r"|URLSearchParams|document\s*\.\s*(referrer|cookie|URL))",
    re.IGNORECASE,
)
ALERTISH = re.compile(r"\b(alert|prompt|confirm|atob|fromCharCode)\s*\(", re.IGNORECASE)


def shannon(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def ratio(pred, s: str) -> float:
    return (sum(1 for ch in s if pred(ch)) / len(s)) if s else 0.0


def url_features(url: str) -> list[float]:
    try:
        p = urllib.parse.urlparse(url)
    except ValueError:
        p = urllib.parse.urlparse("")
    query = p.query or ""
    frag = p.fragment or ""
    payload_area = query + frag
    decoded = urllib.parse.unquote(payload_area)
    params = urllib.parse.parse_qsl(query, keep_blank_values=True)
    values = [v for _, v in params]
    longest = max((len(v) for v in values), default=0)

    return [
        min(len(url) / 200.0, 3.0),
        min(len(query) / 100.0, 3.0),
        min(len(frag) / 100.0, 3.0),
        min(len(params) / 10.0, 3.0),
        min(longest / 100.0, 3.0),
        shannon(payload_area) / 6.0,
        ratio(str.isdigit, url),
        ratio(lambda c: not c.isalnum(), url),
        min(url.count("%") / 20.0, 3.0),
        min(url.count("%25") / 5.0, 3.0),          # double-encoding marker
        min(url.count("/") / 10.0, 3.0),
        float("<" in decoded),
        float(">" in decoded),
        float(bool(DANGEROUS_TAGS.search(decoded))),
        float(bool(EVENT_HANDLER.search(decoded))),
        float(bool(JS_URI.search(decoded))),
        float(p.scheme == "https"),
        float(bool(frag)),
    ]


def script_features(script: str) -> list[float]:
    s = script or ""
    return [
        min(len(s) / 200.0, 3.0),
        shannon(s) / 6.0,
        ratio(lambda c: not c.isalnum() and not c.isspace(), s),
        ratio(str.isupper, s),
        min(len(DANGEROUS_TAGS.findall(s)) / 5.0, 3.0),
        min(len(EVENT_HANDLER.findall(s)) / 5.0, 3.0),
        float(bool(JS_URI.search(s))),
        float(bool(SINK_CALL.search(s))),
        float(bool(SOURCE_READ.search(s))),
        # source AND sink in the same script is the DOM-XSS signature
        float(bool(SOURCE_READ.search(s)) and bool(SINK_CALL.search(s))),
        float(bool(ALERTISH.search(s))),
        min(s.count("(") / 10.0, 3.0),
        min(s.count("`") / 5.0, 3.0),
        min(s.count("&#") / 5.0, 3.0),             # entity encoding
        min(s.count("\\x") / 5.0, 3.0),            # hex escaping
        float("sanitize" in s.lower() or "escapehtml" in s.lower()),
    ]


def hashed_ngrams(text: str, n: int = 3) -> np.ndarray:
    """Stateless char n-gram hashing. Signed hashing reduces collision bias."""
    vec = np.zeros(N_HASH, dtype=np.float32)
    if not text:
        return vec
    t = text[:600]
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        h = 2166136261
        for ch in g:                                # FNV-1a, portable to JS
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        vec[h % N_HASH] += 1.0 if (h >> 31) & 1 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def featurise(record: dict) -> np.ndarray:
    url = record.get("url", "") or ""
    script = record.get("script", "") or ""
    v = np.zeros(DIM, dtype=np.float32)
    v[0:18] = url_features(url)
    v[18:34] = script_features(script)

    sink = record.get("sink", "none") or "none"
    if sink in SINKS:
        v[34 + SINKS.index(sink)] = 1.0
    v[44] = 1.0 if record.get("origin") == "same" else 0.0
    v[45] = 1.0 if record.get("origin") == "cross" else 0.0

    combined = urllib.parse.unquote(url) + " " + script
    v[OFFSET_HASH:] = hashed_ngrams(combined)
    return v


def featurise_all(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([featurise(r) for r in records]).astype(np.float32)
    y = np.array([int(r["label"]) for r in records], dtype=np.float32)
    return X, y
