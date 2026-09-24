"""
Phase 1c — realise each payload as an observed browser event.

A payload string alone has no attack category: <img src=x onerror=alert(1)>
is reflected, stored or DOM-based depending purely on where it is injected.
Category is assigned BY CONSTRUCTION, using the same builders benign.py
uses (contexts.py), so the two classes share URL shapes, parameter names,
encoding, sinks and origin distribution. Only the value differs.

LIMITATION — state this in Section 5.3: the injection CONTEXTS are
synthesised, not captured from live traffic. The payloads are real and
publicly documented and the contexts follow standard XSS mechanics, but the
corpus is not a capture of in-the-wild attacks.
"""

from __future__ import annotations

import csv
import random
import urllib.parse
from pathlib import Path

import contexts
from schema import FIELDS

ROOT = Path(__file__).resolve().parents[1]
MAL_IN = ROOT / "data" / "processed" / "malicious.csv"
URL_POOL = ROOT / "data" / "processed" / "benign_urls.txt"
OUT = ROOT / "data" / "processed" / "malicious_records.csv"

SEED = 42


def load_payloads() -> list[dict]:
    with MAL_IN.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_hosts() -> list[str]:
    if not URL_POOL.exists():
        raise SystemExit("Run benign.py first — it writes benign_urls.txt")
    hosts = []
    for line in URL_POOL.read_text(encoding="utf-8").splitlines():
        p = urllib.parse.urlparse(line.strip())
        if p.netloc:
            hosts.append(p.netloc)
    return hosts or ["example.com"]


def main() -> int:
    rng = random.Random(SEED)
    payloads = load_payloads()
    hosts = load_hosts()
    print(f"  payloads: {len(payloads)}   hosts: {len(hosts)}")

    rows = []
    for idx, rec in enumerate(payloads):
        ctx = contexts.CONTEXTS[idx % 3]
        built = contexts.build(rng, ctx, rng.choice(hosts), rec["payload"])
        built.update({"category": ctx, "label": 1, "family": f"mal_{idx}"})
        rows.append({k: built[k] for k in FIELDS})

    rng.shuffle(rows)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    dist = {}
    for r in rows:
        dist[r["category"]] = dist.get(r["category"], 0) + 1
    print(f"\n  malicious records: {len(rows)}")
    print(f"  category distribution: {dist}")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
