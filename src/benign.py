"""
Phase 1b (v3) — the benign half, built from real page content.

History of this file, worth recording in Section 3.3
----------------------------------------------------
v1 drew benign scripts from 22 hand-written templates. Centralised F1 came
out at 0.9996 because the model memorised 22 strings.

v2 replaced them with ~2,600 real JavaScript snippets from library source.
F1 went to 1.0000 — diversity was never the problem. The real leak was
PROVENANCE: 99% of malicious records contained an HTML tag against 11% of
benign ones, because malicious records were markup and benign ones were
JavaScript. "Contains a tag" was the label.

v3 fixes it properly. Benign values are real HTML fragments from published
page templates, so both classes are markup and the detector must judge
whether the markup is dangerous. Benign records are also placed in the
SAME injection contexts as malicious ones (see contexts.py), so URL shape
carries no label either.

Sources
-------
  * 16 real Bootstrap page templates + html5-boilerplate  -> benign HTML
  * library source incl. DOMPurify                        -> benign JS
  * 344k labelled-benign URLs (faizann24 'good' rows)     -> host pool
  * top-10,000 domain list
  * optional: XSS_dataset.csv label-0 rows, if present in the parent
    folder — real Wikipedia markup, adds provenance diversity. 70% is used
    for training; the rest is written to external_holdout_benign.txt and
    never touched again.
"""

from __future__ import annotations

import csv
import random
import re
import urllib.parse
from pathlib import Path

import contexts
from net import fetch as _fetch
from schema import FIELDS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "benign.csv"
URL_POOL = ROOT / "data" / "processed" / "benign_urls.txt"

# Optional local corpus: Thesis/XSS_dataset.csv, one level above the repo.
EXTERNAL_CSV = ROOT.parent / "XSS_dataset.csv"

URLS_SRC = ("https://raw.githubusercontent.com/faizann24"
            "/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv")
DOMAINS_SRC = ("https://raw.githubusercontent.com/zer0h/top-1000000-domains"
               "/master/top-10000-domains")

SB = "https://raw.githubusercontent.com/StartBootstrap/startbootstrap-{}/master/dist/index.html"
HTML_TEMPLATES = ["freelancer", "agency", "creative", "grayscale", "stylish-portfolio",
                  "clean-blog", "resume", "landing-page", "new-age", "business-casual",
                  "bare", "one-page-wonder", "heroic-features", "full-width-pics",
                  "blog-post", "shop-homepage"]
HTML_SOURCES = {f"html_{t}": SB.format(t) for t in HTML_TEMPLATES}
HTML_SOURCES["html_boilerplate"] = (
    "https://raw.githubusercontent.com/h5bp/html5-boilerplate/main/dist/index.html")

JS_SOURCES = {
    "js_jquery": "https://raw.githubusercontent.com/jquery/jquery/main/src/core.js",
    "js_lodash": "https://raw.githubusercontent.com/lodash/lodash/main/lodash.js",
    "js_axios": "https://raw.githubusercontent.com/axios/axios/v1.x/lib/core/Axios.js",
    "js_bootstrap": "https://raw.githubusercontent.com/twbs/bootstrap/main/js/src/modal.js",
    "js_vue": "https://raw.githubusercontent.com/vuejs/core/main/packages/runtime-dom/src/index.ts",
    # a sanitiser's own source: the hardest negatives available
    "js_dompurify": "https://raw.githubusercontent.com/cure53/DOMPurify/main/src/purify.ts",
    "js_select2": "https://raw.githubusercontent.com/select2/select2/develop/src/js/select2/utils.js",
}

N_BENIGN = 8000
SEED = 42
EXTERNAL_TRAIN_FRACTION = 0.7   # remainder held out, never used in training

TAG_CHUNK = re.compile(r"<[a-zA-Z][^>]*>(?:[^<]{0,200})(?:</[a-zA-Z]+>)?")
LICENCE_NOISE = re.compile(r"(copyright|licen[cs]e|all rights reserved|SPDX)", re.I)

PLAIN_TEXT = [
    "how to use div in html", "quarterly report 2026", "order 4821 shipped",
    "welcome back", "contact support", "javascript tutorial for beginners",
    "alert box example", "img src attribute guide", "page 7 of 12",
    "eval vs JSON.parse", "a < b && c > d", "temperature today is 23 degrees",
]


def fetch(name: str, url: str) -> str:
    return _fetch(RAW / name, url, timeout=180)


def html_fragments(source: str) -> list[str]:
    """Element-level fragments of real markup."""
    out = []
    for m in TAG_CHUNK.finditer(source):
        frag = " ".join(m.group(0).split())
        if 25 <= len(frag) <= 400 and not LICENCE_NOISE.search(frag):
            out.append(frag)
    return out


def js_snippets(source: str) -> list[str]:
    lines = [ln.strip() for ln in source.splitlines()]
    out = []
    for i, ln in enumerate(lines):
        if not ln or LICENCE_NOISE.search(ln):
            continue
        for span in (1, 2):
            chunk = " ".join(l for l in lines[i:i + span] if l)
            if 25 <= len(chunk) <= 400:
                out.append(chunk)
    return out


def load_pool(sources: dict, extractor) -> list[str]:
    seen, out = set(), []
    for name, url in sources.items():
        try:
            text = fetch(f"{name}.txt", url)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}")
            continue
        for s in extractor(text):
            key = re.sub(r"\s+", "", s)
            if key not in seen:
                seen.add(key)
                out.append(s)
    return out


def load_external_benign(rng) -> list[str]:
    """Real benign markup from XSS_dataset.csv, if the file is present."""
    if not EXTERNAL_CSV.exists():
        print(f"  (optional {EXTERNAL_CSV.name} not found — skipping)")
        return []
    rows = []
    with EXTERNAL_CSV.open(encoding="utf-8", errors="replace") as fh:
        for rec in csv.DictReader(fh):
            sent = (rec.get("Sentence") or rec.get("sentence") or "").strip()
            lab = (rec.get("Label") or rec.get("label") or "").strip()
            if lab == "0" and 25 <= len(sent) <= 400:
                rows.append(" ".join(sent.split()))
    rng.shuffle(rows)
    n_train = int(len(rows) * EXTERNAL_TRAIN_FRACTION)
    holdout = rows[n_train:]
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed" / "external_holdout_benign.txt").write_text(
        "\n".join(holdout), encoding="utf-8")
    print(f"  external corpus: {len(rows)} benign rows "
          f"({n_train} train, {len(holdout)} held out)")
    return rows[:n_train]


def load_benign_urls() -> list[str]:
    text = fetch("urls.csv", URLS_SRC)
    hosts = []
    for line in text.splitlines()[1:]:
        if not line.endswith(",good"):
            continue
        u = line[: -len(",good")].strip()
        if u and len(u) <= 300:
            hosts.append(u)
    return hosts


def load_domains() -> list[str]:
    return [d.strip() for d in fetch("top-domains.txt", DOMAINS_SRC).splitlines() if d.strip()]


def as_url(raw: str) -> str:
    return raw if raw.startswith(("http://", "https://")) else "https://" + raw


def host_of(u: str) -> str:
    return urllib.parse.urlparse(as_url(u)).netloc or "example.com"


def main() -> int:
    rng = random.Random(SEED)

    html_pool = load_pool(HTML_SOURCES, html_fragments)
    js_pool = load_pool(JS_SOURCES, js_snippets)
    ext_pool = load_external_benign(rng)
    print(f"  benign HTML fragments: {len(html_pool)}")
    print(f"  benign JS snippets   : {len(js_pool)}")
    print(f"  external HTML rows   : {len(ext_pool)}")
    if len(html_pool) + len(ext_pool) < 500:
        print("  ! benign markup pool too small — check network access")
        return 1

    raw_urls = load_benign_urls()
    domains = load_domains()
    hosts = [host_of(u) for u in raw_urls[:40000]] or domains
    print(f"  host pool: {len(hosts)}")

    # Benign values are weighted towards markup, because malicious values are
    # always markup — the classes must look alike in kind, not just in count.
    def pick_value() -> str:
        r = rng.random()
        if ext_pool and r < 0.35:
            return rng.choice(ext_pool)
        if html_pool and r < 0.70:
            return rng.choice(html_pool)
        if js_pool and r < 0.92:
            return rng.choice(js_pool)
        return rng.choice(PLAIN_TEXT)

    rows = []
    for i in range(N_BENIGN):
        value = pick_value()
        host = rng.choice(hosts)
        ctx = contexts.CONTEXTS[i % 3]
        built = contexts.build(rng, ctx, host, value)
        built.update({"category": "benign", "label": 0, "family": f"benign_{i}"})
        rows.append({k: built[k] for k in FIELDS})

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    URL_POOL.write_text(
        "\n".join(f"https://{h}" for h in rng.sample(hosts, min(20000, len(hosts)))),
        encoding="utf-8")

    tag = re.compile(r"<\s*[a-zA-Z]")
    with_tag = sum(1 for r in rows if tag.search(r["script"]))
    print(f"\n  benign records: {len(rows)}")
    print(f"  unique benign values: {len({r['script'] for r in rows})}")
    print(f"  containing an HTML tag: {with_tag}/{len(rows)} ({with_tag/len(rows):.1%})")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
