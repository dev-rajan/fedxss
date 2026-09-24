"""
Shared context construction for BOTH classes.

The first version of this pipeline built malicious records with URLs like
/search?q=<encoded payload> and benign records from real crawled URLs. The
two classes therefore had different URL shapes, and the detector could
separate them on URL structure alone without ever inspecting the content.

Here, benign and malicious records are constructed by the SAME functions
from the same host pool. Only the VALUE carried in the context differs:
a real XSS payload, or ordinary page content. Everything else — path
shape, parameter names, encoding, sink, origin — is drawn from identical
distributions.

This is what makes the task non-trivial: the detector must decide whether
the value flowing into a sink is dangerous, which is the question Section
3.2 actually poses, rather than whether the URL looks unusual.
"""

from __future__ import annotations

import urllib.parse

REFLECTED_PARAMS = ["q", "search", "s", "keyword", "query", "name", "redirect", "msg"]
REFLECTED_PATHS = ["/search", "/results", "/find", "/index.php", "/error", "/welcome"]

STORED_PATHS = ["/comments", "/profile", "/post", "/review", "/forum/thread", "/guestbook"]

DOM_SINKS_ATTACK = ["innerHTML", "outerHTML", "document.write",
                    "insertAdjacentHTML", "eval", "setAttribute"]

# Page code that reads an attacker-controllable source and writes to a sink.
# Vulnerable page code is vulnerable regardless of what flows through it, so
# benign records use these templates too — the value decides the label.
DOM_TEMPLATES = [
    ('document.getElementById("out").innerHTML = location.hash.slice(1);', "innerHTML"),
    ('document.write(decodeURIComponent(location.search.substring(1)));', "document.write"),
    ('el.insertAdjacentHTML("beforeend", new URLSearchParams(location.search).get("t"));',
     "insertAdjacentHTML"),
    ('out.innerHTML = new URLSearchParams(location.search).get("t");', "innerHTML"),
    ('target.outerHTML = window.name;', "outerHTML"),
    ('node.setAttribute("title", location.hash.slice(1));', "setAttribute"),
]

CONTEXTS = ["reflected", "stored", "dom"]


def make_reflected(rng, host: str, value: str) -> dict:
    path = rng.choice(REFLECTED_PATHS)
    param = rng.choice(REFLECTED_PARAMS)
    return {
        "url": f"https://{host}{path}?{param}={urllib.parse.quote(value, safe='')}",
        "script": value,
        "sink": rng.choice(["innerHTML", "document.write", "none"]),
        "origin": "same",
    }


def make_stored(rng, host: str, value: str) -> dict:
    path = rng.choice(STORED_PATHS)
    return {
        # the payload is in the rendered markup, not the URL
        "url": f"https://{host}{path}/{rng.randint(100, 99999)}",
        "script": value,
        "sink": rng.choice(["innerHTML", "insertAdjacentHTML", "outerHTML"]),
        "origin": "same",
    }


def make_dom(rng, host: str, value: str) -> dict:
    tpl, tpl_sink = DOM_TEMPLATES[rng.randrange(len(DOM_TEMPLATES))]
    carrier = "#" if ("hash" in tpl or "name" in tpl) else "?t="
    return {
        "url": f"https://{host}/app{carrier}{urllib.parse.quote(value, safe='')}",
        "script": f"{tpl} /* value: {value} */",
        "sink": tpl_sink if rng.random() < 0.7 else rng.choice(DOM_SINKS_ATTACK),
        "origin": rng.choice(["same", "same", "cross"]),
    }


BUILDERS = {"reflected": make_reflected, "stored": make_stored, "dom": make_dom}


def build(rng, context: str, host: str, value: str) -> dict:
    return BUILDERS[context](rng, host, value)
