"""
Shared schema for a single browser-observed event.

Section 3.2 states the extension hooks the navigation, DOM-mutation and
script-execution lifecycles. One record therefore represents one observed
event, NOT a bare payload — this is what the feature extractor consumes.

Fields
------
url       : the URL the browser navigated to or requested
script    : the inline script / injected markup observed in the page
sink      : the DOM sink the value reached (see SINKS); "none" if no sink
origin    : "same" | "cross" — whether the script source matches page origin
category  : "reflected" | "stored" | "dom" | "benign"
label     : 1 = XSS, 0 = benign
family    : de-duplication / split-grouping key (see note below)

`family` exists to prevent train/test leakage. A single base payload is
embedded into several contexts, so all records derived from it share a
family id and must be assigned to the same split. Ordinary random splitting
would put near-identical rows on both sides and inflate every metric.
"""

FIELDS = ["url", "script", "sink", "origin", "category", "label", "family"]

# DOM-sink classes named in Section 3.2 / 3.4.
SINKS = [
    "none",
    "innerHTML",
    "outerHTML",
    "document.write",
    "insertAdjacentHTML",
    "eval",
    "setAttribute",
    "src",
    "href",
    "textContent",   # safe sink — appears in benign records as a hard negative
]
