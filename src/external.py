"""
External validation on an independent corpus.

Why this matters more than any other single result
--------------------------------------------------
The strongest objection to this study is that the injection contexts are
synthesised: payloads are real, but where they were injected was
constructed. A reader is entitled to ask whether the detector learned XSS
or learned the generator.

This script answers that directly, using data the model has never seen and
which came from somewhere else entirely:

  * BENIGN  — the 30% of XSS_dataset.csv's benign rows held out by
              benign.py and written to external_holdout_benign.txt. Real
              Wikipedia page markup.
  * MALICIOUS — every malicious row of XSS_dataset.csv. These were never
              used at any point in training: benign.py reads only label-0
              rows, so the whole positive class of that corpus is
              untouched.

Both classes come from a published dataset assembled by other people for a
different study. If performance holds here, the detector generalises
beyond its own generator. If it collapses, that is a finding too and must
be reported rather than buried.

Note on comparability: these records carry no URL, DOM sink or origin
metadata, because that corpus is sentence-level. Those features are
therefore absent (sink="none", origin="same", url=""), and the detector
runs on script content alone. That is a HARDER setting than the main
evaluation, not an easier one, so treat the result as a lower bound and
say so in Chapter 4.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from baselines import load, metrics, git_commit, rule_predict
from features import DIM, featurise_all
from fed_core import run_federated
from fed_sim import load_shards
from model import HIDDEN, MLP

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EXTERNAL_CSV = ROOT.parent / "XSS_dataset.csv"
HOLDOUT = PROC / "external_holdout_benign.txt"

SEEDS = list(range(10))
ROUNDS, LOCAL_EPOCHS, LR, BATCH = 30, 5, 0.3, 64


def build_records() -> list[dict]:
    """Assemble the external evaluation set. No URL/sink/origin available."""
    records = []

    if not HOLDOUT.exists():
        raise SystemExit(f"missing {HOLDOUT} — run benign.py first")
    for line in HOLDOUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append({"url": "", "script": line.strip(), "sink": "none",
                            "origin": "same", "category": "benign", "label": 0})
    n_ben = len(records)

    if not EXTERNAL_CSV.exists():
        raise SystemExit(f"missing {EXTERNAL_CSV}")
    with EXTERNAL_CSV.open(encoding="utf-8", errors="replace") as fh:
        for rec in csv.DictReader(fh):
            sent = (rec.get("Sentence") or rec.get("sentence") or "").strip()
            lab = (rec.get("Label") or rec.get("label") or "").strip()
            if lab == "1" and sent:
                records.append({"url": "", "script": " ".join(sent.split()),
                                "sink": "none", "origin": "same",
                                "category": "external_mal", "label": 1})

    n_mal = len(records) - n_ben
    print(f"  external set: {n_ben} benign (held out), {n_mal} malicious "
          f"(never used), {len(records)} total")
    return records


def main() -> int:
    ext = build_records()
    Xe, ye = featurise_all(ext)

    train = load("train")
    Xtr, ytr = featurise_all(train)
    test = load("test")
    Xte, yte = featurise_all(test)

    # rule-based baseline on the same external data, for context
    rule = metrics(ye, rule_predict(ext))
    print(f"\n  rule-based (external)   F1={rule['f1']:.4f}  "
          f"P={rule['precision']:.4f}  R={rule['recall']:.4f}  "
          f"FPR={rule['fpr']:.4f}")

    out = {"n_external": len(ext),
           "n_benign": int((ye == 0).sum()),
           "n_malicious": int((ye == 1).sum()),
           "rule_based_external": rule}

    for name in ("centralised", "federated_iid"):
        runs_int, runs_ext = [], []
        for seed in SEEDS:
            net = MLP(DIM, HIDDEN, seed=seed)
            if name == "centralised":
                rng = np.random.default_rng(seed)
                for _ in range(ROUNDS):
                    net.fit_epoch(Xtr, ytr, lr=LR, batch_size=BATCH, rng=rng)
            else:
                shards = load_shards("iid", 10)
                params, _ = run_federated(
                    shards, Xte, yte, dim=DIM, hidden=HIDDEN, rounds=ROUNDS,
                    lr=LR, local_epochs=LOCAL_EPOCHS, batch_size=BATCH, seed=seed)
                net.set_parameters(params)

            pi = net.predict_proba(Xte)
            runs_int.append(metrics(yte, (pi >= 0.5).astype(np.float32), pi))
            pe = net.predict_proba(Xe)
            runs_ext.append(metrics(ye, (pe >= 0.5).astype(np.float32), pe))

        def ag(runs, k):
            return float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))

        out[name] = {"internal_runs": runs_int, "external_runs": runs_ext}
        print(f"\n  {name}")
        for k in ("f1", "precision", "recall", "fpr", "auroc"):
            mi, si = ag(runs_int, k)
            me, se = ag(runs_ext, k)
            print(f"    {k:10} internal {mi:.4f}+/-{si:.4f}   "
                  f"external {me:.4f}+/-{se:.4f}   delta {me-mi:+.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "external.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "note": "external benign = held-out 30% of XSS_dataset.csv benign rows; "
                "external malicious = all XSS_dataset.csv malicious rows, never "
                "used in training. No URL/sink/origin features available, so "
                "this is a harder setting than the main evaluation.",
        "results": out,
    }, indent=2))
    print(f"\n  written to {RESULTS / 'external.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
