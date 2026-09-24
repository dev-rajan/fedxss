"""
Phase 2b — the two non-federated baselines. Fills Table 4.1 rows 1 and 2.

  1. Rule-based sanitiser-style filter. Approximates what a DOMPurify-style
     blocklist catches: dangerous tags, inline event handlers, javascript:
     URIs, and source->sink flows. No learning. This is the "current
     practice" comparator RQ1 needs.

  2. Centralised supervised MLP — every client's data pooled on one machine.
     This is the UPPER BOUND for the federated results: federated learning
     cannot beat it, only approach it. RQ1 asks how close it gets.

Both are evaluated on the same held-out test set with the same metrics, and
every run records its seed and config to results/baselines.json.

Sanity check when reading the output: an F1 of 1.0000 means the dataset is
separable by something trivial, not that the model is good. Published work
on hard XSS data reports 0.95-0.99.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

import numpy as np

from features import DIM, featurise_all
from features import DANGEROUS_TAGS, EVENT_HANDLER, JS_URI, SINK_CALL, SOURCE_READ
from model import HIDDEN, MLP

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEEDS = list(range(10))   # must match fed_sim.py for the paired tests
EPOCHS = 30
LR = 0.3
BATCH = 64


def load(split: str) -> list[dict]:
    with (PROC / f"{split}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Rank-based AUROC (Mann-Whitney U), implemented directly.

    Threshold-independent, which matters here: precision at the fixed 0.5
    threshold sits at or near 1.0 throughout, so threshold-dependent
    metrics understate how much separation the model actually has.
    Ties receive averaged ranks, so a model outputting constant scores
    scores 0.5 rather than something flattering.
    """
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    sp = y_prob[order]
    i = 0
    while i < len(sp):                      # average ranks within tied groups
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at_recall(y_true: np.ndarray, y_prob: np.ndarray,
                  target_recall: float = 0.95) -> dict:
    """
    False-positive rate at the threshold that first achieves target recall.

    This is the operationally meaningful number for a browser extension:
    "if we insist on catching 95% of attacks, what fraction of benign page
    events do we flag?" A user notices false positives; they never see the
    attacks that were caught.
    """
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return {"threshold": float("nan"), "fpr": float("nan"),
                "achieved_recall": float("nan")}
    for t in np.unique(y_prob)[::-1]:
        pred = y_prob >= t
        rec = pred[pos].sum() / n_pos
        if rec >= target_recall:
            return {"threshold": float(t),
                    "fpr": float(pred[neg].sum() / n_neg),
                    "achieved_recall": float(rec)}
    return {"threshold": float(np.min(y_prob)), "fpr": 1.0, "achieved_recall": 1.0}


def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            y_prob: np.ndarray | None = None,
            target_recall: float = 0.95) -> dict:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if y_prob is not None:
        out["auroc"] = auroc(y_true, y_prob)
        far = fpr_at_recall(y_true, y_prob, target_recall)
        out["fpr_at_recall"] = far["fpr"]
        out["threshold_at_recall"] = far["threshold"]
        out["achieved_recall"] = far["achieved_recall"]
        out["target_recall"] = target_recall
    return out


def rule_predict(records: list[dict]) -> np.ndarray:
    out = np.zeros(len(records), dtype=np.float32)
    for i, r in enumerate(records):
        blob = urllib.parse.unquote(r.get("url", "") or "") + " " + (r.get("script", "") or "")
        flagged = (
            bool(DANGEROUS_TAGS.search(blob))
            or bool(EVENT_HANDLER.search(blob))
            or bool(JS_URI.search(blob))
            or (bool(SOURCE_READ.search(blob)) and bool(SINK_CALL.search(blob)))
        )
        out[i] = 1.0 if flagged else 0.0
    return out


def train_centralised(Xtr, ytr, Xte, yte, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    net = MLP(DIM, HIDDEN, seed=seed)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.fit_epoch(Xtr, ytr, lr=LR, batch_size=BATCH, rng=rng)
    train_s = time.perf_counter() - t0
    prob = net.predict_proba(Xte)
    m = metrics(yte, (prob >= 0.5).astype(np.float32), prob)
    m["train_seconds"] = train_s
    m["n_params"] = net.n_params()
    return m


def per_category(net, records, X, y) -> dict:
    out = {}
    pred = (net.predict_proba(X) >= 0.5).astype(np.float32)
    for cat in ["reflected", "stored", "dom", "benign"]:
        idx = [i for i, r in enumerate(records) if r["category"] == cat]
        if idx:
            out[cat] = {"n": len(idx), "accuracy": float((y[idx] == pred[idx]).mean())}
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "not-a-git-repo"


def summarise(runs: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "fpr"]
    if runs and "auroc" in runs[0]:
        keys = keys + ["auroc", "fpr_at_recall"]
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in keys}


def main() -> int:
    train, test = load("train"), load("test")
    Xtr, ytr = featurise_all(train)
    Xte, yte = featurise_all(test)
    print(f"  train {Xtr.shape}   test {Xte.shape}   feature dim {DIM}")

    rule = metrics(yte, rule_predict(test))   # deterministic: no scores, no AUROC
    print(f"\n  rule-based       acc={rule['accuracy']:.4f}  P={rule['precision']:.4f}  "
          f"R={rule['recall']:.4f}  F1={rule['f1']:.4f}  FPR={rule['fpr']:.4f}")

    runs = []
    for s in SEEDS:
        m = train_centralised(Xtr, ytr, Xte, yte, s)
        runs.append(m)
        print(f"  centralised s={s}  acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  FPR={m['fpr']:.4f}  "
              f"({m['train_seconds']:.1f}s)")

    summary = summarise(runs)
    print(f"\n  centralised mean F1 = {summary['f1']['mean']:.4f} "
          f"(sd {summary['f1']['std']:.4f})")

    # per-category breakdown using seed 0
    net = MLP(DIM, HIDDEN, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(EPOCHS):
        net.fit_epoch(Xtr, ytr, lr=LR, batch_size=BATCH, rng=rng)
    cats = per_category(net, test, Xte, yte)
    print("\n  per-category accuracy (seed 0):")
    for cat, v in cats.items():
        print(f"    {cat:10} n={v['n']:5}  acc={v['accuracy']:.4f}")

    if summary["f1"]["mean"] >= 0.9995:
        print("\n  ** WARNING: F1 at ceiling — check the dataset for a trivial split **")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "config": {"dim": DIM, "hidden": HIDDEN, "epochs": EPOCHS,
                   "lr": LR, "batch_size": BATCH, "seeds": SEEDS},
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "rule_based": rule,
        "centralised_runs": runs,
        "centralised_summary": summary,
        "per_category_seed0": cats,
    }, indent=2))
    print(f"\n  written to {RESULTS / 'baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
