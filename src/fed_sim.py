"""
Phase 3 — federated experiments without differential privacy.

Answers RQ1 (how close does FedAvg get to centralised?) and RQ4 (what does
non-IID client data cost?). Fills Table 4.1 row 3 and the heterogeneity
panel of Chapter 4.

Grid: {iid, mild, severe} x {10, 100} clients x 5 seeds, run twice — once
with full participation and once with partial participation, which is the
realistic browser setting and the one where heterogeneity actually bites.

Each run writes its full config and per-round history to
results/fed_nodp.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from baselines import load, metrics, summarise, git_commit
from fed_core import bytes_per_round, run_federated
from features import DIM, featurise_all
from model import HIDDEN, MLP

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "data" / "processed" / "shards"
RESULTS = ROOT / "results"

PROFILES = ["iid", "cat_mild", "cat_severe", "lab_mild", "lab_severe"]
CLIENT_COUNTS = [10, 100]
# Ten, not five. The Wilcoxon signed-rank test over five paired seeds has
# a smallest attainable two-sided p of 0.0625, so no RQ1 comparison could
# ever reach significance. Ten pairs lower that floor to 0.002. See the
# header of stats.py.
SEEDS = list(range(10))

ROUNDS = 30
# Five, not one. With a single local epoch clients barely move before
# averaging, so there is no client drift and heterogeneity costs nothing —
# measured at 0.0001 F1, i.e. nothing at all. At five local epochs the
# drift appears (IID 0.9946 vs severe label skew 0.9915) AND the IID
# federated result rises closer to centralised. One local epoch was the
# wrong configuration on both counts.
LOCAL_EPOCHS = 5
LR = 0.3
BATCH = 64

# Fraction of clients sampled per round. 1.0 = full participation.
PARTICIPATION = [1.0, 0.2]


def load_shards(profile: str, n_clients: int):
    out = []
    for i in range(n_clients):
        path = SHARDS / profile / str(n_clients) / f"client_{i}.csv"
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            out.append(featurise_all(rows))
        else:
            out.append((np.zeros((0, DIM), np.float32), np.zeros(0, np.float32)))
    return out


def main() -> int:
    test = load("test")
    Xte, yte = featurise_all(test)
    print(f"  test {Xte.shape}")

    all_results = {}
    for frac in PARTICIPATION:
        print(f"\n  participation = {frac:.0%}")
        for profile in PROFILES:
            for n in CLIENT_COUNTS:
                shards = load_shards(profile, n)
                k = max(1, int(round(n * frac)))
                runs = []
                t0 = time.perf_counter()
                for seed in SEEDS:
                    params, hist = run_federated(
                        shards, Xte, yte, dim=DIM, hidden=HIDDEN, rounds=ROUNDS,
                        clients_per_round=k, lr=LR, local_epochs=LOCAL_EPOCHS,
                        batch_size=BATCH, seed=seed)
                    net = MLP(DIM, HIDDEN, seed=seed)
                    net.set_parameters(params)
                    prob = net.predict_proba(Xte)
                    m = metrics(yte, (prob >= 0.5).astype(np.float32), prob)
                    m["history"] = hist
                    runs.append(m)
                summary = summarise(runs)
                key = f"p{frac}/{profile}/{n}"
                all_results[key] = {"clients_per_round": k, "runs": runs,
                                    "summary": summary}
                print(f"    {profile:7}/{n:<4} k={k:<4} F1={summary['f1']['mean']:.4f} "
                      f"(sd {summary['f1']['std']:.4f})  "
                      f"FPR={summary['fpr']['mean']:.4f}  "
                      f"[{time.perf_counter()-t0:.0f}s]")

    n_params = MLP(DIM, HIDDEN).n_params()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fed_nodp.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "lr": LR,
                   "batch_size": BATCH, "seeds": SEEDS, "dim": DIM,
                   "hidden": HIDDEN, "profiles": PROFILES,
                   "client_counts": CLIENT_COUNTS,
                   "participation": PARTICIPATION},
        "n_params": n_params,
        "upload_bytes_per_client_per_round": bytes_per_round(n_params),
        "results": all_results,
    }, indent=2))
    print(f"\n  model parameters: {n_params}")
    print(f"  upload per client per round: {bytes_per_round(n_params)/1024:.1f} KiB")
    print(f"  written to {RESULTS / 'fed_nodp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
