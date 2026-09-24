"""
Phase 1d — merge, split, and partition into federated client shards.

Split
-----
Grouped by `family` so that no payload appears on both sides of the
train/test boundary. A plain random split would leak and inflate every
number in Table 4.1.

Partitions (Section 3.3)
------------------------
  iid        uniform random assignment
  mild       Dirichlet alpha = 1.0 over attack categories
  severe     Dirichlet alpha = 0.1 over attack categories

Dirichlet skew is applied over the CATEGORY dimension (reflected / stored /
dom), which is what RQ4 asks about: clients that observe disjoint subsets of
attack types. Benign records are spread uniformly so every client can still
learn the negative class.

Writes data/processed/shards/{profile}/{n_clients}/client_{i}.csv plus a
manifest recording the per-client class and category counts, which you can
cite directly in Section 4.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from schema import FIELDS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SHARDS = PROC / "shards"

TEST_FRACTION = 0.2
CATEGORIES = ["reflected", "stored", "dom"]
# Two families of heterogeneity:
#   cat_*    Dirichlet over ATTACK CATEGORY, benign spread uniformly.
#            This is what Section 3.3 originally specified: clients that
#            observe disjoint subsets of attack types.
#   lab_*    Dirichlet over LABEL (benign vs malicious). This is the
#            standard non-IID setting in the federated-learning literature
#            and models the realistic case where most users browse safely
#            and rarely observe an attack at all, while a few see many.
#
# Category skew alone produced no measurable effect, because the features
# that separate attack from benign are shared across all three attack
# types: a client holding only reflected payloads still learns a boundary
# that transfers. Label skew is the sharper test and is reported alongside.
PROFILES = {
    "iid": (None, None),
    "cat_mild": ("category", 1.0),
    "cat_severe": ("category", 0.1),
    "lab_mild": ("label", 1.0),
    "lab_severe": ("label", 0.1),
}


def load_all() -> list[dict]:
    rows = []
    for name in ("malicious_records.csv", "benign.csv"):
        p = PROC / name
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run collect.py, benign.py, synth.py first.")
        with p.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def grouped_split(rows, seed):
    """Hold out whole families so no payload spans the split."""
    rng = random.Random(seed)
    fams = sorted({r["family"] for r in rows})
    rng.shuffle(fams)
    n_test = int(len(fams) * TEST_FRACTION)
    test_fams = set(fams[:n_test])
    train = [r for r in rows if r["family"] not in test_fams]
    test = [r for r in rows if r["family"] in test_fams]
    return train, test


def _dirichlet_split(rng, items, n_clients, alpha, clients):
    if not items:
        return
    rng.shuffle(items)
    props = rng.dirichlet(np.repeat(alpha, n_clients))
    cuts = (np.cumsum(props) * len(items)).astype(int)[:-1]
    for ci, chunk in enumerate(np.split(np.array(items, dtype=object), cuts)):
        clients[ci].extend(chunk.tolist())


def partition(train, n_clients, kind, alpha, seed):
    """Return a list of n_clients row-lists."""
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(n_clients)]

    if kind is None:  # IID
        idx = rng.permutation(len(train))
        for pos, i in enumerate(idx):
            clients[pos % n_clients].append(train[i])
        return clients

    groups = defaultdict(list)
    for r in train:
        if kind == "category":
            groups[r["category"] if r["label"] == 1 else "benign"].append(r)
        else:                                    # label skew
            groups["mal" if r["label"] == 1 else "benign"].append(r)

    if kind == "category":
        # skew the attack categories; keep benign uniform so every client
        # still has a negative class to learn from
        for cat in CATEGORIES:
            _dirichlet_split(rng, groups.get(cat, []), n_clients, alpha, clients)
        ben = groups["benign"]
        rng.shuffle(ben)
        for pos, r in enumerate(ben):
            clients[pos % n_clients].append(r)
    else:
        # skew BOTH classes independently: some clients end up almost
        # entirely benign, which is what a real browser fleet looks like
        _dirichlet_split(rng, groups["mal"], n_clients, alpha, clients)
        _dirichlet_split(rng, groups["benign"], n_clients, alpha, clients)

    for c in clients:
        rng.shuffle(c)
    return clients


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clients", type=int, nargs="+", default=[10, 100])
    args = ap.parse_args()

    rows = load_all()
    train, test = grouped_split(rows, args.seed)
    write(train, PROC / "train.csv")
    write(test, PROC / "test.csv")
    print(f"  total {len(rows)}  ->  train {len(train)}  test {len(test)}")
    print(f"  train positive rate: {sum(r['label'] for r in train)/len(train):.3f}")
    print(f"  test  positive rate: {sum(r['label'] for r in test)/len(test):.3f}")

    manifest = {}
    for profile, (kind, alpha) in PROFILES.items():
        for n in args.clients:
            clients = partition(train, n, kind, alpha, args.seed)
            key = f"{profile}/{n}"
            stats = []
            for i, c in enumerate(clients):
                write(c, SHARDS / profile / str(n) / f"client_{i}.csv")
                cats = defaultdict(int)
                for r in c:
                    cats[r["category"]] += 1
                stats.append({
                    "n": len(c),
                    "pos": sum(r["label"] for r in c),
                    "cats": dict(cats),
                })
            manifest[key] = stats
            sizes = [s["n"] for s in stats]
            rates = [s["pos"] / s["n"] for s in stats if s["n"]]
            print(f"  {key:16} sizes {min(sizes)}-{max(sizes)}  "
                  f"pos-rate {min(rates):.2f}-{max(rates):.2f}")

    (SHARDS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  shards written to {SHARDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
