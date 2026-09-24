"""
Phase 5a — export a trained model for the browser client.

Trains the federated (IID, 10 clients, no DP) configuration, which is the
one reported as the deployable setting in Chapter 4, and writes its
parameters plus the feature-extractor configuration to a single JSON file
the extension loads at startup.

Weights are emitted as plain arrays rather than ONNX. The detector is a
two-layer MLP with 20,609 parameters; a JSON array is roughly 440 kB, loads
with JSON.parse, and needs no runtime. Shipping ONNX Runtime Web to
evaluate two matrix products would add several megabytes of WebAssembly for
no benefit, and Section 3.2's argument is that the client must be light.
Record that choice in Section 3.4.

The file also carries the exact feature layout, so a mismatch between the
Python extractor and the JavaScript one fails loudly at load rather than
silently producing wrong features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines import load, metrics
from fed_core import run_federated
from features import DIM, N_HASH, OFFSET_HASH, featurise_all
from fed_sim import load_shards
from model import HIDDEN, MLP
from schema import SINKS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extension" / "model.json"

SEED = 0
ROUNDS, LOCAL_EPOCHS, LR, BATCH = 30, 5, 0.3, 64


def main() -> int:
    test = load("test")
    Xte, yte = featurise_all(test)

    shards = load_shards("iid", 10)
    params, _ = run_federated(shards, Xte, yte, dim=DIM, hidden=HIDDEN,
                              rounds=ROUNDS, lr=LR, local_epochs=LOCAL_EPOCHS,
                              batch_size=BATCH, seed=SEED)
    net = MLP(DIM, HIDDEN, seed=SEED)
    net.set_parameters(params)

    prob = net.predict_proba(Xte)
    m = metrics(yte, (prob >= 0.5).astype(np.float32), prob)
    print(f"  exported model: F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}  "
          f"FPR={m['fpr']:.4f}")

    W1, b1, W2, b2 = net.get_parameters()
    payload = {
        "architecture": {"dim": DIM, "hidden": HIDDEN,
                         "offset_hash": OFFSET_HASH, "n_hash": N_HASH},
        "sinks": SINKS,
        "threshold": 0.5,
        "provenance": {
            "training": "federated FedAvg, IID, 10 clients, no DP",
            "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
            "seed": SEED,
            "test_f1": m["f1"], "test_auroc": m["auroc"], "test_fpr": m["fpr"],
        },
        "weights": {
            "W1": [float(x) for x in W1.ravel()],   # row-major, dim x hidden
            "b1": [float(x) for x in b1.ravel()],
            "W2": [float(x) for x in W2.ravel()],   # hidden x 1
            "b2": [float(x) for x in b2.ravel()],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  {net.n_params()} parameters -> {OUT} "
          f"({OUT.stat().st_size/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
