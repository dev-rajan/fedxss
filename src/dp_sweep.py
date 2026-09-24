"""
Phase 4 (runner) — the privacy/utility sweep. Fills Table 4.2 and Figure 4.1.

Answers RQ3: what does a given differential-privacy guarantee cost in
detection quality?

For each target epsilon the noise multiplier is calibrated to the ACTUAL
number of DP-SGD steps a client takes in this configuration, using the
median client shard size. Report the calibrated sigma next to epsilon in
Table 4.2: epsilon alone is not checkable by a reader.

The no-DP row comes from fed_nodp.json and should be repeated in the table
so it reads as one series from "no privacy" through to epsilon = 1.

Configuration choice: this runs over the 10-client shards at FULL
participation only. That is where the non-private results were stable.
Including the low-participation cells would confound degradation caused by
noise with degradation caused by unlucky client draws, and the two effects
would be impossible to separate in the write-up. Say so in Section 3.5.
Both the best case (iid) and the worst stable case (lab_severe) are run, so
the privacy cost is reported under easy and hard conditions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from baselines import load, metrics, summarise, git_commit
from dp import DPConfig, calibrate_noise, epsilon_for, steps_per_client
from fed_core import run_federated
from features import DIM, featurise_all
from fed_sim import load_shards
from model import HIDDEN, MLP

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PROFILES = ["iid", "lab_severe"]     # best and worst stable non-private cases
N_CLIENTS = 10
SEEDS = list(range(10))
EPSILONS = [10.0, 5.0, 2.0, 1.0]
DELTA = 1e-5
CLIP = 1.0

ROUNDS = 30
LOCAL_EPOCHS = 5
LR = 0.3
BATCH = 64


def main() -> int:
    test = load("test")
    Xte, yte = featurise_all(test)

    out = {}
    for profile in PROFILES:
        shards = load_shards(profile, N_CLIENTS)
        sizes = [len(X) for X, _ in shards if len(X)]
        n_median = int(np.median(sizes))
        q = min(1.0, BATCH / n_median)
        steps = steps_per_client(n_median, BATCH, LOCAL_EPOCHS, ROUNDS)
        print(f"\n  {profile}: median shard {n_median}, q={q:.4f}, "
              f"{steps} DP-SGD steps per client")

        for eps in EPSILONS:
            sigma = calibrate_noise(eps, q, steps, DELTA)
            achieved = epsilon_for(sigma, q, steps, DELTA)
            cfg = DPConfig(clip=CLIP, noise_multiplier=sigma,
                           sample_rate=q, delta=DELTA, epsilon=eps)
            runs = []
            t0 = time.perf_counter()
            for seed in SEEDS:
                params, _ = run_federated(
                    shards, Xte, yte, dim=DIM, hidden=HIDDEN, rounds=ROUNDS,
                    lr=LR, local_epochs=LOCAL_EPOCHS, batch_size=BATCH,
                    seed=seed, dp=cfg)
                net = MLP(DIM, HIDDEN, seed=seed)
                net.set_parameters(params)
                prob = net.predict_proba(Xte)
                runs.append(metrics(yte, (prob >= 0.5).astype(np.float32), prob))
            s = summarise(runs)
            key = f"{profile}/eps{eps}"
            out[key] = {"target_epsilon": eps, "achieved_epsilon": achieved,
                        "sigma": sigma, "clip": CLIP, "delta": DELTA,
                        "sample_rate": q, "steps": steps,
                        "runs": runs, "summary": s}
            print(f"    eps={eps:<5} sigma={sigma:7.3f}  "
                  f"F1={s['f1']['mean']:.4f} (sd {s['f1']['std']:.4f})  "
                  f"FPR={s['fpr']['mean']:.4f}  [{time.perf_counter()-t0:.0f}s]")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dp_sweep.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "config": {"profiles": PROFILES, "n_clients": N_CLIENTS,
                   "seeds": SEEDS, "epsilons": EPSILONS, "delta": DELTA,
                   "clip": CLIP, "rounds": ROUNDS,
                   "local_epochs": LOCAL_EPOCHS, "lr": LR, "batch_size": BATCH,
                   "participation": 1.0},
        "guarantee": "example-level (local) DP via per-example gradient "
                     "clipping and Poisson-subsampled Gaussian noise; "
                     "accounted with Renyi DP (dp-accounting)",
        "results": out,
    }, indent=2))
    print(f"\n  written to {RESULTS / 'dp_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
