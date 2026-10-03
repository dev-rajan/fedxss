"""
Emit compact inputs for the Chapter 4 figures.

Writes results/figdata.json containing only what the plots need:
  * per-round convergence histories (seed 0) for the key configurations
  * per-seed final F1 for every configuration, for the stability scatter
  * client shard composition from the partition manifest

The full results files are 240 kB because they retain every run's history;
this condenses them so the figures can be generated without re-running any
experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SHARDS = ROOT / "data" / "processed" / "shards"
OUT = RES / "figdata.json"

WANT = [
    "p1.0/iid/10", "p1.0/cat_severe/10", "p1.0/lab_severe/10",
    "p0.2/iid/10", "p0.2/lab_mild/10", "p0.2/lab_severe/10",
    "p0.2/lab_severe/100",
]


def main() -> int:
    fed = json.loads((RES / "fed_nodp.json").read_text())
    out = {"convergence": {}, "per_seed_f1": {}, "shards": {}}

    for key, cell in fed["results"].items():
        runs = cell["runs"]
        out["per_seed_f1"][key] = [round(r["f1"], 4) for r in runs]
        if key in WANT:
            out["convergence"][key] = [
                [h["round"], round(h["accuracy"], 4)] for h in runs[0].get("history", [])
            ]

    man = SHARDS / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        for prof in ("iid/10", "cat_severe/10", "lab_severe/10"):
            if prof in m:
                out["shards"][prof] = [
                    {"n": s["n"], "pos": s["pos"], "cats": s.get("cats", {})}
                    for s in m[prof]
                ]

    OUT.write_text(json.dumps(out, indent=1))
    print(f"  convergence series : {len(out['convergence'])}")
    print(f"  per-seed cells     : {len(out['per_seed_f1'])}")
    print(f"  shard profiles     : {len(out['shards'])}")
    print(f"  written to {OUT}  ({OUT.stat().st_size/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
