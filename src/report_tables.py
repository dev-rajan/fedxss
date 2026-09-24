"""
Emit exactly the numbers each Chapter 4 table needs, as plain text.

The results JSON files are large (200 kB+) because they retain every run's
per-round history. This condenses them to the handful of figures that
appear in the dissertation, so the tables can be filled without anyone
transcribing numbers by hand — transcription is where table errors come
from, and a wrong number in Chapter 4 is worse than a missing one.

Writes results/tables.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tables.txt"

KEYS = ["accuracy", "precision", "recall", "f1", "fpr"]
EXTRA = ["auroc", "fpr_at_recall"]   # present once probabilities are recorded


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def agg(runs):
    keys = KEYS + [k for k in EXTRA if runs and k in runs[0]]
    return {k: (float(np.mean([r[k] for r in runs])),
                float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0)
            for k in keys}


def line(label, a, n=None):
    parts = "  ".join(f"{k}={a[k][0]:.4f}+/-{a[k][1]:.4f}" for k in a)
    suffix = f"  (n={n})" if n else ""
    return f"{label:34} {parts}{suffix}"


def main() -> int:
    lines = []
    base = load("baselines.json")
    fed = load("fed_nodp.json")
    dp = load("dp_sweep.json")

    if base:
        lines.append("== TABLE 4.1 sources ==")
        rb = base["rule_based"]
        lines.append(line("rule-based (deterministic)",
                          {k: (rb[k], 0.0) for k in KEYS}))
        lines.append(line("centralised supervised",
                          agg(base["centralised_runs"]),
                          len(base["centralised_runs"])))
        pc = base.get("per_category_seed0", {})
        if pc:
            lines.append("  per-category accuracy (seed 0): " + ", ".join(
                f"{k}={v['accuracy']:.4f} (n={v['n']})" for k, v in pc.items()))

    if fed:
        lines.append("")
        for key in sorted(fed["results"]):
            lines.append(line(f"federated {key}", agg(fed["results"][key]["runs"]),
                              len(fed["results"][key]["runs"])))
        lines.append("")
        lines.append(f"model parameters: {fed['n_params']}")
        lines.append("upload per client per round: "
                     f"{fed['upload_bytes_per_client_per_round']/1024:.1f} KiB")
        cfg = fed["config"]
        lines.append(f"config: rounds={cfg['rounds']} local_epochs={cfg['local_epochs']} "
                     f"lr={cfg['lr']} batch={cfg['batch_size']} "
                     f"seeds={len(cfg['seeds'])} participation={cfg['participation']}")

    if dp:
        lines.append("")
        lines.append("== TABLE 4.2 sources ==")
        for key in sorted(dp["results"]):
            v = dp["results"][key]
            lines.append(line(f"{key} (sigma={v['sigma']:.3f})", agg(v["runs"]),
                              len(v["runs"])))
        c = dp["config"]
        lines.append(f"delta={c['delta']}  clip={c['clip']}  "
                     f"rounds={c['rounds']}  local_epochs={c['local_epochs']}")
        lines.append(f"guarantee: {dp['guarantee']}")

    # dataset facts for section 4.1
    proc = ROOT / "data" / "processed"
    if (proc / "train.csv").exists():
        import csv
        counts = {}
        for split in ("train", "test"):
            with (proc / f"{split}.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            counts[split] = (len(rows), sum(1 for r in rows if r["label"] == "1"))
        lines.append("")
        lines.append("== SECTION 4.1 dataset ==")
        for s, (n, pos) in counts.items():
            lines.append(f"{s}: {n} records, {pos} positive ({pos/n:.3f})")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
