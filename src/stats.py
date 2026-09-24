"""
Phase 6 — statistical analysis of the experimental results.

Test selection, and why it differs from Section 3.6
---------------------------------------------------
Section 3.6 specified Wilcoxon signed-rank plus repeated-measures ANOVA.
Two corrections were forced by the data:

1. POWER. The Wilcoxon signed-rank test over n paired seeds has a smallest
   attainable two-sided p of 0.0625 at n = 5. With five seeds the test
   cannot return p < 0.05 no matter how large the effect. Ten seeds give a
   floor of 0.002. Every comparison below reports its own floor, so an
   underpowered result cannot be mistaken for a null one.

2. NORMALITY. Repeated-measures ANOVA assumes approximately normal
   residuals. Under severe label skew with few clients sampled per round,
   FedAvg sometimes fails to converge, producing values like 0.6269
   alongside 0.9874 in the same cell. That distribution is not normal and
   ANOVA on it would be invalid. Friedman's test — the non-parametric
   repeated-measures equivalent — is used instead, with Shapiro-Wilk
   reported so the choice is evidenced rather than asserted.

For unstable cells, mean and standard deviation are actively misleading.
Those cells are summarised instead by median, full range, a percentile
bootstrap interval, and an explicit count of non-convergent runs. A stated
failure mode is a stronger contribution than a smoothed-over average.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# A run is treated as non-convergent if its F1 falls this far below the
# best run in the same cell. Reported explicitly rather than hidden.
CONVERGENCE_GAP = 0.15
BOOTSTRAP_N = 10000


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the signed-rank test can return with n pairs."""
    if n < 1:
        return 1.0
    try:
        return float(sps.wilcoxon(np.arange(1, n + 1), np.zeros(n)).pvalue)
    except ValueError:
        return 1.0


def bootstrap_ci(values, alpha=0.05, n_boot=BOOTSTRAP_N, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float(v[0]), float(v[0])) if len(v) else (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(v)
    n_fail = int((v < v.max() - CONVERGENCE_GAP).sum())
    d = {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "ci95": [lo, hi],
        "values": [float(x) for x in v],
        "non_convergent_runs": n_fail,
        "unstable": bool(n_fail > 0),
    }
    if len(v) >= 3:
        d["shapiro_p"] = float(sps.shapiro(v).pvalue)
    return d


def fmt(d: dict) -> str:
    """Report mean +/- sd only when the cell is stable and plausibly normal."""
    if d["unstable"]:
        return (f"median {d['median']:.4f}  range [{d['min']:.4f}, {d['max']:.4f}]  "
                f"** {d['non_convergent_runs']}/{d['n']} runs failed to converge **")
    return (f"{d['mean']:.4f} +/- {d['std']:.4f}  "
            f"95% CI [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")


def paired_test(a, b, label_a: str, label_b: str) -> dict:
    """Wilcoxon signed-rank, with its own power floor reported."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    floor = wilcoxon_floor(n)
    out = {"comparison": f"{label_a} vs {label_b}", "n_pairs": n,
           "min_attainable_p": floor,
           "median_difference": float(np.median(a - b))}
    if np.allclose(a, b):
        out.update({"p_value": 1.0, "note": "identical values"})
        return out
    res = sps.wilcoxon(a, b)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["significant_at_0.05"] = bool(res.pvalue < 0.05)
    if floor > 0.05:
        out["WARNING"] = (f"underpowered: with {n} pairs the smallest "
                          f"attainable p is {floor:.4f}; a non-significant "
                          f"result here is uninformative, not evidence of "
                          f"no difference")
    return out


def friedman(groups: dict) -> dict:
    """Non-parametric repeated-measures test across >= 3 related groups."""
    names = list(groups)
    arrays = [np.asarray(groups[k], float) for k in names]
    if len(arrays) < 3 or len({len(a) for a in arrays}) != 1:
        return {"error": "need >=3 groups of equal length"}
    res = sps.friedmanchisquare(*arrays)
    return {"groups": names, "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "significant_at_0.05": bool(res.pvalue < 0.05)}


def grp_items(dp_cells: dict, prof: str):
    """(epsilon_key, values) pairs for one profile, ordered by epsilon."""
    items = [(k.split("/")[1], v) for k, v in dp_cells.items()
             if k.startswith(prof + "/")]
    return sorted(items, key=lambda kv: -float(kv[0].replace("eps", "")))


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    base = load("baselines.json")
    fed = load("fed_nodp.json")
    if base is None or fed is None:
        print("  run baselines.py and fed_sim.py first")
        return 1

    report = {}

    cen = [r["f1"] for r in base["centralised_runs"]]
    report["centralised"] = describe(cen)
    report["rule_based_f1"] = base["rule_based"]["f1"]
    print(f"  rule-based F1       {base['rule_based']['f1']:.4f}")
    print(f"  centralised F1      {fmt(report['centralised'])}\n")

    cells = {k: [r["f1"] for r in v["runs"]] for k, v in fed["results"].items()}
    report["federated"] = {k: describe(v) for k, v in cells.items()}
    for k in sorted(cells):
        print(f"  {k:26} {fmt(report['federated'][k])}")

    # -- RQ1: does federating cost accuracy? --------------------------------
    print("\n  RQ1 — federated vs centralised (Wilcoxon signed-rank)")
    report["rq1"] = {}
    for k in sorted(cells):
        if len(cells[k]) != len(cen):
            continue
        t = paired_test(cells[k], cen, k, "centralised")
        report["rq1"][k] = t
        flag = " [UNDERPOWERED]" if "WARNING" in t else ""
        print(f"    {k:26} median diff {t['median_difference']:+.4f}  "
              f"p={t['p_value']:.4f}{flag}")

    # -- RQ4: does heterogeneity matter? ------------------------------------
    print("\n  RQ4 — heterogeneity (Friedman)")
    report["rq4"] = {}
    for prefix in sorted({k.rsplit("/", 2)[0] for k in cells}):
        for n_clients in sorted({k.rsplit("/", 1)[1] for k in cells}):
            grp = {k.split("/")[-2]: v for k, v in cells.items()
                   if k.startswith(prefix + "/") and k.endswith("/" + n_clients)}
            if len(grp) >= 3:
                res = friedman(grp)
                report["rq4"][f"{prefix}/{n_clients}"] = res
                if "p_value" in res:
                    print(f"    {prefix}/{n_clients:4}  chi2={res['statistic']:.3f}  "
                          f"p={res['p_value']:.4f}")

    unstable = [k for k, d in report["federated"].items() if d["unstable"]]
    if unstable:
        print("\n  ** unstable cells (report median and range, never mean+/-sd):")
        for k in unstable:
            d = report["federated"][k]
            print(f"     {k:26} {d['non_convergent_runs']}/{d['n']} failed; "
                  f"values {[round(x, 3) for x in d['values']]}")

    # -- RQ3: what does privacy cost, and does it interact with skew? -------
    dp = load("dp_sweep.json")
    if dp is not None:
        print("\n  RQ3 — privacy/utility (Friedman across epsilon)")
        report["rq3"] = {}
        dp_cells = {k: [r["f1"] for r in v["runs"]]
                    for k, v in dp["results"].items()}
        for prof in sorted({k.split("/")[0] for k in dp_cells}):
            grp = {k.split("/")[1]: v for k, v in dp_cells.items()
                   if k.startswith(prof + "/")}
            if len(grp) >= 3:
                res = friedman(grp)
                report["rq3"][prof] = res
                if "p_value" in res:
                    print(f"    {prof:12} chi2={res['statistic']:.3f}  "
                          f"p={res['p_value']:.4f}")

        # The interaction is the actual finding: privacy is cheap under IID
        # and expensive under label skew. Quantified as the difference of
        # differences against the matching non-private cell.
        print("\n  RQ3 — privacy cost by profile (vs non-private, same profile)")
        report["rq3_interaction"] = {}
        for prof in sorted({k.split("/")[0] for k in dp_cells}):
            base_key = f"p1.0/{prof}/10"
            if base_key not in cells:
                continue
            nodp = np.mean(cells[base_key])
            entry = {"non_private_f1": float(nodp), "costs": {}}
            for eps_key, vals in sorted(grp_items(dp_cells, prof)):
                cost = float(nodp - np.mean(vals))
                entry["costs"][eps_key] = cost
                print(f"    {prof:12} {eps_key:10} cost {cost:+.4f}")
            report["rq3_interaction"][prof] = entry

        # false-positive rate is the deployability metric for an extension
        print("\n  RQ3 — false-positive rate (deployability)")
        for k, v in sorted(dp["results"].items()):
            print(f"    {k:22} FPR={v['summary']['fpr']['mean']:.4f}")

    (RESULTS / "stats.json").write_text(json.dumps(report, indent=2))
    print(f"\n  written to {RESULTS / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
