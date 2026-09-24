"""Dump Python feature vectors for a sample of records, for JS parity testing."""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import featurise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "data" / "processed" / "test.csv").open(encoding="utf-8")))
random.seed(0)
sample = random.sample(rows, min(300, len(rows)))
out = [{"record": {k: r[k] for k in ("url", "script", "sink", "origin")},
        "vector": [float(x) for x in featurise(r)]} for r in sample]
(ROOT / "tools" / "parity_vectors.json").write_text(json.dumps(out))
print(f"dumped {len(out)} vectors")

# Also emit a larger, representative record sample for the browser
# benchmark. Timing five short synthetic strings measures the best case;
# real page content is longer and more varied, and the difference is about
# fivefold — too large to leave unaddressed in Table 4.3.
bench = random.sample(rows, min(2000, len(rows)))
bench_out = [{k: r[k] for k in ("url", "script", "sink", "origin")} for r in bench]
(ROOT / "extension" / "bench_samples.json").write_text(json.dumps(bench_out))
print(f"dumped {len(bench_out)} benchmark records to extension/bench_samples.json")
