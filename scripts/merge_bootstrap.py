"""Merge parallel bootstrap shards (stage2_bootstrap_sweep.py --seed k --out X_k.json) into one summary.

python -m scripts.merge_bootstrap --dataset ces --baseline round_robin_shuffled --out OUT.json shard1.json shard2.json ...
(reads each shard's <shard>.raw.json; summary format identical to stage2_bootstrap_sweep.py)"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.stage2_bootstrap_sweep import PCTS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--baseline", required=True)
ap.add_argument("shards", nargs="+")
a = ap.parse_args()
raws = [json.loads(Path(s + ".raw.json").read_text()) for s in a.shards]
meta = json.loads(Path(a.shards[0]).read_text())
by_metric = {}
n_boot = 0
for m in raws[0]:
    cells = {}
    for k in raws[0][m]:
        arr = np.array(sum((r[m][k] for r in raws), []))
        n_boot = len(arr)
        lo, hi = np.percentile(arr, PCTS)
        cells[k] = {
            "median_diff": float(np.median(arr)),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    by_metric[m] = cells
meta.update({"n_boot": n_boot, "by_metric": by_metric, "baseline": a.baseline, "shards": len(raws)})
Path(a.out).write_text(json.dumps(meta, indent=2))
print("wrote", a.out, "n_boot", n_boot)
