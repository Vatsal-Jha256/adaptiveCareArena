"""College Experience Study (CES) adapter.

Nepal, Liu, Pillai, Wang, Vojdanovski, Huckins, Rogers, Meyer & Campbell
(2024), "Capturing the College Experience: A Four-Year Mobile Sensing
Study of Mental Health, Resilience, and Behavior of College Students
during the Pandemic", IMWUT 8(1) Art. 38. doi:10.1145/3643501

Why this dataset is here. Every result in this repo rested on one
cohort (CrossCheck, 78 subjects, 6,364 EMA events). CES supplies an
independent cohort at 220 participants and 35,348 complete PHQ-4 events,
median 170 per participant over a 4.8-year span -- which turns
"replicable at more seeds" into "holds on different people", a
categorically stronger claim.

Structural fit, verified against the files rather than assumed:

  * `phq4_score` is the sum of four items (`phq4-1..4`), each 0-3, the
    same shape as CrossCheck's signed ten-item `ema_score`. `emaselect`
    therefore transfers directly.
  * The passive-sensing schema is nearly the same feature family set
    (`act_*`, `audio_*`, `call_*`, `light_*`, `loc_dist`, `sms_*`,
    `unlock_*`), stored as four daily epochs. Both studies come from the
    same lab lineage, so a cross-dataset comparison is a genuine
    replication rather than a different measurement problem.

**The provided `phq4_score` column is not authoritative.** It disagrees
with the sum of its own items in 182 of 35,348 rows (0.5%), by -4 to +4.
This adapter RECOMPUTES the score from the items, which is the
instrument's definition, and exposes `score_mismatches()` so the
discrepancy is inspectable rather than silently absorbed.

Sensing is 522 MB across 651 columns; only the epoch families actually
used are read (`usecols`), because loading the whole frame to use ~20
columns is a needless several-GB spike."""

from __future__ import annotations

from functools import wraps
from pathlib import Path

import pandas as pd

PHQ4_ITEMS = ("phq4-1", "phq4-2", "phq4-3", "phq4-4")

# Epoch families kept as passive context. Chosen to match CrossCheck's
# adapter where the families coincide, so a cross-dataset comparison is
# not confounded by one cohort getting richer features than the other.
PASSIVE_EPOCH_PREFIXES = (
    "act_still",
    "act_walking",
    "act_in_vehicle",
    "audio_convo_duration",
    "audio_voice",
    "loc_dist",
    "loc_visit_num",
    "unlock_duration",
    "unlock_num",
    "sms_in_num",
    "sms_out_num",
    "call_in_duration",
)
N_EPOCHS = 4  # CES uses 4 daily epochs; CrossCheck uses 5


def _cache_on_instance(method):
    """Cache a no-argument method's result on the instance.

    Unlike ``functools.lru_cache`` on a method, this does not keep every
    instance alive in a module-level cache."""
    attr = f"_cached_{method.__name__}"

    @wraps(method)
    def wrapper(self):
        if not hasattr(self, attr):
            setattr(self, attr, method(self))
        return getattr(self, attr)

    return wrapper


class CESDataset:
    """Adapter for the College Experience Study EMA (PHQ-4) data under ``local_root``."""

    def __init__(self, local_root: str = "data/raw/ces"):
        self.local_root = Path(local_root)

    @_cache_on_instance
    def load_ema(self) -> pd.DataFrame:
        """Return complete PHQ-4 responses, one row per report.

        Rows missing any PHQ-4 item or a parseable date are dropped, the total is
        recomputed from the items, and ``day_ordinal`` counts days from the first
        report in the file. The result is cached.
        """
        df = pd.read_csv(self.local_root / "EMA" / "general_ema.csv")
        df = df.dropna(subset=list(PHQ4_ITEMS)).copy()
        # Recomputed, not taken from the file -- see module docstring.
        df["phq4_recomputed"] = df[list(PHQ4_ITEMS)].sum(axis=1)
        df["date"] = pd.to_datetime(df["day"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        df["day_ordinal"] = (df["date"] - df["date"].min()).dt.days
        return df.sort_values(["uid", "day_ordinal"]).reset_index(drop=True)

    def score_mismatches(self) -> dict:
        """How far the released `phq4_score` departs from its own items.
        Reported rather than silently corrected, because a 0.5% internal
        inconsistency is a property of the data a reader should know."""
        df = self.load_ema()
        if "phq4_score" not in df.columns:
            return {"checked": 0, "mismatched": 0}
        both = df.dropna(subset=["phq4_score"])
        diff = both["phq4_score"] - both["phq4_recomputed"]
        return {
            "checked": int(len(both)),
            "mismatched": int((diff != 0).sum()),
            "fraction": float((diff != 0).mean()),
            "range": [float(diff.min()), float(diff.max())],
        }

    @_cache_on_instance
    def load_sensing(self) -> pd.DataFrame:
        """Daily passive features, epochs summed to a daily total.

        `min_count=1` on the sum: pandas returns 0.0 for an all-missing
        group, which is indistinguishable from a real zero (a day the
        phone recorded no calls). Those mean opposite things to a model,
        so an all-missing day must stay NaN."""
        path = self.local_root / "Sensing" / "sensing.csv"
        wanted = {"uid", "day"}
        cols_needed = []
        for p in PASSIVE_EPOCH_PREFIXES:
            cols_needed += [f"{p}_ep_{i}" for i in range(N_EPOCHS)]
        import csv as _csv

        with open(path) as f:
            available = set(next(_csv.reader(f)))
        use = sorted(wanted | (set(cols_needed) & available))
        df = pd.read_csv(path, usecols=use)

        out = df[["uid", "day"]].copy()
        for p in PASSIVE_EPOCH_PREFIXES:
            cs = [f"{p}_ep_{i}" for i in range(N_EPOCHS) if f"{p}_ep_{i}" in df.columns]
            if cs:
                out[f"{p}_daily"] = (
                    df[cs].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
                )
        out["date"] = pd.to_datetime(out["day"], format="%Y%m%d", errors="coerce")
        return out.dropna(subset=["date"])

    def list_subjects(self, min_ema: int = 20) -> list[str]:
        """Return participants with at least ``min_ema`` complete reports, sorted."""
        counts = self.load_ema().groupby("uid").size()
        return sorted(counts[counts >= min_ema].index)

    def describe(self) -> dict:
        """Return aggregate counts for the loaded cohort (no participant rows)."""
        ema = self.load_ema()
        per = ema.groupby("uid").size()
        return {
            "n_participants": int(ema["uid"].nunique()),
            "n_complete_phq4_events": int(len(ema)),
            "median_events_per_participant": float(per.median()),
            "span_days": int(ema["day_ordinal"].max()),
            "score_mismatches": self.score_mismatches(),
        }
