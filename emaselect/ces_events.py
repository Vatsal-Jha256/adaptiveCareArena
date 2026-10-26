"""CES (College Experience Study) events for the item-selection
environment -- the independent-cohort counterpart to `dataset.py`'s
CrossCheck events.

Same contract, different instrument: four PHQ-4 items (0-3 each) whose
sum is the score, against CrossCheck's ten signed items. Everything
downstream takes the item set as data, so nothing else changes.

Two things are deliberately mirrored from the CrossCheck builder so the
two cohorts are compared like for like rather than one being handed a
better setup:

  * the target is the next event's score by default, not this event's
    (reconstructing today's score from today's items is deterministic
    and its difficulty is dominated by the learner, not the data)
  * the free context is passive sensing plus the last previous score,
    never the current event's own items

The score is RECOMPUTED from the items rather than read from
`phq4_score`, which disagrees with its own items in 0.5% of rows -- see
`caredata/ces.py`."""

from __future__ import annotations

import pandas as pd

from caredata.ces import PHQ4_ITEMS, CESDataset
from emaselect.dataset import EMAEvent

__all__ = ["PHQ4_ITEMS", "CES_BURDEN_PROFILES", "build_ces_events"]

# Burden profiles. As with CrossCheck these are experimental conditions,
# not measured burden -- nobody has timed what each PHQ-4 item costs a
# respondent. The sensitivity ordering is a stated modelling assumption:
# the two depression items (down/hopeless, anhedonia) are the ones a
# clinician would weigh more carefully than the anxiety items.
UNIFORM = {i: 1.0 for i in PHQ4_ITEMS}
SENSITIVITY_WEIGHTED = {
    "phq4-1": 0.8,  # feeling nervous/anxious
    "phq4-2": 0.9,  # unable to stop worrying
    "phq4-3": 1.6,  # feeling down/depressed/hopeless
    "phq4-4": 1.4,  # little interest or pleasure
}
INVERTED_SENSITIVITY = {i: round(2.4 - SENSITIVITY_WEIGHTED[i], 2) for i in PHQ4_ITEMS}

CES_BURDEN_PROFILES = {
    "uniform": UNIFORM,
    "sensitivity_weighted": SENSITIVITY_WEIGHTED,
    "inverted_sensitivity": INVERTED_SENSITIVITY,
}


def build_ces_events(
    local_root: str = "data/raw/ces",
    target_mode: str = "forecast",
    min_ema: int = 20,
    history_days: int = 7,
) -> list[EMAEvent]:
    """Build one ``EMAEvent`` per complete PHQ-4 report for subjects with ``min_ema`` reports.

    ``target_mode='forecast'`` targets the next report's score from the
    history; ``'reconstruct'`` targets the same report's score from the asked
    items. Raises ``ValueError`` for any other mode.
    """
    if target_mode not in ("forecast", "reconstruct"):
        raise ValueError(f"unknown target_mode {target_mode!r}")

    ds = CESDataset(local_root=local_root)
    ema = ds.load_ema()
    sensing = ds.load_sensing()
    keep = set(ds.list_subjects(min_ema=min_ema))
    ema = ema[ema["uid"].isin(keep)]

    passive_cols = [c for c in sensing.columns if c.endswith("_daily")]
    sens_by_uid = {
        uid: g.sort_values("day_ordinal" if "day_ordinal" in g else "date")
        for uid, g in sensing.groupby("uid")
    }

    events: list[EMAEvent] = []
    for uid, g in ema.groupby("uid"):
        g = g.sort_values("day_ordinal")
        ords = g["day_ordinal"].tolist()
        scores = dict(zip(ords, g["phq4_recomputed"].tolist(), strict=False))
        sg = sens_by_uid.get(uid)
        if sg is not None:
            sg = sg.copy()
            sg["ord"] = (sg["date"] - ema["date"].min()).dt.days

        for _pos, (_idx, row) in enumerate(g.iterrows()):
            ordv = float(row["day_ordinal"])

            if target_mode == "reconstruct":
                target = float(row["phq4_recomputed"])
            else:
                future = [d for d in ords if d > ordv]
                if not future:
                    continue  # no next event -- drop rather than invent one
                target = float(scores[min(future)])

            feats: dict[str, float] = {}
            if sg is not None and passive_cols:
                w = sg[(sg["ord"] <= ordv) & (sg["ord"] > ordv - history_days)]
                for c in passive_cols:
                    vals = (
                        pd.to_numeric(w[c], errors="coerce") if len(w) else pd.Series(dtype=float)
                    )
                    feats[f"{c}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                    feats[f"{c}_last"] = float(vals.iloc[-1]) if len(vals) else float("nan")

            # Last previous score, strictly before today.
            prior = [d for d in ords if d < ordv]
            feats["last_known_score"] = float(scores[max(prior)]) if prior else float("nan")
            feats["days_since_last_ema"] = float(ordv - max(prior)) if prior else float("nan")

            events.append(
                EMAEvent(
                    subject_id=str(uid),
                    day_ordinal=ordv,
                    item_values={i: float(row[i]) for i in PHQ4_ITEMS},
                    context_features=feats,
                    target_y=target,
                )
            )
    return events
