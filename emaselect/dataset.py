"""Builds item-level EMA events from real CrossCheck.

One example = one EMA administration that actually happened, carrying:

  * `item_values`   -- the ten recorded responses (the acquirable evidence)
  * `context_features` -- passive sensing + EMA history, free and always
                       available, so a policy that asks nothing still has
                       a prediction to make
  * `target_y`      -- depends on `target_mode`:
      "forecast"    (default) the ema_score at the patient's next EMA
      "reconstruct" the ema_score at this event

The offline-evaluation boundary this respects, and why it matters: we
only ever choose among items recorded at an event that really occurred.
We never simulate asking someone on a day they were not asked, because
their answer on that day does not exist and inventing it would make every
downstream number fiction. This is a retrospective item-selection
benchmark, not evidence about what would happen if a system pinged a
patient at an arbitrary time.

Why "forecast" is the default target. Reconstructing this event's score
from its own items is deterministic by construction (the score is an
exact sum of them), so the task is trivial in principle and its
difficulty is dominated by how well the learner approximates a linear
sum -- an artifact of the model class, not a fact about the data.
Forecasting the next EMA is the question Stage 1's result actually makes
interesting: persistence beat every learned model there, persistence
needs today's score, and today's score costs all ten questions. So the
live question becomes "how much of today's questionnaire do we actually
need in order to forecast as well as full persistence does?" That turns
Stage 1's negative result into the premise of this experiment instead of
a blocker.

Leakage: `context_features` here are strictly same-day-or-earlier passive
sensing plus the last previous ema_score -- never the current event's own
items, which are exactly what the policy is deciding whether to buy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from caredata.crosscheck import CrossCheckDataset
from emaselect.items import EMA_ITEMS, ema_score_from_items

# Passive daily signals used as free context, named by their column
# Prefix: the released daily table stores each of these as five six-hour
# epoch columns (`<prefix>_ep_0` .. `_ep_4`), which are summed to a daily
# total before windowing. Deliberately a small, dense set -- this
# benchmark is about the value OF asking, so the free side must be a fair
# passive baseline, not an exhaustive feature dump that buries the
# question.
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
# Already daily, no epoch structure.
PASSIVE_DAILY_COLUMNS = ("sleep_duration", "quality_activity", "quality_audio")


@dataclass
class EMAEvent:
    """One questionnaire administration.

    Attributes:
        subject_id: participant identifier.
        day_ordinal: day on which the questionnaire was answered.
        item_values: answer to each item.
        context_features: free passive-sensing context known beforehand.
        target_y: value the model is trained to predict.
    """

    subject_id: str
    day_ordinal: float
    item_values: dict[str, float]
    context_features: dict[str, float] = field(default_factory=dict)
    target_y: float = 0.0


def build_ema_events(
    local_root: str = "data/raw/crosscheck",
    history_days: int = 7,
    target_mode: str = "forecast",
) -> list[EMAEvent]:
    """Build one ``EMAEvent`` per CrossCheck questionnaire with all ten items answered.

    ``target_mode='forecast'`` targets the next response; ``'reconstruct'``
    targets the same response's score. Raises ``ValueError`` for any other mode.
    """
    if target_mode not in ("forecast", "reconstruct"):
        raise ValueError(f"unknown target_mode {target_mode!r}")
    ds = CrossCheckDataset(local_root=local_root)
    daily = ds.load_daily().copy()

    # Collapse each epoch family to one daily total once, up front. This
    # was previously re-loading the entire daily table inside the
    # per-subject loop, which is O(n_subjects) full reloads of the same
    # frame for no benefit.
    derived: dict[str, pd.Series] = {}
    for prefix in PASSIVE_EPOCH_PREFIXES:
        cols = [f"{prefix}_ep_{i}" for i in range(5) if f"{prefix}_ep_{i}" in daily.columns]
        if cols:
            derived[f"{prefix}_daily"] = (
                daily[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            )
    if derived:
        daily = pd.concat([daily, pd.DataFrame(derived, index=daily.index)], axis=1)

    passive_cols = [c for c in derived] + [c for c in PASSIVE_DAILY_COLUMNS if c in daily.columns]

    events: list[EMAEvent] = []
    for sid, full in daily.groupby("eureka_id"):
        full = full.sort_values("day_ordinal")
        # Rows with a complete EMA are the events; the full history
        # (including EMA-free days) supplies passive context.
        g = full.dropna(subset=list(EMA_ITEMS))
        event_ords = g["day_ordinal"].tolist()
        scored = {
            float(r["day_ordinal"]): ema_score_from_items({i: float(r[i]) for i in EMA_ITEMS})
            for _, r in g.iterrows()
        }

        for _, row in g.iterrows():
            ordv = float(row["day_ordinal"])
            window = full[
                (full["day_ordinal"] <= ordv) & (full["day_ordinal"] > ordv - history_days)
            ]
            if window.empty:
                continue

            feats: dict[str, float] = {}
            for col in passive_cols:
                vals = pd.to_numeric(window[col], errors="coerce")
                feats[f"{col}_mean"] = float(vals.mean())
                feats[f"{col}_last"] = float(vals.iloc[-1])

            # Last previous ema_score, strictly before today -- today's own
            # score is the target and must never appear as a feature.
            prior = full[(full["day_ordinal"] < ordv) & full["ema_score"].notna()]
            feats["last_known_ema_score"] = (
                float(prior["ema_score"].iloc[-1]) if len(prior) else float("nan")
            )
            feats["days_since_last_ema"] = (
                float(ordv - prior["day_ordinal"].iloc[-1]) if len(prior) else float("nan")
            )
            feats["weekday_sin"] = float(np.sin(2 * np.pi * row.get("weekday", 0) / 7))
            feats["weekday_cos"] = float(np.cos(2 * np.pi * row.get("weekday", 0) / 7))

            item_values = {i: float(row[i]) for i in EMA_ITEMS}

            if target_mode == "reconstruct":
                target = ema_score_from_items(item_values)
            else:
                future = [d for d in event_ords if d > ordv]
                if not future:
                    continue  # no next EMA exists for this event -- drop it
                target = scored[float(min(future))]

            events.append(
                EMAEvent(
                    subject_id=str(sid),
                    day_ordinal=ordv,
                    item_values=item_values,
                    context_features=feats,
                    target_y=float(target),
                )
            )
    return events


def assert_no_item_leakage(events: list[EMAEvent], items=None) -> None:
    """The current event's own item responses must never appear in the
    free context -- they are precisely what a policy pays to observe.

    `items` defaults to CrossCheck's set for backwards compatibility;
    pass the instrument's own items when checking another dataset."""
    item_set = set(EMA_ITEMS if items is None else items)
    for e in events:
        overlap = set(e.context_features) & item_set
        if overlap:
            raise ValueError(
                f"item leakage into free context for {e.subject_id}: {sorted(overlap)}"
            )
