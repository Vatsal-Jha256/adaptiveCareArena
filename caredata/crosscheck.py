"""CrossCheck adapter -- real schema, verified 2026-09-02 directly against
the downloaded files (data/raw/crosscheck/CrossCheck_Daily_Data.csv,
CrossCheck_Hourly_Data_v1_08132020.csv) and the official Excel data
dictionary (docs/DATA.md describes the schema).

Corrected 2026-09-02 (Stage 1 escalation attempt, after persistence beat
ridge/GBM at both prior feature-richness levels): the daily table's `day`
column is not a relative day-index -- it is a real calendar date encoded
as an int (YYYYMMDD, e.g. 20160228), confirmed by cross-referencing
against the hourly table's proper ISO timestamps for the same
study_id/date range. The original adapter treated `day` as if it were a
day-count and computed `horizon = target_day - day` and
`days_since_last_ema = day - last_ema_day` via raw integer subtraction --
correct only within a calendar month, silently wrong across a month
boundary (e.g. 20160301 - 20160228 = 73, not 2). Subtracting raw
YYYYMMDD integers would corrupt `days_since_last_ema` and the
administration gap for most subjects, so dates are converted to real
dates and all arithmetic uses ordinal days. This does not affect the
leakage invariant (target_observed_at is always
prediction_time_t + horizon); it is a signal-quality matter.

Also drops 22 single-row "day=19691231" sentinel rows (one per affected
subject, all-NaN ema_score, a discharge/exit placeholder artifact --
confirmed by inspection, not guessed) that previously sorted to the
front of each subject's sequence and could enter an early feature
window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from task.types import CareExample, assert_no_future_leakage

_SENTINEL_DAY = 19691231

# Hourly table columns used for circadian/regularity features -- the
# hourly table. accel_mean/screen/distance are not available at hourly
# granularity in the daily table's ep_0-4 six-hour aggregates -- these are
# new information (hour-of-day timing), not a duplicate of
# PASSIVE_BLOCKS/EPOCH_BLOCKS.
_HOURLY_NIGHT_HOURS = frozenset(range(0, 7))  # 12am-6:59am
_HOURLY_DAY_HOURS = frozenset(range(8, 22))  # 8am-9:59pm

EMA_ITEMS = (
    "ema_CALM",
    "ema_DEPRESSED",
    "ema_HARM",
    "ema_HOPEFUL",
    "ema_SEEING_THINGS",
    "ema_SLEEPING",
    "ema_SOCIAL",
    "ema_STRESSED",
    "ema_THINK",
    "ema_VOICES",
)
EMA_DERIVED = ("ema_neg_score", "ema_pos_score", "ema_score", "ema_resp_time_median")

# Full-day aggregate ("_ep_0") passive-sensing blocks.
PASSIVE_BLOCKS = (
    "act_in_vehicle_ep_0",
    "act_on_bike_ep_0",
    "act_on_foot_ep_0",
    "act_running_ep_0",
    "act_still_ep_0",
    "act_tilting_ep_0",
    "audio_amp_mean_ep_0",
    "audio_amp_std_ep_0",
    "audio_convo_duration_ep_0",
    "audio_convo_num_ep_0",
    "audio_voice_ep_0",
    "call_in_duration_ep_0",
    "call_in_num_ep_0",
    "call_miss_num_ep_0",
    "call_out_duration_ep_0",
    "call_out_num_ep_0",
    "light_mean_ep_0",
    "light_std_ep_0",
    "loc_dist_ep_0",
    "loc_visit_num_ep_0",
    "sms_in_num_ep_0",
    "sms_out_num_ep_0",
    "unlock_duration_ep_0",
    "unlock_num_ep_0",
)

# Six-hour-epoch breakdowns (_ep_1=12am-6am, _ep_2=6am-12pm, _ep_3=12pm-6pm,
# _ep_4=6pm-12am) of the same blocks. Added after a first pass
# (full-day aggregates only) failed to beat simple persistence: try a richer feature set before
# concluding the forecasting task itself is unanswerable. Only the blocks
# most plausibly informative about circadian/sleep-wake disruption are
# included here (activity, location, unlock, audio-conversation) to avoid
# a 4x explosion of noisy features on top of an already-weak signal.
EPOCH_BLOCKS = tuple(
    f"{base}_ep_{i}"
    for base in (
        "act_still",
        "act_on_foot",
        "loc_dist",
        "loc_visit_num",
        "unlock_num",
        "unlock_duration",
        "audio_convo_num",
        "audio_voice",
    )
    for i in (1, 2, 3, 4)
)
QUALITY_FIELDS = (
    "quality_activity",
    "quality_audio",
    "quality_gps_on",
    "quality_light",
    "quality_loc",
)


class CrossCheckDataset:
    """Adapter for the CrossCheck daily and hourly tables under ``local_root``.

    EMA is administered every 2-3 days, so most daily rows have EMA columns
    but no genuine EMA value. Everything here counts and targets only rows
    with a real response.
    """

    def __init__(self, local_root: str = "data/raw/crosscheck"):
        self.local_root = local_root
        self._daily: pd.DataFrame | None = None
        self._hourly_raw: pd.DataFrame | None = None
        self._hourly_daily_agg: pd.DataFrame | None = None

    def describe(self) -> dict:
        """Return a static description of the cohort and its role in the benchmark."""
        return {
            "name": "crosscheck",
            "role": "anchor_budgeted_acquisition",
            "population": "75 outpatients with schizophrenia-spectrum diagnoses, 63 completed >=12mo",
            "validated_against_real_files": True,
        }

    def load_daily(self) -> pd.DataFrame:
        """Reads the real CSV unless `_daily` was already injected (tests
        set it directly with a raw fixture to skip file I/O) -- either
        way, derives date/day_ordinal/weekday exactly once and caches the
        derived frame, so a test fixture only needs a real `day` column
        (YYYYMMDD ints), not the derived ones."""
        if self._daily is not None and "day_ordinal" in self._daily.columns:
            return self._daily
        df = self._daily
        if df is None:
            path = f"{self.local_root}/CrossCheck_Daily_Data.csv"
            df = pd.read_csv(path)
        df = df[df["day"] != _SENTINEL_DAY].copy()
        df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d")
        min_date = df["date"].min()
        df["day_ordinal"] = (df["date"] - min_date).dt.days.astype(float)
        df["weekday"] = df["date"].dt.weekday.astype(float)
        df = df.sort_values(["eureka_id", "day_ordinal"]).reset_index(drop=True)
        self._daily = df
        return df

    def _study_to_eureka_map(self) -> dict[int, str]:
        """Only 1:1 (study_id -> eureka_id) mappings -- study_id=-1 is a
        shared "no hourly mapping" sentinel used by 26 subjects, and two
        other study_ids each map to 2-3 eureka_ids ambiguously. Using
        either would silently attribute one subject's hourly sensor data
        to another; excluded rather than guessed."""
        df = self.load_daily()
        counts = df.groupby("study_id")["eureka_id"].nunique()
        unique_study_ids = counts[counts == 1].index
        unique_study_ids = [s for s in unique_study_ids if s != -1]
        m = df[df["study_id"].isin(unique_study_ids)][["study_id", "eureka_id"]].drop_duplicates()
        return dict(zip(m["study_id"], m["eureka_id"], strict=False))

    def hourly_daily_aggregates(self) -> pd.DataFrame:
        """Collapses the hourly table to one row per (eureka_id, date):
        circadian night/day activity ratio, imputation rate (filled_pct)
        mean/std, and total screen/distance -- signal not present in the
        daily table's six-hour epoch aggregates. Cached; only computed
        for subjects with an unambiguous study_id mapping (see
        `_study_to_eureka_map`).

        `filled_pct` is the fraction OF features imputed for that hour,
        per the CrossCheck data dictionary -- so high filled_pct means
        less real sensor data, not more; it is not a data-completeness
        measure."""
        if self._hourly_daily_agg is not None:
            return self._hourly_daily_agg

        if self._hourly_raw is not None:
            hr = self._hourly_raw.copy()
        else:
            path = f"{self.local_root}/CrossCheck_Hourly_Data_v1_08132020.csv"
            hr = pd.read_csv(path)
        s2e = self._study_to_eureka_map()
        hr["study_id"] = hr["study_id"].astype(int)
        hr = hr[hr["study_id"].isin(s2e.keys())].copy()
        hr["eureka_id"] = hr["study_id"].map(s2e)
        hr["timestamp"] = pd.to_datetime(hr["day"])
        hr["date"] = hr["timestamp"].dt.date
        hr["hour_of_day"] = hr["timestamp"].dt.hour

        is_night = hr["hour_of_day"].isin(_HOURLY_NIGHT_HOURS)
        is_day = hr["hour_of_day"].isin(_HOURLY_DAY_HOURS)

        rows = []
        for (eid, date), g in hr.groupby(["eureka_id", "date"]):
            night_mean = g.loc[is_night.loc[g.index], "accel_mean"].mean()
            day_mean = g.loc[is_day.loc[g.index], "accel_mean"].mean()
            rows.append(
                {
                    "eureka_id": eid,
                    "date": pd.Timestamp(date),
                    "hourly_night_day_accel_ratio": (
                        float(night_mean / day_mean)
                        if day_mean and not np.isnan(day_mean) and day_mean > 0
                        else np.nan
                    ),
                    "hourly_filled_pct_mean": float(g["filled_pct"].mean()),
                    "hourly_filled_pct_std": float(g["filled_pct"].std()) if len(g) > 1 else 0.0,
                    # min_count=1: pandas' sum() returns 0.0 for an
                    # all-missing group, which is indistinguishable from a
                    # genuine zero (a day the phone recorded no screen time or
                    # no movement). Those mean opposite things to a model --
                    # "no data" versus "measured, and it was zero" -- so an
                    # all-NaN group must stay NaN and be routed as missing.
                    "hourly_screen_sum_total": float(
                        g["screen_sum_1hr_screen_duration_sum"].sum(min_count=1)
                    ),
                    "hourly_distance_sum_total": float(g["distance_sum"].sum(min_count=1)),
                }
            )
        self._hourly_daily_agg = pd.DataFrame(rows)
        return self._hourly_daily_agg

    def list_subjects(self) -> list[str]:
        """Return subject ids present in the daily table, sorted."""
        return sorted(self.load_daily()["eureka_id"].astype(str).unique().tolist())

    def ema_administration_density(self) -> pd.DataFrame:
        """Counts genuine EMA responses (row
        where ema_score is not NaN) per subject -- every day has ema_*
        columns in the raw file, but only every 2-3 days has a genuine
        answer, so a naive notna().sum() on the wrong basis would overstate
        density. This check decides whether the forecasting task is even answerable
        per subject."""
        df = self.load_daily()
        counts = df.groupby("eureka_id")["ema_score"].apply(lambda s: s.notna().sum())
        totals = df.groupby("eureka_id")["day"].count()
        out = pd.DataFrame({"n_ema_responses": counts, "n_days": totals})
        out["days_per_ema"] = out["n_days"] / out["n_ema_responses"].replace(0, pd.NA)
        return out.reset_index()

    def build_task_table(
        self,
        min_history_days: int = 7,
        use_epoch_features: bool = False,
        use_hourly_features: bool = False,
    ) -> list[CareExample]:
        """One CareExample per (subject, day) where a future genuine EMA
        response exists -- forecasts the next actual administration's
        ema_score, never a forward-filled value (a leakage risk). Passive-sensing features use only rows with date' <= date.
        `prediction_time_t`/`horizon`/target arithmetic use `day_ordinal`
        (real elapsed days since the subject's first row), not the raw
        YYYYMMDD int -- see module docstring."""
        df = self.load_daily()
        hourly_agg = self.hourly_daily_aggregates() if use_hourly_features else None
        examples: list[CareExample] = []

        for sid, g in df.groupby("eureka_id"):
            g = g.sort_values("day_ordinal").reset_index(drop=True)
            ema_ords = g.loc[g["ema_score"].notna(), "day_ordinal"].tolist()
            if len(ema_ords) < 2:
                continue  # not enough genuine EMA responses to forecast anything

            hourly_by_date = None
            if hourly_agg is not None:
                sub = hourly_agg[hourly_agg["eureka_id"] == sid]
                if len(sub):
                    hourly_by_date = sub.set_index("date")

            for i, ordv in enumerate(g["day_ordinal"]):
                if i < min_history_days:
                    continue
                future_ema_ords = [d for d in ema_ords if d > ordv]
                if not future_ema_ords:
                    continue
                target_ord = min(future_ema_ords)
                target_row = g.loc[g["day_ordinal"] == target_ord].iloc[0]

                window = g.iloc[max(0, i - min_history_days + 1) : i + 1]
                feats: dict[str, float] = {}
                for col in PASSIVE_BLOCKS:
                    if col in window.columns:
                        feats[f"{col}_mean"] = float(window[col].mean())
                        feats[f"{col}_last"] = float(window[col].iloc[-1])
                if use_epoch_features:
                    for col in EPOCH_BLOCKS:
                        if col in window.columns:
                            feats[f"{col}_mean"] = float(window[col].mean())
                            feats[f"{col}_last"] = float(window[col].iloc[-1])
                for q in QUALITY_FIELDS:
                    if q in window.columns:
                        feats[f"{q}_mean"] = float(window[q].mean())
                if "sleep_duration" in window.columns:
                    feats["sleep_duration_mean"] = float(window["sleep_duration"].mean())

                feats["weekday_sin"] = float(np.sin(2 * np.pi * window["weekday"].iloc[-1] / 7))
                feats["weekday_cos"] = float(np.cos(2 * np.pi * window["weekday"].iloc[-1] / 7))

                if hourly_by_date is not None:
                    window_dates = [pd.Timestamp(d) for d in window["date"]]
                    matched = hourly_by_date.reindex(window_dates)
                    for col in (
                        "hourly_night_day_accel_ratio",
                        "hourly_filled_pct_mean",
                        "hourly_filled_pct_std",
                        "hourly_screen_sum_total",
                        "hourly_distance_sum_total",
                    ):
                        vals = matched[col].dropna()
                        feats[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")

                past_mask = (g["day_ordinal"] <= ordv) & g["ema_score"].notna()
                past_ema = g.loc[past_mask, "ema_score"]
                feats["last_known_ema_score"] = (
                    float(past_ema.iloc[-1]) if len(past_ema) else float("nan")
                )
                feats["days_since_last_ema"] = (
                    float(ordv - g.loc[past_mask, "day_ordinal"].iloc[-1])
                    if len(past_ema)
                    else float("nan")
                )

                examples.append(
                    CareExample(
                        subject_id=str(sid),
                        prediction_time_t=float(ordv),
                        horizon=float(target_ord - ordv),
                        context_features=feats,
                        target_y=float(target_row["ema_score"]),
                        target_observed_at=float(target_ord),
                        domain_id="crosscheck",
                        provenance="adaptiveCareArena.caredata.crosscheck",
                    )
                )
        return examples

    def validate_temporal_contract(self, examples: list[CareExample]) -> None:
        """Raise ``ValueError`` if any example's target precedes ``t + horizon``."""
        assert_no_future_leakage(examples)
