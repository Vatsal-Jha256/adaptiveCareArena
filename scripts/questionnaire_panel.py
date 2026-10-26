"""Load complete questionnaire reports for composed benchmark tasks."""

from __future__ import annotations

import numpy as np

from emaselect.ces_events import PHQ4_ITEMS, build_ces_events
from emaselect.dataset import build_ema_events
from emaselect.items import EMA_ITEMS, ITEM_SIGN

MIN_REPORTS = 20
RELATIVE_HORIZON = 270


def load_questionnaire_panel(
    dataset: str,
    align: str = "relative",
    local_root: str | None = None,
) -> tuple[dict, tuple, float]:
    if dataset == "crosscheck":
        events = build_ema_events(
            local_root=local_root or "data/raw/crosscheck", target_mode="reconstruct"
        )
        items = tuple(EMA_ITEMS)
        signs = ITEM_SIGN
        default_threshold = -5.0
    elif dataset == "ces":
        events = build_ces_events(
            local_root=local_root or "data/raw/ces",
            target_mode="reconstruct",
            min_ema=MIN_REPORTS,
        )
        items = tuple(PHQ4_ITEMS)
        signs = {item: -1 for item in items}
        default_threshold = -9.0
    else:
        raise ValueError(f"unsupported questionnaire panel: {dataset}")

    grouped: dict = {}
    for event in events:
        grouped.setdefault(event.subject_id, []).append(event)
    global_start = min(event.day_ordinal for event in events)
    panel = {}
    for subject, subject_events in grouped.items():
        subject_events = sorted(subject_events, key=lambda event: event.day_ordinal)
        if len(subject_events) < MIN_REPORTS:
            continue
        start = subject_events[0].day_ordinal if align == "relative" else global_start
        rows = [
            (
                int(event.day_ordinal - start),
                np.array([signs[item] * event.item_values[item] for item in items], float),
            )
            for event in subject_events
            if align == "calendar" or event.day_ordinal - start < RELATIVE_HORIZON
        ]
        if len(rows) >= MIN_REPORTS:
            panel[subject] = rows
    return panel, items, default_threshold
