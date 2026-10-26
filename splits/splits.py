"""Leakage-safe splits. Operates on any object with
.subject_id / .prediction_time_t / .target_y (CareExample satisfies this)."""

from __future__ import annotations

import numpy as np


def leave_subject_out(examples: list, test_subjects: set[str]) -> tuple[list, list]:
    """Split ``examples`` into (train, test) by subject, so no subject is in both."""
    train = [e for e in examples if e.subject_id not in test_subjects]
    test = [e for e in examples if e.subject_id in test_subjects]
    return train, test


def kfold_subject_groups(subjects: list[str], k: int, rng: np.random.Generator) -> list[set[str]]:
    """Partition ``subjects`` into ``k`` disjoint folds, shuffled with ``rng``."""
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    folds: list[set[str]] = [set() for _ in range(k)]
    for i, sid in enumerate(shuffled):
        folds[i % k].add(sid)
    return folds


def future_time_split(
    examples: list,
    train_fraction: float = 0.7,
    group_by_target_event: bool = True,
    embargo: float = 0.0,
) -> tuple[list, list]:
    """Per-subject temporal split with target-event grouping and an
    optional embargo.

    The naive version of this function (order rows by `prediction_time_t`,
    cut at a fraction) leaked: several examples can share one target event
    -- the same EMA answer reached from different prediction times and
    horizons -- so the identical label landed in both train and test. An
    independent review measured it on the real CrossCheck Stage 1 split:
    47 target events, 4.26% of test rows, appeared on both sides. That
    does not merely add noise; it lets a model memorise a specific answer
    and then be scored on that same answer, which inflates the learned
    models relative to persistence and invalidates the reported
    future-time MAE and its CI.

    Two guards, both off by default only in the sense that they are
    parameterised -- `group_by_target_event` defaults to True because the
    unguarded behaviour is a bug, not a legitimate option:

    1. **Event grouping.** The split point is chosen over unique
       `(subject_id, target_observed_at)` events, and every row sharing a
       target event goes entirely to one side. No event can straddle.

    2. **Purge + embargo.** Even after grouping, a training row whose
       target is observed after the test period begins carries future
       information. Such rows are purged. `embargo` widens that exclusion
       backwards by an additional time margin (same units as
       `target_observed_at`), which is what handles targets that are
       correlated across adjacent days rather than merely identical.

    Set `group_by_target_event=False` only to reproduce the older,
    leaky numbers for comparison -- never for a reported result."""
    by_subject: dict[str, list] = {}
    for e in examples:
        by_subject.setdefault(e.subject_id, []).append(e)

    train, test = [], []
    for _sid, exs in by_subject.items():
        if not group_by_target_event:
            exs_sorted = sorted(exs, key=lambda e: e.prediction_time_t)
            cutoff = max(1, int(round(len(exs_sorted) * train_fraction)))
            cutoff = min(cutoff, len(exs_sorted) - 1) if len(exs_sorted) > 1 else len(exs_sorted)
            train.extend(exs_sorted[:cutoff])
            test.extend(exs_sorted[cutoff:])
            continue

        events = sorted({e.target_observed_at for e in exs})
        if len(events) < 2:
            # A single target event cannot be split without putting the
            # same label on both sides -- assign it wholly to train.
            train.extend(exs)
            continue

        cutoff = max(1, int(round(len(events) * train_fraction)))
        cutoff = min(cutoff, len(events) - 1)
        test_start = events[cutoff]
        train_events = set(events[:cutoff])

        for e in exs:
            if e.target_observed_at in train_events:
                # Purge + embargo: a training row whose target is observed
                # at or after (test_start - embargo) is too close to the
                # evaluation period to be safe.
                if e.target_observed_at < test_start - embargo:
                    train.append(e)
            else:
                test.append(e)

    return train, test


def count_shared_target_events(train: list, test: list) -> int:
    """Number of `(subject_id, target_observed_at)` events appearing on
    both sides of a split -- the leakage metric that made the original
    `future_time_split` defect measurable rather than theoretical. Must be
    0 for any split backing a reported number."""
    train_events = {(e.subject_id, e.target_observed_at) for e in train}
    test_events = {(e.subject_id, e.target_observed_at) for e in test}
    return len(train_events & test_events)
