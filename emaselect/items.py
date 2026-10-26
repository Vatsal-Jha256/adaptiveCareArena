"""The 10 CrossCheck EMA items, and what asking each one costs the patient.

This is the local half of the dual-scarcity problem: patient attention is
scarce, so which questions are worth asking? It is a different question
from whether passive sensing can forecast the next EMA, and does not
depend on the answer to that.

Verified directly against the released daily table (6,364 events with all
ten items present, 2026-09-03):

    ema_score = sum(POSITIVE_ITEMS) - sum(NEGATIVE_ITEMS)

exactly, with every item on a 0-3 scale. That exactness is what makes the
benchmark well-posed rather than circular: asking all ten reconstructs
the score with zero error at maximum burden, asking none leaves only
passive sensing at zero burden, and the science is the shape of the curve
between those two fixed endpoints."""

from __future__ import annotations

# Higher = better (well-being direction).
POSITIVE_ITEMS = ("ema_CALM", "ema_HOPEFUL", "ema_SLEEPING", "ema_SOCIAL", "ema_THINK")
# Higher = worse (symptom direction).
NEGATIVE_ITEMS = ("ema_DEPRESSED", "ema_HARM", "ema_SEEING_THINGS", "ema_STRESSED", "ema_VOICES")

EMA_ITEMS = tuple(sorted(POSITIVE_ITEMS + NEGATIVE_ITEMS))
ITEM_SIGN = {**{i: +1 for i in POSITIVE_ITEMS}, **{i: -1 for i in NEGATIVE_ITEMS}}

# --- burden profiles -------------------------------------------------
# Costs are relative and are experimental conditions, not measured
# patient burden. Nobody has measured the seconds or distress each of
# these questions costs; claiming otherwise would be inventing data. The
# requirement, as with any cost-sensitive acquisition study, is that conclusions hold across several plausible profiles.

# Every question costs the same -- the neutral default.
UNIFORM = {item: 1.0 for item in EMA_ITEMS}

# Sensitivity-weighted: asking someone about self-harm, hallucinations or
# voices is not equivalent to asking whether they slept. These items are
# the ones a clinician would not put on a daily automated prompt without
# thought, and an acquisition policy that reaches for them freely should
# have to pay for it. The ordering here is a modelling assumption stated
# in the open, not a clinical finding.
SENSITIVITY_WEIGHTED = {
    "ema_SLEEPING": 0.5,
    "ema_SOCIAL": 0.6,
    "ema_CALM": 0.7,
    "ema_HOPEFUL": 0.8,
    "ema_THINK": 1.0,
    "ema_STRESSED": 1.0,
    "ema_DEPRESSED": 1.5,
    "ema_SEEING_THINGS": 2.0,
    "ema_VOICES": 2.0,
    "ema_HARM": 3.0,
}

# Inverted sensitivity: a deliberate adversarial control. If a policy's
# ranking is driven by cost rather than by information, its behaviour
# should change sharply here; if conclusions survive both this and
# SENSITIVITY_WEIGHTED, they are not artifacts of one cost ordering.
INVERTED_SENSITIVITY = {item: round(3.5 - SENSITIVITY_WEIGHTED[item], 2) for item in EMA_ITEMS}

BURDEN_PROFILES = {
    "uniform": UNIFORM,
    "sensitivity_weighted": SENSITIVITY_WEIGHTED,
    "inverted_sensitivity": INVERTED_SENSITIVITY,
}


def ema_score_from_items(item_values: dict) -> float:
    """The released scoring rule, reproduced exactly."""
    return float(sum(ITEM_SIGN[i] * item_values[i] for i in EMA_ITEMS))
