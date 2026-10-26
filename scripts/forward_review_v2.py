"""Forward-looking review task, v2.2 (after three adversarial code reviews).

AT-risk restriction (new, default on; --all-states reverses it): a landmark is a candidate only if the
patient's latest report (as the belief sees it) is above tau. Onset is a transition out of the non-severe
state, so patients already severe are not at risk of it; leaving them in the queue lets need-based scores
spend their top slots on patients who cannot have an onset (this is why last report fell below random on
CES before the restriction). Output files carry the suffix _ar.

Candidate (risk) sets, selectable with --risk:
  report     : a landmark exists only when a report arrives. Membership is therefore observable at
               decision time and never depends on a future report or a future retention endpoint.
               A no-report future window has no observed onset; this estimates report-landmark
               detection, not latent deterioration during missingness.
  observable : (i, r) is a candidate only if i has >= MIN_HIST reports through r and >= 1 report in
               (r, r+h]  (labels always observed). Note this uses future attendance to define who is in
               the queue, so it is an outcome-observable estimand, not an operational one.
  active     : any (i, r) with >= MIN_HIST reports through r and r <= i's last report day (the last
               observed report is not assumed to be a discharge date); windows with no report are
               labelled non-onset.
  all        : any (i, r) with >= MIN_HIST reports through r through the panel horizon; no-report
               windows are non-onset.
Episode = first report <= tau in (r, r+h] by a patient whose latest report through r is > tau
(one episode per (subject, first-crossing day)). Episode recall = share of episodes captured on any
eligible landmark within h days before it. Daily queues are fixed per model (deterministic shared
tie-break keys). Observed paired differences come from the data; bootstrap replicates (subject
clusters, queue fixed) give percentile intervals only. Random baseline: mean over 30 random queues of
that queue's episode recall (always in [0,1]).

Score-level diagnostics (candidate-day AUC for onset), all with subject-cluster bootstrap CIs:
  auc_exact     strata = exact last report (integer scale)
  auc_exact_ep  same, positives weighted 1/(rows in their episode) (episode-balanced positive-landmark
                weights; an episode's total influence still depends on its strata's negatives)
  auc_subj      strata = (subject, exact last report)  (within-person updating)

python -m scripts.forward_review_v2 --dataset crosscheck --h 3 --risk observable
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import yaml

from stage2_env.replay_experiment import load_panel

MIN_HIST = 6
BUDGETS = (3, 5, 10)
FRACS = (0.05, 0.1, 0.2)
KAL = {"crosscheck": (0.02, 0.25), "ces": None}
MODELS = ("last", "ema3", "ema6", "ema15", "kal", "z", "pool")


SALT = 0


def jitter(i, r, seed=None):
    seed = SALT if seed is None else seed
    x = (i * 2654435761 + r * 40503 + seed * 97) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    return (x % 1000003) / 1000003 * 1e-9


def build(traj, tau, h, kal, truth=None, risk="observable", at_risk=True):
    """traj: values the belief sees; truth: values used only to define onsets (default traj)."""
    truth = traj if truth is None else truth
    n_days = max(len(t) for t in traj.values())
    rep = {i: [(d, v) for d, v in enumerate(t) if v is not None] for i, t in traj.items()}
    rep_days = {i: np.array([d for d, _ in rl]) for i, rl in rep.items()}
    rep_truth = {i: np.array([truth[i][d] for d, _ in rl]) for i, rl in rep.items()}
    by_day = {}
    for rl in rep.values():
        for d, v in rl:
            by_day.setdefault(d, []).append(v)
    state = {
        i: dict(
            k=0,
            ema3=None,
            ema6=None,
            ema15=None,
            last=None,
            lasttruth=None,
            kx=None,
            kp=None,
            hist=[],
            lastday=-1,
        )
        for i in rep
    }
    rows, pool = [], []
    raw_eps = set()
    for i, rl in rep.items():  # raw episodes: crossing reports after >= MIN_HIST prior reports
        for j in range(MIN_HIST, len(rl)):
            if rep_truth[i][j - 1] > tau and rep_truth[i][j] <= tau:
                raw_eps.add((i, int(rep_days[i][j])))
    for r in range(n_days):
        for i in rep:
            s = state[i]
            dd = rep_days[i]
            j = np.searchsorted(dd, r)
            report_today = j < len(dd) and dd[j] == r
            if report_today:
                v = traj[i][r]
                need = -v
                for key, a in (("ema3", 0.3), ("ema6", 0.6), ("ema15", 0.15)):
                    s[key] = need if s[key] is None else a * need + (1 - a) * s[key]
                if kal is not None:
                    q, rr = kal
                    if s["kx"] is None:
                        s["kx"], s["kp"] = need, 1.0
                    else:
                        p = s["kp"] + q
                        g = p / (p + rr)
                        s["kx"] = s["kx"] + g * (need - s["kx"])
                        s["kp"] = (1 - g) * p
                prev = list(s["hist"])
                s["hist"].append(need)
                s["z_prev"] = (np.mean(prev), np.std(prev) + 1e-6) if len(prev) >= 3 else None
                s["last"], s["lasttruth"], s["lastday"] = v, truth[i][r], r
                s["k"] += 1
            if s["k"] < MIN_HIST:
                continue
            lo = np.searchsorted(dd, r, side="right")
            hi = np.searchsorted(dd, r + h, side="right")
            if at_risk and s["last"] <= tau:
                continue  # onset risk set: patients not already in the severe state (as the reviewer sees it)
            if risk == "report" and not report_today:
                continue
            if risk == "observable" and hi <= lo:
                continue
            if risk == "active" and r > dd[-1]:
                continue
            win = rep_truth[i][lo:hi]
            sev = np.where(win <= tau)[0]
            onset = int(s["lasttruth"] > tau and len(sev) > 0)
            ep = (i, int(dd[lo + sev[0]])) if onset else None
            z = (-s["last"] - s["z_prev"][0]) / s["z_prev"][1] if s["z_prev"] else 0.0
            pct = bisect.bisect_left(pool, -s["last"]) / len(pool) if len(pool) > 20 else 0.5
            rows.append(
                dict(
                    i=i,
                    r=r,
                    last=-s["last"],
                    lastraw=s["last"],
                    ema3=s["ema3"],
                    ema6=s["ema6"],
                    ema15=s["ema15"],
                    kal=s["kx"] if kal is not None else s["ema3"],
                    z=z,
                    pool=pct,
                    onset=onset,
                    ep=ep,
                )
            )
        for v in by_day.get(r, []):
            bisect.insort(pool, -v)
    return rows, raw_eps


def queues(rows, B, seed=0, n_random=30):
    by_r = {}
    for k, row in enumerate(rows):
        by_r.setdefault(row["r"], []).append(k)
    res = {m: set() for m in MODELS}
    pair = {m: 0 for m in MODELS}
    rnd_sets = [set() for _ in range(n_random)]
    rr = np.random.default_rng(seed)
    for idx in by_r.values():
        sub = [rows[k] for k in idx]
        jit = np.array([jitter(x["i"], x["r"]) for x in sub])
        Bd = (
            B if isinstance(B, int) else max(1, int(np.ceil(B * len(sub))))
        )  # float B = fraction of the day's candidates
        for m in MODELS:
            sc = np.array([x[m] for x in sub]) + jit
            for t in np.argsort(-sc)[:Bd]:
                if sub[t]["onset"]:
                    res[m].add(sub[t]["ep"])
                    pair[m] += 1
        for rep in range(n_random):
            for t in rr.permutation(len(sub))[:Bd]:
                if sub[t]["onset"]:
                    rnd_sets[rep].add(sub[t]["ep"])
    return res, pair, rnd_sets


class StratAUC:
    """Precomputed pos-vs-neg comparison matrices inside strata; weights by subject multiplicity."""

    def __init__(self, rows, key, pos_w=None, within=False):
        """within=True: strata are inside one subject, so a subject drawn m times contributes its
        comparisons with weight m (not m^2)."""
        self.mats = []
        self.within = within
        groups = {}
        for k, x in enumerate(rows):
            groups.setdefault(key(x), []).append(k)
        s = np.array([x["_s"] for x in rows])
        y = np.array([x["onset"] for x in rows])
        subj = np.array([x["i"] for x in rows])
        for idx in groups.values():
            idx = np.array(idx)
            p, n = idx[y[idx] == 1], idx[y[idx] == 0]
            if len(p) == 0 or len(n) == 0:
                continue
            C = (s[p][:, None] > s[n][None, :]).astype(float) + 0.5 * (
                s[p][:, None] == s[n][None, :]
            )
            wp = np.array([pos_w[k] if pos_w is not None else 1.0 for k in p])
            self.mats.append((C, subj[p], subj[n], wp))

    def value(self, mult):
        num = den = 0.0
        for C, sp, sn, wp in self.mats:
            a = mult[sp] * wp
            b = np.ones_like(mult[sn]) if self.within else mult[sn]
            num += a @ C @ b
            den += a.sum() * b.sum()
        return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument(
        "--risk", default="observable", choices=("report", "observable", "active", "all")
    )
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--tie-salts", type=int, default=5)
    ap.add_argument("--align", default="relative", choices=("relative", "calendar"))
    ap.add_argument(
        "--all-states",
        action="store_true",
        help="include patients already severe in the queue (the v2.1 definition)",
    )
    ap.add_argument(
        "--frac",
        action="store_true",
        help="budgets are fractions of the day's candidates: 5%%, 10%%, 20%%",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="no 270-day horizon truncation (relative alignment only; calendar is always full)",
    )
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f"configs/dev/replay_{a.dataset}.yaml"))
    tau = a.tau if a.tau is not None else cfg["deterioration_threshold"]
    if a.align == "calendar":
        cfg["data"] = dict(cfg["data"], align="calendar", horizon=None)
    elif a.full:
        cfg["data"] = dict(cfg["data"], horizon=None)
    specs, _, traj, _ = load_panel(cfg["data"])
    rows, raw_eps = build(traj, tau, a.h, KAL[a.dataset], risk=a.risk, at_risk=not a.all_states)
    eps = sorted({x["ep"] for x in rows if x["onset"]})
    subj_with_ep = sorted({i for i, _ in eps})
    n_subj = len(traj)
    print(
        f"{a.dataset} tau={tau} h={a.h} risk={a.risk}: candidates={len(rows)}, onset rows={sum(x['onset'] for x in rows)}, "
        f"raw crossing episodes={len(raw_eps)}, eligible episodes={len(eps)} from {len(subj_with_ep)}/{n_subj} subjects"
    )
    out = {
        "dataset": a.dataset,
        "tau": tau,
        "h": a.h,
        "risk": a.risk,
        "align": a.align,
        "full": bool(a.full),
        "at_risk": not a.all_states,
        "candidates": len(rows),
        "onset_rows": sum(x["onset"] for x in rows),
        "raw_episodes": len(raw_eps),
        "episodes": len(eps),
        "subjects_with_episodes": len(subj_with_ep),
        "subjects": n_subj,
        "by_B": {},
    }
    ep_i = np.array([e[0] for e in eps])
    uniq = np.array(sorted(set(ep_i)))
    n_s = np.array([(ep_i == s).sum() for s in uniq])
    rng = np.random.default_rng(0)
    for B in FRACS if a.frac else BUDGETS:
        res, pair, rnd_sets = queues(rows, B)
        cap = {m: np.array([1.0 if e in res[m] else 0.0 for e in eps]) for m in MODELS}
        cap["random"] = np.mean([[1.0 if e in s_ else 0.0 for e in eps] for s_ in rnd_sets], axis=0)
        rec = {m: float(cap[m].mean()) for m in cap}
        assert all(0 <= v <= 1 for v in rec.values()), rec
        per_s = {m: np.array([cap[m][ep_i == s].sum() for s in uniq]) for m in cap}
        cmp_ = [(m, "random") for m in MODELS] + [
            ("ema3", "last"),
            ("kal", "last"),
            ("ema6", "last"),
            ("ema15", "last"),
            ("kal", "ema3"),
        ]
        boots = {c: [] for c in cmp_}
        for _ in range(a.n_boot):
            w = rng.multinomial(len(uniq), np.ones(len(uniq)) / len(uniq))
            den = (w * n_s).sum()
            for m1, m2 in cmp_:
                boots[(m1, m2)].append(((w * per_s[m1]).sum() - (w * per_s[m2]).sum()) / den)
        diffs = {
            f"{m1}-{m2}": [
                float(rec[m1] - rec[m2]),
                float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)),
            ]
            for (m1, m2), v in boots.items()
        }
        out["by_B"][B] = {
            "episode_recall": rec,
            "pair_hits": {m: int(pair[m]) for m in MODELS},
            "diffs": diffs,
        }
        print(B, {m: round(v, 3) for m, v in rec.items()})
        for k in ("ema3-last", "kal-last", "last-random"):
            print("   ", k, "{:+.3f} [{:+.3f},{:+.3f}]".format(*diffs[k]))
    # tie-break sensitivity: observed EMA-last under several deterministic jitter salts
    global SALT
    out["tie_salt_ema3_last"] = {}
    for B in FRACS if a.frac else BUDGETS:
        vals = []
        for salt in range(a.tie_salts):
            SALT = salt
            rs, _, _ = queues(rows, B, n_random=1)
            vals.append(
                float(
                    np.mean([e in rs["ema3"] for e in eps])
                    - np.mean([e in rs["last"] for e in eps])
                )
            )
        out["tie_salt_ema3_last"][B] = [min(vals), max(vals)]
    SALT = 0
    print("tie-salt range ema3-last", out["tie_salt_ema3_last"])
    # score-level diagnostics
    n_pos_in_ep = {}
    for x in rows:
        if x["onset"]:
            n_pos_in_ep[x["ep"]] = n_pos_in_ep.get(x["ep"], 0) + 1
    pos_w = [1.0 / n_pos_in_ep[x["ep"]] if x["onset"] else 1.0 for x in rows]
    out["auc"] = {}
    ids = np.array(sorted({x["i"] for x in rows}))
    idmax = max(ids) + 1
    for key in ("ema3", "kal", "ema15", "z"):
        for x in rows:
            x["_s"] = x[key]
        variants = {
            "exact": StratAUC(rows, lambda x: x["lastraw"]),
            "exact_epbal": StratAUC(rows, lambda x: x["lastraw"], pos_w),
            "subject_exact": StratAUC(rows, lambda x: (x["i"], x["lastraw"]), within=True),
        }
        out["auc"][key] = {}
        for name, sa in variants.items():
            ones = np.ones(idmax)
            pt = sa.value(ones)
            bs = []
            for _ in range(a.n_boot):
                m = np.bincount(
                    rng.choice(ids, size=len(ids), replace=True), minlength=idmax
                ).astype(float)
                bs.append(sa.value(m))
            out["auc"][key][name] = [
                float(pt),
                float(np.nanpercentile(bs, 2.5)),
                float(np.nanpercentile(bs, 97.5)),
            ]
            print("AUC", key, name, "{:.3f} [{:.3f},{:.3f}]".format(*out["auc"][key][name]))
    align_suffix = "" if a.align == "relative" else "_calendar"
    full_suffix = "_full" if (a.full and a.align == "relative") else ""
    frac_suffix = "_frac" if a.frac else ""
    risk_suffix = "" if a.all_states else "_ar"
    output_name = (
        f"forward_v21_{a.dataset}_h{a.h}_tau{abs(tau):g}_{a.risk}"
        f"{align_suffix}{full_suffix}{frac_suffix}{risk_suffix}.json"
    )
    Path("outputs", output_name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
