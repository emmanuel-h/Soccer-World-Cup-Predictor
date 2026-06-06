#!/usr/bin/env python3
"""
Sweep draw-classifier threshold over WC 2018 and WC 2022 to find the best
calibration.  All match probabilities and classifier scores are computed
analytically (no MC), so the sweep is fast.

Usage: python3 calibrate_draw_clf.py
"""

import itertools
from datetime import datetime

import predictor as P
import backtest_2022 as B22
import backtest_2018 as B18


# ── Analytical match-data collector ───────────────────────────────────────────

def _collect(data, cutoff, groups, stakes, coeffs, apply_prior=None):
    """
    Return a list of dicts, one per group-stage fixture:
      home, away, p_h, p_d, p_a, clf_p, actual
    Uses DC analytical probs only (no MC).
    """
    cutoff_data = [r for r in data if r["date"] < cutoff]
    matches = []
    orig = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()   # neutral venue
    try:
        for group_name, teams in groups.items():
            att, dfn, sig, mu = P.compute_strengths(cutoff_data, teams, today=cutoff)
            if apply_prior:
                apply_prior(teams, att, dfn, sig)
            for home, away in itertools.combinations(teams, 2):
                sh, sa = stakes.get((home, away), (1.0, 1.0))
                eff = dict(att)
                if sh != 1.0: eff[home] = att[home] * sh
                if sa != 1.0: eff[away] = att[away] * sa
                lh = max(0.3, min(eff[home] * dfn[away] * mu, 7.0))
                la = max(0.3, min(eff[away] * dfn[home] * mu, 7.0))
                ph, pd, pa = P.probs_from_grid(P.build_dc_grid(lh, la))
                clf_p = P.draw_clf_prob(lh, la, pd, coeffs)
                # Look up actual result
                if groups is B22.WC2022_GROUPS:
                    res = B22.WC2022_RESULTS.get((home, away))
                    if res is None:
                        continue
                    actual = res[0]
                else:
                    actual, _ = B18._lookup_result(data, home, away)
                    if actual is None:
                        continue
                matches.append(dict(home=home, away=away,
                                    p_h=ph, p_d=pd, p_a=pa, clf_p=clf_p,
                                    actual=actual))
    finally:
        P.WC2026_HOSTS = orig
    return matches


# ── Threshold evaluator (no MC) ────────────────────────────────────────────────

def _evaluate(matches, threshold, draw_bias=P.DRAW_BIAS):
    correct = pred_d = 0
    by: dict[str, list[bool]] = {"W1": [], "D": [], "W2": []}
    for m in matches:
        ph, pd, pa, cp = m["p_h"], m["p_d"], m["p_a"], m["clf_p"]
        if cp >= threshold:
            pred = "D"
        elif ph >= pa and ph >= pd + draw_bias:
            pred = "W1"
        elif pa >= ph and pa >= pd + draw_bias:
            pred = "W2"
        else:
            pred = "D"
        if pred == "D":
            pred_d += 1
        hit = (pred == m["actual"])
        correct += hit
        by[m["actual"]].append(hit)
    return correct, pred_d, by


# ── Main sweep ─────────────────────────────────────────────────────────────────

def main():
    CUTOFF_22 = datetime(2022, 11, 19)
    CUTOFF_18 = datetime(2018, 6, 14)

    BAR = "═" * 88
    print(f"\n{BAR}")
    print("  DRAW CLASSIFIER — THRESHOLD CALIBRATION SWEEP")
    print(BAR)

    print("\n[1] Loading data …")
    data = P.load_data()

    print("[2] Training classifiers (once per cutoff) …")
    coeffs_22, _ = P.fit_draw_classifier(data, today=CUTOFF_22)
    coeffs_18, _ = P.fit_draw_classifier(data, today=CUTOFF_18)
    print(f"  WC 2022 β = {[round(b,3) for b in coeffs_22]}")
    print(f"  WC 2018 β = {[round(b,3) for b in coeffs_18]}")

    print("[3] Pre-computing analytical match data …")
    m22 = _collect(data, CUTOFF_22, B22.WC2022_GROUPS, B22.WC2022_STAKES,
                   coeffs_22, apply_prior=B22._apply_fifa_prior)
    m18 = _collect(data, CUTOFF_18, B18.WC2018_GROUPS, B18.WC2018_STAKES,
                   coeffs_18, apply_prior=None)

    nd22 = sum(1 for m in m22 if m["actual"] == "D")
    nd18 = sum(1 for m in m18 if m["actual"] == "D")
    N = len(m22) + len(m18)

    # clf score percentiles for context
    scores_22 = sorted(m["clf_p"] for m in m22)
    scores_18 = sorted(m["clf_p"] for m in m18)
    p = lambda s, q: s[int(len(s)*q)]
    print(f"\n  WC 2022 clf scores: "
          f"p25={p(scores_22,.25):.3f}  med={p(scores_22,.50):.3f}  "
          f"p75={p(scores_22,.75):.3f}  max={scores_22[-1]:.3f}")
    print(f"  WC 2018 clf scores: "
          f"p25={p(scores_18,.25):.3f}  med={p(scores_18,.50):.3f}  "
          f"p75={p(scores_18,.75):.3f}  max={scores_18[-1]:.3f}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    thresholds = [round(0.19 + 0.01*i, 2) for i in range(20)]  # 0.19 → 0.38
    thresholds.append(float("inf"))   # no-classifier baseline

    print(f"\n{BAR}")
    print(f"  {'Thresh':>8}  "
          f"{'── WC 2022 (n=48, draws=' + str(nd22) + ') ──':^30}  "
          f"{'── WC 2018 (n=48, draws=' + str(nd18) + ') ──':^30}  "
          f"{'Comb':>6}")
    print(f"  {'':>8}  "
          f"{'Acc':>7} {'D%':>5} {'D acc':>7}  "
          f"{'Acc':>7} {'D%':>5} {'D acc':>7}  "
          f"{'Acc':>8}")
    print(f"  {'─'*84}")

    rows = []
    best_combined = -1
    for t in thresholds:
        c22, pd22, by22 = _evaluate(m22, t)
        c18, pd18, by18 = _evaluate(m18, t)
        comb = c22 + c18
        da22 = sum(by22["D"]) / len(by22["D"]) if by22["D"] else 0
        da18 = sum(by18["D"]) / len(by18["D"]) if by18["D"] else 0
        rows.append((t, c22, pd22, da22, c18, pd18, da18, comb))
        if comb > best_combined:
            best_combined = comb

    for (t, c22, pd22, da22, c18, pd18, da18, comb) in rows:
        tlabel = f"{t:.2f}" if t < 99 else "  ∞"
        mark = " ◄" if comb == best_combined else ""
        print(
            f"  {tlabel:>8}  "
            f"{c22/48*100:>6.1f}% {pd22/48*100:>4.0f}%  {da22*100:>6.0f}%  "
            f"  {c18/48*100:>6.1f}% {pd18/48*100:>4.0f}%  {da18*100:>6.0f}%  "
            f"  {comb}/{N}={comb/N*100:>4.1f}%{mark}"
        )

    print(f"  {'─'*84}")

    # ── Recommendation ────────────────────────────────────────────────────────
    # Best combined; tie-break: prefer fewer predicted draws (less aggressive)
    best_rows = [(t, c22, pd22, da22, c18, pd18, da18, comb)
                 for (t, c22, pd22, da22, c18, pd18, da18, comb) in rows
                 if comb == best_combined]
    best_t, *_, best_comb = max(best_rows, key=lambda r: r[0])   # highest threshold wins tie

    print(f"\n  Best combined: threshold = {best_t:.2f}"
          f"   ({best_comb}/{N} = {best_comb/N*100:.1f}%)")

    # Show the corresponding training-data target_draw_rate
    # (fraction of training test-set scores that would be above best_t)
    for coeffs, cutoff, label in [(coeffs_22, CUTOFF_22, "WC 2022"),
                                   (coeffs_18, CUTOFF_18, "WC 2018")]:
        cutoff_data = [r for r in data if r["date"] < cutoff]
        from datetime import timedelta
        td_cutoff = cutoff - timedelta(days=P._DRAW_CLF_YEARS * 365.25)
        train = [r for r in data
                 if r["date"] >= td_cutoff and r["date"] < cutoff
                 and P._tournament_weight(r.get("tournament","")) >= P._DRAW_CLF_MIN_TW]
        orig = P.WC2026_HOSTS; P.WC2026_HOSTS = set()
        try:
            teams = list({t for r in train for t in (r["home_team"], r["away_team"])})
            att, dfn, _, mu = P.compute_strengths(data, teams, today=cutoff)
        finally:
            P.WC2026_HOSTS = orig
        ts = []
        for r in train:
            ht, at = r["home_team"], r["away_team"]
            lh = max(0.3, min(att.get(ht,1)*dfn.get(at,1)*mu, 7))
            la = max(0.3, min(att.get(at,1)*dfn.get(ht,1)*mu, 7))
            _, pd, _ = P.probs_from_grid(P.build_dc_grid(lh, la))
            ts.append(P.draw_clf_prob(lh, la, pd, coeffs))
        rate = sum(1 for s in ts if s >= best_t) / len(ts)
        print(f"  {label}: best_t={best_t:.2f} corresponds to "
              f"target_draw_rate ≈ {rate:.3f}  (training N={len(ts)})")

    print(f"{BAR}\n")


if __name__ == "__main__":
    main()
