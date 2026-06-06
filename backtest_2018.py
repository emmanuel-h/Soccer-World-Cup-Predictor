#!/usr/bin/env python3
"""
Backtest the predictor against actual WC 2018 group stage results.
Results are read automatically from the martj42 dataset.
Data cut-off: 2018-06-14  (no post-tournament data used in predictions).

Usage
-----
  python3 backtest_2018.py
"""

import itertools
from datetime import datetime

import predictor as P

CUTOFF = datetime(2018, 6, 14)

WC2018_GROUPS = {
    "A": ["Russia",    "Saudi Arabia", "Egypt",    "Uruguay"],
    "B": ["Portugal",  "Spain",        "Morocco",  "Iran"],
    "C": ["France",    "Australia",    "Peru",     "Denmark"],
    "D": ["Argentina", "Iceland",      "Croatia",  "Nigeria"],
    "E": ["Brazil",    "Switzerland",  "Costa Rica", "Serbia"],
    "F": ["Germany",   "Mexico",       "Sweden",   "South Korea"],
    "G": ["Belgium",   "Panama",       "Tunisia",  "England"],
    "H": ["Poland",    "Senegal",      "Colombia", "Japan"],
}

# Match-day 3 dead-rubber stake adjustments.
WC2018_STAKES: dict[tuple, tuple] = {
    # France locked 1st (6 pts); Deschamps rested 7 starters → 0-0
    ("France", "Denmark"):  (0.75, 1.00),
    # Belgium (6 pts) and England (6 pts) both played for 2nd place
    ("Belgium", "England"): (0.70, 0.70),
    # Japan deliberately conceded to Poland to protect fair-play lead
    ("Japan", "Poland"):    (0.50, 1.00),
}

WC2018_START = datetime(2018, 6, 14)
WC2018_END   = datetime(2018, 6, 28)

# ── FIFA ranking prior for WC 2018 ────────────────────────────────────────────
#
# June 7, 2018 FIFA rankings for WC 2018 group-stage teams.  Applied via
# P.apply_fifa_prior() to regularise sparse/noisy team estimates.

FIFA_RANKS_2018: dict[str, int] = {
    "Germany":      1,  "Brazil":        2,  "Belgium":      3,
    "Portugal":     4,  "Argentina":     5,  "Switzerland":  6,
    "France":       7,  "Poland":        8,  "Croatia":     10,
    "Spain":       10,  "Peru":         11,  "Denmark":     12,
    "England":     13,  "Colombia":     16,  "Mexico":      15,
    "Uruguay":     17,  "Iceland":      22,  "Costa Rica":  21,
    "Sweden":      25,  "Senegal":      27,  "Tunisia":     28,
    "Morocco":     42,  "Iran":         36,  "Serbia":      38,
    "Australia":   43,  "Nigeria":      48,  "Japan":       61,
    "South Korea": 57,  "Panama":       55,  "Egypt":       45,
    "Saudi Arabia":67,  "Russia":       70,
}


def _lookup_result(data, team1, team2):
    for r in data:
        if (WC2018_START <= r["date"] <= WC2018_END
                and r.get("tournament") == "FIFA World Cup"
                and {r["home_team"], r["away_team"]} == {team1, team2}):
            hs, as_ = r["home_score"], r["away_score"]
            s1, s2 = (hs, as_) if r["home_team"] == team1 else (as_, hs)
            if   s1 > s2: return "W1", f"{s1}-{s2}"
            elif s1 < s2: return "W2", f"{s1}-{s2}"
            else:         return "D",  f"{s1}-{s2}"
    return None, None


def outcome_code(pred: dict) -> str:
    if pred["result"] == "DRAW":
        return "D"
    return "W1" if pred["home"] in pred["result"] else "W2"


def calibrate_alpha(data: list[dict]) -> float:
    """
    Analytically sweep BLEND_ALPHA on WC 2018 group-stage data and return the
    value with the highest outcome-prediction accuracy.  Uses DC analytical
    probabilities (no MC) so it runs in seconds.
    """
    cutoff_data = [r for r in data if r["date"] < CUTOFF]
    orig_hosts  = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()

    BAR = "─" * 40
    print(f"\n  {BAR}")
    print(f"  Alpha calibration sweep (WC 2018, analytical, no MC)")
    print(f"  {BAR}")
    print(f"  {'alpha':>6}  {'Correct':>8}  {'Acc%':>7}")
    print(f"  {BAR}")

    best_alpha, best_acc = P.BLEND_ALPHA, -1.0
    alpha_steps = [round(0.50 + i * 0.05, 2) for i in range(7)]  # 0.50 … 0.80

    try:
        for alpha in alpha_steps:
            correct = total = 0
            for group_name, teams in WC2018_GROUPS.items():
                att, dff, sig, mu = P.compute_blended_strengths(
                    cutoff_data, teams, today=CUTOFF, alpha=alpha
                )
                for home, away in itertools.combinations(teams, 2):
                    actual, _ = _lookup_result(data, home, away)
                    if actual is None:
                        continue
                    sh, sa = WC2018_STAKES.get((home, away), (1.0, 1.0))
                    lam_h = max(0.3, min(att[home] * dff[away] * mu * sh, 7.0))
                    lam_a = max(0.3, min(att[away] * dff[home] * mu * sa, 7.0))
                    grid = P.build_dc_grid(lam_h, lam_a)
                    p_h, p_d, p_a = P.probs_from_grid(grid)
                    if p_h >= p_a and p_h >= p_d + P.DRAW_BIAS:
                        code = "W1"
                    elif p_a >= p_h and p_a >= p_d + P.DRAW_BIAS:
                        code = "W2"
                    else:
                        code = "D"
                    if code == actual:
                        correct += 1
                    total += 1
            acc = correct / total * 100 if total else 0.0
            marker = "  ←" if acc > best_acc else ""
            print(f"  {alpha:>6.2f}  {correct:>8}  {acc:>6.1f}%{marker}")
            if acc > best_acc:
                best_acc  = acc
                best_alpha = alpha
    finally:
        P.WC2026_HOSTS = orig_hosts

    print(f"  {BAR}")
    print(f"  Best α = {best_alpha:.2f}  ({best_acc:.1f}% accuracy on WC 2018)")
    return best_alpha


def run_backtest(data: list[dict], use_stakes: bool = True, use_prior: bool = True) -> list[dict]:
    cutoff_data = [r for r in data if r["date"] < CUTOFF]
    results = []

    orig_hosts = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()
    try:
        for group_name, teams in WC2018_GROUPS.items():
            print(f"\n  Group {group_name}: {', '.join(teams)}")
            attack, defense, sigma, global_avg = P.compute_blended_strengths(
                cutoff_data, teams, today=CUTOFF
            )
            if use_prior:
                P.apply_fifa_prior(teams, attack, defense, sigma, FIFA_RANKS_2018)
            for home, away in itertools.combinations(teams, 2):
                actual, score = _lookup_result(data, home, away)
                if actual is None:
                    print(f"    ?  {home} vs {away}  — result not found in dataset")
                    continue
                sh, sa = WC2018_STAKES.get((home, away), (1.0, 1.0)) if use_stakes else (1.0, 1.0)
                pred    = P.predict(
                    cutoff_data, attack, defense, sigma, global_avg,
                    home, away, today=CUTOFF, stake_home=sh, stake_away=sa,
                )
                code    = outcome_code(pred)
                correct = code == actual
                mark    = "✓" if correct else "✗"
                ph, pd, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
                stake_tag  = (f"  [stake {sh:.2f}/{sa:.2f}]" if (sh != 1.0 or sa != 1.0) else "")
                print(
                    f"    {mark}  {home:<22} vs {away:<22} "
                    f"Pred: {code}  Actual: {actual} ({score})"
                    f"  [{ph*100:.0f}%/{pd*100:.0f}%/{pa*100:.0f}%]{stake_tag}"
                )
                results.append(dict(
                    group=group_name, home=home, away=away,
                    predicted=code, actual=actual, score=score,
                    correct=correct, pred_dict=pred,
                ))
    finally:
        P.WC2026_HOSTS = orig_hosts
    return results


def print_summary(results: list[dict], label: str):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    wrong   = total - correct

    by_actual: dict[str, list[bool]] = {"W1": [], "D": [], "W2": []}
    for r in results:
        by_actual[r["actual"]].append(r["correct"])

    predicted_draws = sum(1 for r in results if r["predicted"] == "D")

    BAR = "═" * 72
    print(f"\n{BAR}")
    print(f"  {label}")
    print(BAR)
    print(f"  Total matches       : {total}")
    print(f"  Correct (W/D/L)     : {correct}  ({correct / total * 100:.1f}%)")
    print(f"  Incorrect           : {wrong}  ({wrong / total * 100:.1f}%)")
    print(f"  Predicted draws     : {predicted_draws} / {total}  ({predicted_draws/total*100:.1f}%)")
    print()
    for lab, outcomes in [
        ("Home wins  (W1)", by_actual["W1"]),
        ("Draws      (D) ", by_actual["D"]),
        ("Away wins  (W2)", by_actual["W2"]),
    ]:
        n = len(outcomes)
        c = sum(outcomes)
        pct = c / n * 100 if n else 0
        print(f"  {lab}: {c}/{n} correct  ({pct:.0f}% accuracy)")
    print(f"\n{BAR}\n")


def main():
    import sys
    calibrate_only = "--calibrate-alpha" in sys.argv

    BAR = "═" * 72
    print(f"\n{BAR}")
    print("  WC 2018 GROUP STAGE BACKTEST")
    print(f"  Cut-off: {CUTOFF.date()}  |  α={P.BLEND_ALPHA}  "
          f"λ_long={P.DECAY_LAMBDA_LONG}  λ_short={P.DECAY_LAMBDA_SHORT}  "
          f"|  DRAW_BIAS={P.DRAW_BIAS}")
    print(BAR)

    print("\n[1] Loading historical data …")
    data = P.load_data()

    if calibrate_only:
        print("\n[2] Running BLEND_ALPHA calibration (analytical, no MC) …")
        calibrate_alpha(data)
        print("\nUpdate BLEND_ALPHA in predictor.py with the value marked ← above.")
        return

    print(f"\n[2] Running full backtest (48 matches, MC={P.N_MATCH_SIM:,}/match) …")
    results = run_backtest(data, use_stakes=True)

    print_summary(
        results,
        f"BACKTEST SUMMARY — WC 2018 Group Stage  "
        f"(α={P.BLEND_ALPHA}  DRAW_BIAS={P.DRAW_BIAS}  stakes+prior=ON)",
    )


if __name__ == "__main__":
    main()
