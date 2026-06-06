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


def run_backtest(data: list[dict], use_stakes: bool = True) -> list[dict]:
    cutoff_data = [r for r in data if r["date"] < CUTOFF]
    results = []

    orig_hosts = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()
    try:
        for group_name, teams in WC2018_GROUPS.items():
            print(f"\n  Group {group_name}: {', '.join(teams)}")
            attack, defense, sigma, global_avg = P.compute_strengths(
                cutoff_data, teams, today=CUTOFF
            )
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
    BAR = "═" * 72
    print(f"\n{BAR}")
    print("  WC 2018 GROUP STAGE BACKTEST")
    print(f"  Cut-off: {CUTOFF.date()}  |  λ={P.DECAY_LAMBDA}  "
          f"half-life≈{round(0.693/P.DECAY_LAMBDA/365.25,1)}y  |  DRAW_BIAS={P.DRAW_BIAS}")
    print(BAR)

    print("\n[1] Loading historical data …")
    data = P.load_data()

    print(f"\n[2] Running full backtest (48 matches, MC={P.N_MATCH_SIM:,}/match) …")
    results = run_backtest(data, use_stakes=True)

    print_summary(
        results,
        f"BACKTEST SUMMARY — WC 2018 Group Stage  "
        f"(λ={P.DECAY_LAMBDA}  DRAW_BIAS={P.DRAW_BIAS}  stakes ON  prior OFF)",
    )


if __name__ == "__main__":
    main()
