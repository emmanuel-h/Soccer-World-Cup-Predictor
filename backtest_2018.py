#!/usr/bin/env python3
"""
Backtest the predictor against actual WC 2018 group stage results.
Results are read automatically from the martj42 dataset.
Data cut-off: 2018-06-14  (no post-tournament data used in predictions).

Runs both WITHOUT and WITH the draw classifier to compare their impact.

Usage
-----
  python3 backtest_2018.py
"""

import itertools
from datetime import datetime
from typing import Optional

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
# Only fixtures where documented squad rotation affected line-ups.
WC2018_STAKES: dict[tuple, tuple] = {
    # France locked 1st (6 pts); Deschamps rested 7 starters → 0-0
    ("France", "Denmark"):  (0.75, 1.00),
    # Belgium (6 pts) and England (6 pts) both played for 2nd place to
    # avoid the Germany/Brazil bracket — mutual low-intensity game
    ("Belgium", "England"): (0.70, 0.70),
    # Japan (4 pts, qualified) deliberately conceded late to Poland to
    # preserve fair-play lead over Senegal — controversial but documented
    ("Japan", "Poland"):    (0.50, 1.00),
}

WC2018_START = datetime(2018, 6, 14)
WC2018_END   = datetime(2018, 6, 28)


def _lookup_result(
    data: list[dict], team1: str, team2: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Search the dataset for the WC 2018 group-stage match between team1 and team2.
    Returns (outcome_code, score_str) where:
      outcome_code  "W1" = team1 won, "D" = draw, "W2" = team2 won
      score_str     "<team1_goals>-<team2_goals>"
    """
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


def run_backtest(
    data: list[dict],
    use_stakes: bool = True,
    draw_coeffs: Optional[list[float]] = None,
    draw_threshold: float = P.DRAW_CLF_THRESHOLD,
    label: str = "",
) -> list[dict]:
    cutoff_data = [r for r in data if r["date"] < CUTOFF]
    results = []

    orig_hosts = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()   # WC 2018 is at a neutral venue
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

                sh, sa = (WC2018_STAKES.get((home, away), (1.0, 1.0))
                          if use_stakes else (1.0, 1.0))
                pred = P.predict(
                    cutoff_data, attack, defense, sigma, global_avg,
                    home, away, today=CUTOFF,
                    stake_home=sh, stake_away=sa,
                    draw_coeffs=draw_coeffs, draw_threshold=draw_threshold,
                )
                code    = outcome_code(pred)
                correct = code == actual
                mark    = "✓" if correct else "✗"
                ph, pd, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
                clf_tag = (f"  clf={pred['clf_p_draw']*100:.0f}%"
                           if pred.get("clf_p_draw") is not None else "")
                stake_tag = (f"  [stake {sh:.2f}/{sa:.2f}]"
                             if (sh != 1.0 or sa != 1.0) else "")
                print(
                    f"    {mark}  {home:<22} vs {away:<22} "
                    f"Pred: {code}  Actual: {actual} ({score})"
                    f"  [{ph*100:.0f}%/{pd*100:.0f}%/{pa*100:.0f}%]"
                    f"{clf_tag}{stake_tag}"
                )
                results.append({
                    "group":     group_name,
                    "home":      home,
                    "away":      away,
                    "predicted": code,
                    "actual":    actual,
                    "score":     score,
                    "correct":   correct,
                    "pred_dict": pred,
                })
    finally:
        P.WC2026_HOSTS = orig_hosts

    return results


def print_summary(results: list[dict], label: str):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_actual: dict[str, list[bool]] = {"W1": [], "D": [], "W2": []}
    for r in results:
        by_actual[r["actual"]].append(r["correct"])
    predicted_draws = sum(1 for r in results if r["predicted"] == "D")

    BAR = "─" * 72
    print(f"\n  {label}")
    print(f"  {BAR}")
    print(f"  Overall : {correct}/{total}  ({correct/total*100:.1f}%)"
          f"   Predicted draws: {predicted_draws}/{total}"
          f"  ({predicted_draws/total*100:.1f}%)")
    for lab, outcomes in [
        ("Home wins  (W1)", by_actual["W1"]),
        ("Draws      (D) ", by_actual["D"]),
        ("Away wins  (W2)", by_actual["W2"]),
    ]:
        n, c = len(outcomes), sum(outcomes)
        pct = c / n * 100 if n else 0
        print(f"  {lab}: {c}/{n} correct  ({pct:.0f}%)")
    return correct, by_actual, predicted_draws


def main():
    BAR = "═" * 72
    print(f"\n{BAR}")
    print("  WC 2018 GROUP STAGE BACKTEST — before / after draw classifier")
    print(f"  Cut-off: {CUTOFF.date()}  |  λ={P.DECAY_LAMBDA}  "
          f"half-life≈{round(0.693/P.DECAY_LAMBDA/365.25,1)}y  |  DRAW_BIAS={P.DRAW_BIAS}")
    print(BAR)

    print("\n[1] Loading historical data …")
    data = P.load_data()

    print("\n[2] Training draw classifier (competitive matches before 2018-06-14) …")
    draw_coeffs, draw_threshold = P.fit_draw_classifier(data, today=CUTOFF)
    print(f"  β = [{', '.join(f'{b:.3f}' for b in draw_coeffs)}]"
          f"   threshold = {draw_threshold:.3f}")

    # ── Without classifier ────────────────────────────────────────────────────
    print(f"\n{BAR}")
    print("  RUN 1 — WITHOUT draw classifier  (DRAW_BIAS fallback only)")
    print(BAR)
    print(f"\n[3] Running backtest (MC={P.N_MATCH_SIM:,}/match) …")
    results_base = run_backtest(data, use_stakes=True, draw_coeffs=None)

    # ── With classifier ───────────────────────────────────────────────────────
    print(f"\n{BAR}")
    print("  RUN 2 — WITH draw classifier")
    print(BAR)
    print(f"\n[4] Running backtest (MC={P.N_MATCH_SIM:,}/match) …")
    results_clf = run_backtest(
        data, use_stakes=True,
        draw_coeffs=draw_coeffs, draw_threshold=draw_threshold,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    BAR2 = "═" * 72
    print(f"\n{BAR2}")
    print("  COMPARISON SUMMARY — WC 2018 Group Stage  (stakes ON, prior OFF)")
    print(BAR2)
    r1_cor, r1_by, r1_pd = print_summary(results_base, "Without classifier")
    r2_cor, r2_by, r2_pd = print_summary(results_clf,  "With classifier")

    print(f"\n  {'Metric':<26}  {'Without':>8}  {'With':>8}  {'Δ':>6}")
    print(f"  {'─'*54}")
    total = len(results_base)

    def pct(c, n): return f"{c/n*100:.1f}%" if n else "n/a"

    rows = [
        ("Overall",        r1_cor, r2_cor, total),
        ("Home wins (W1)", sum(r1_by["W1"]), sum(r2_by["W1"]), len(r1_by["W1"])),
        ("Draws (D)",      sum(r1_by["D"]),  sum(r2_by["D"]),  len(r1_by["D"])),
        ("Away wins (W2)", sum(r1_by["W2"]), sum(r2_by["W2"]), len(r1_by["W2"])),
    ]
    for name, c1, c2, n in rows:
        delta = c2 - c1
        sign  = "+" if delta > 0 else ""
        print(f"  {name:<26}  {pct(c1,n):>8}  {pct(c2,n):>8}  {sign}{delta:>+4}")

    print(f"\n  Predicted draws         : {r1_pd:>4} ({pct(r1_pd,total)})  →  "
          f"{r2_pd:>4} ({pct(r2_pd,total)})")
    print(f"{BAR2}\n")


if __name__ == "__main__":
    main()
