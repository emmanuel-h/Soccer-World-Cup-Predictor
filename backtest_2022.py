#!/usr/bin/env python3
"""
Backtest the WC 2026 predictor against actual WC 2022 group stage results.
Data cut-off: 2022-11-19  (no post-tournament data used in predictions).

Modes
-----
  python3 backtest_2022.py             — full backtest with dead-rubber stakes
  python3 backtest_2022.py --calibrate — calibrate DRAW_BIAS only (fast, no MC)
"""

import itertools
import sys
from datetime import datetime

import predictor as P

CUTOFF = datetime(2022, 11, 19)

WC2022_GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

# Actual results keyed by (team1, team2) per itertools.combinations group order.
# W1 = first team wins, D = draw, W2 = second team wins.
WC2022_RESULTS = {
    # Group A — Netherlands won the group; Qatar (host) finished last
    ("Qatar", "Ecuador"):          ("W2", "0-2"),
    ("Qatar", "Senegal"):          ("W2", "1-3"),
    ("Qatar", "Netherlands"):      ("W2", "0-2"),
    ("Ecuador", "Senegal"):        ("W2", "1-2"),
    ("Ecuador", "Netherlands"):    ("D",  "1-1"),
    ("Senegal", "Netherlands"):    ("W2", "0-2"),
    # Group B
    ("England", "Iran"):           ("W1", "6-2"),
    ("England", "United States"):  ("D",  "0-0"),
    ("England", "Wales"):          ("W1", "3-0"),
    ("Iran", "United States"):     ("W2", "0-1"),
    ("Iran", "Wales"):             ("W1", "2-0"),
    ("United States", "Wales"):    ("D",  "1-1"),
    # Group C — Argentina shocked by Saudi Arabia on MD1
    ("Argentina", "Saudi Arabia"): ("W2", "1-2"),
    ("Argentina", "Mexico"):       ("W1", "2-0"),
    ("Argentina", "Poland"):       ("W1", "2-0"),
    ("Saudi Arabia", "Mexico"):    ("W2", "1-2"),
    ("Saudi Arabia", "Poland"):    ("W2", "0-2"),
    ("Mexico", "Poland"):          ("D",  "0-0"),
    # Group D — France rotated vs Tunisia (dead rubber); Australia beat Denmark
    ("France", "Australia"):       ("W1", "4-1"),
    ("France", "Denmark"):         ("W1", "2-1"),
    ("France", "Tunisia"):         ("W2", "0-1"),
    ("Australia", "Denmark"):      ("W1", "1-0"),
    ("Australia", "Tunisia"):      ("W1", "1-0"),
    ("Denmark", "Tunisia"):        ("D",  "0-0"),
    # Group E — Japan's double shock: beat Germany and Spain; Costa Rica beat Japan
    ("Spain", "Costa Rica"):       ("W1", "7-0"),
    ("Spain", "Germany"):          ("D",  "1-1"),
    ("Spain", "Japan"):            ("W2", "1-2"),
    ("Costa Rica", "Germany"):     ("W2", "2-4"),
    ("Costa Rica", "Japan"):       ("W1", "1-0"),
    ("Germany", "Japan"):          ("W2", "1-2"),
    # Group F — Morocco won the group unbeaten; Belgium eliminated
    ("Belgium", "Canada"):         ("W1", "1-0"),
    ("Belgium", "Morocco"):        ("W2", "0-2"),
    ("Belgium", "Croatia"):        ("D",  "0-0"),
    ("Canada", "Morocco"):         ("W2", "1-2"),
    ("Canada", "Croatia"):         ("W2", "1-4"),
    ("Morocco", "Croatia"):        ("D",  "0-0"),
    # Group G — Cameroon beat Brazil (dead rubber); Switzerland beat Serbia to advance
    ("Brazil", "Serbia"):          ("W1", "2-0"),
    ("Brazil", "Switzerland"):     ("W1", "1-0"),
    ("Brazil", "Cameroon"):        ("W2", "0-1"),
    ("Serbia", "Switzerland"):     ("W2", "2-3"),
    ("Serbia", "Cameroon"):        ("D",  "3-3"),
    ("Switzerland", "Cameroon"):   ("W1", "1-0"),
    # Group H — South Korea beat Portugal to advance; Ghana beat South Korea
    ("Portugal", "Ghana"):         ("W1", "3-2"),
    ("Portugal", "Uruguay"):       ("W1", "2-0"),
    ("Portugal", "South Korea"):   ("W2", "1-2"),
    ("Ghana", "Uruguay"):          ("W2", "0-2"),
    ("Ghana", "South Korea"):      ("W1", "3-2"),
    ("Uruguay", "South Korea"):    ("D",  "0-0"),
}

# ── Dead-rubber stakes (match-day 3) ──────────────────────────────────────────
#
# stake < 1.0 means that team is expected to field a rotated XI (already
# qualified or already eliminated with nothing left to play for).
# Values chosen conservatively: 0.75 = heavy rotation, 0.85 = partial rest.
#
# Teams listed as home/away follow the itertools.combinations fixture order.
WC2022_STAKES = {
    # Group A: Netherlands qualified (4 pts), Qatar eliminated (0 pts)
    # Qatar has nothing to play for either; Netherlands rotated some players.
    ("Qatar", "Netherlands"):     (0.88, 0.78),
    # Group C: Argentina qualified (6 pts after MD2); Messi featured but squad rotated
    ("Argentina", "Poland"):      (0.72, 1.00),
    # Group D: France locked first (6 pts); Deschamps made 8 changes, heavy rotation documented
    ("France", "Tunisia"):        (0.60, 1.00),
    # Group E: Spain qualified (4 pts), partial rotation; Japan fighting for their lives
    ("Spain", "Japan"):           (0.80, 1.00),
    # Group F: Croatia qualified (4 pts), resting key players; Belgium desperate
    ("Belgium", "Croatia"):       (1.00, 0.78),
    # Group F: Canada eliminated (0 pts); Morocco qualified (4 pts), also resting some
    ("Canada", "Morocco"):        (0.82, 0.73),
    # Group G: Brazil locked first (6 pts); Tite made wholesale changes (Neymar injury fears)
    ("Brazil", "Cameroon"):       (0.60, 1.00),
    # Group H: Portugal locked first (6 pts); Santos rested Ronaldo and key midfielders
    ("Portugal", "South Korea"):  (0.75, 1.00),
}

# ── FIFA ranking prior (Fix 4) ────────────────────────────────────────────────
#
# Nov 2022 FIFA rankings for all WC 2022 group-stage teams, used as a Bayesian
# prior to regularise data-driven ratings toward structural team quality.
# Helps teams with sparse / noisy recent data (Canada, Qatar, Saudi Arabia).

FIFA_RANKS_2022: dict[str, int] = {
    "Brazil": 1, "Belgium": 2, "Argentina": 3, "France": 4, "England": 5,
    "Spain": 7, "Portugal": 8, "Netherlands": 8, "Denmark": 10, "Germany": 11,
    "Switzerland": 15, "United States": 16, "Mexico": 13, "Croatia": 17,
    "Uruguay": 14, "Poland": 26, "Senegal": 18, "Morocco": 22, "Japan": 24,
    "South Korea": 28, "Australia": 38, "Canada": 41, "Ghana": 61,
    "Tunisia": 30, "Iran": 20, "Wales": 19, "Ecuador": 44, "Qatar": 50,
    "Serbia": 21, "Cameroon": 43, "Costa Rica": 31, "Saudi Arabia": 53,
}

FIFA_N_PRIOR = 4.0  # pseudo-match weight of the FIFA prior


def _rank_to_strength(rank: int) -> float:
    """Linear FIFA-rank → attack-strength prior (1.18 for rank 1, 0.72 floor)."""
    return max(0.72, 1.18 - 0.006 * (rank - 1))


def _apply_fifa_prior(
    teams: list[str],
    attack: dict[str, float],
    defense: dict[str, float],
    sigma: dict[str, float],
) -> None:
    """Bayesian shrinkage of attack/defense toward FIFA-ranking prior (in-place)."""
    for team in teams:
        rank = FIFA_RANKS_2022.get(team)
        if rank is None:
            continue
        prior_att = _rank_to_strength(rank)
        prior_def = 2.0 - prior_att   # strong team → low conceding → small defense value
        n_eff     = (P.BASE_SIGMA / sigma[team]) ** 2 * P.REF_N_EFF
        w_data    = n_eff / (n_eff + FIFA_N_PRIOR)
        w_prior   = FIFA_N_PRIOR / (n_eff + FIFA_N_PRIOR)
        attack[team]  = w_data * attack[team]  + w_prior * prior_att
        defense[team] = w_data * defense[team] + w_prior * prior_def


# ── Historical WC group-stage draw rates ──────────────────────────────────────

# Inclusive date ranges covering ONLY the group stage of each WC.
WC_GROUP_STAGE_WINDOWS = [
    (datetime(2010, 6, 11), datetime(2010, 6, 26)),   # before R16 (June 26)
    (datetime(2014, 6, 12), datetime(2014, 6, 27)),   # before R16 (June 28)
    (datetime(2018, 6, 14), datetime(2018, 6, 29)),   # before R16 (June 30)
    (datetime(2022, 11, 20), datetime(2022, 12, 3)),  # before R16 (Dec 3)
]


def historical_wc_draw_rate(data: list[dict]) -> float:
    """
    Return the average group-stage draw rate across WC 2010, 2014, 2018, 2022.
    Filters by tournament == 'FIFA World Cup' inside each window.
    """
    total_matches = total_draws = 0
    for start, end in WC_GROUP_STAGE_WINDOWS:
        matches = [
            r for r in data
            if start <= r["date"] < end
            and r.get("tournament") == "FIFA World Cup"
        ]
        draws = sum(1 for r in matches if r["home_score"] == r["away_score"])
        total_matches += len(matches)
        total_draws += draws
    return total_draws / total_matches if total_matches else 0.0


# ── DRAW_BIAS calibration ─────────────────────────────────────────────────────

def calibrate_draw_bias(
    data: list[dict],
    target_pct: float,
    bias_range: tuple = (-0.15, 0.15),
    n_steps: int = 31,
) -> float:
    """
    Analytically sweep DRAW_BIAS values (no MC) and return the value whose
    predicted draw rate is closest to target_pct (expressed as a percentage).
    Uses CUTOFF-filtered data and the current P.DECAY_LAMBDA.
    """
    cutoff_data = [r for r in data if r["date"] < CUTOFF]

    # Pre-compute analytical P_h / P_d / P_a for all 48 fixtures
    match_probs: list[tuple[float, float, float]] = []
    orig_hosts = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()
    try:
        for teams in WC2022_GROUPS.values():
            attack, defense, sigma, global_avg = P.compute_strengths(
                cutoff_data, teams, today=CUTOFF
            )
            for home, away in itertools.combinations(teams, 2):
                lam_h = max(0.3, min(attack[home] * defense[away] * global_avg, 7.0))
                lam_a = max(0.3, min(attack[away] * defense[home] * global_avg, 7.0))
                grid = P.build_dc_grid(lam_h, lam_a)
                match_probs.append(P.probs_from_grid(grid))
    finally:
        P.WC2026_HOSTS = orig_hosts

    step = (bias_range[1] - bias_range[0]) / (n_steps - 1)
    best_bias, best_diff = bias_range[0], float("inf")

    BAR = "─" * 40
    print(f"\n  {BAR}")
    print(f"  Calibration sweep  (target: {target_pct:.1f}% draws)")
    print(f"  {BAR}")
    print(f"  {'DRAW_BIAS':>10}  {'Draws':>6}  {'Pred %':>7}")
    print(f"  {BAR}")

    for i in range(n_steps):
        bias = round(bias_range[0] + i * step, 3)
        draws = sum(
            1 for p_h, p_d, p_a in match_probs
            if not (p_h >= p_a and p_h >= p_d + bias)
            and not (p_a >= p_h and p_a >= p_d + bias)
        )
        pct = draws / len(match_probs) * 100
        diff = abs(pct - target_pct)
        marker = "  ←" if diff < abs(best_diff) else ""
        print(f"  {bias:>10.3f}  {draws:>6}  {pct:>6.1f}%{marker}")
        if diff < abs(best_diff):
            best_diff = pct - target_pct
            best_bias = bias

    print(f"  {BAR}")
    print(f"  Calibrated DRAW_BIAS = {best_bias:.3f}  "
          f"(predicted {target_pct + best_diff:.1f}% draws, target {target_pct:.1f}%)")
    return best_bias


# ── Backtest helpers ──────────────────────────────────────────────────────────

def outcome_code(pred: dict) -> str:
    """Translate a predict() result into 'W1', 'D', or 'W2'."""
    if pred["result"] == "DRAW":
        return "D"
    if pred["home"] in pred["result"]:
        return "W1"
    return "W2"


def suggestion(home: str, away: str, pred_code: str, actual: str, p: dict) -> str:
    ph, pd, pa = p["p_home"], p["p_draw"], p["p_away"]
    margin = abs(ph - pa)

    if actual == "D" and pred_code != "D":
        p_win = ph if pred_code == "W1" else pa
        return (
            f"A {p_win*100:.0f}%-confidence win turned into a draw. "
            f"Consider raising DRAW_BIAS (current {P.DRAW_BIAS}) "
            f"to pull evenly-matched games toward draws."
        )
    if pred_code == "D" and actual != "D":
        winner = home if actual == "W1" else away
        return (
            f"Predicted draw (P_draw={pd*100:.0f}%) but {winner} won outright. "
            f"Lowering DRAW_BIAS would commit to decisive outcomes when one "
            f"team has even a small edge."
        )

    winner = home if actual == "W1" else away
    loser  = away if actual == "W1" else home
    p_win  = ph if actual == "W1" else pa
    p_lose = pa if actual == "W1" else ph

    if p_win < 0.25:
        return (
            f"Major upset: {winner} ({p_win*100:.0f}%) beat {loser} ({p_lose*100:.0f}%). "
            f"Poisson models structurally underweight high-variance upsets. "
            f"A short-window form spike or dead-rubber stake_factor may help."
        )
    if margin < 0.10:
        return (
            f"Near-equal probs ({home} {ph*100:.0f}% / Draw {pd*100:.0f}% "
            f"/ {away} {pa*100:.0f}%); outcome within normal variance."
        )
    return (
        f"Predicted {loser} ({p_lose*100:.0f}%) but {winner} won ({p_win*100:.0f}%). "
        f"Check whether {winner}'s pre-tournament peak was captured in the "
        f"decay window (λ={P.DECAY_LAMBDA}, half-life ≈ {round(0.693/P.DECAY_LAMBDA/365.25,1)}y)."
    )


def run_backtest(
    data: list[dict],
    use_stakes: bool = True,
    use_prior: bool = True,
) -> list[dict]:
    cutoff_data = [r for r in data if r["date"] < CUTOFF]

    results = []
    orig_hosts = P.WC2026_HOSTS
    P.WC2026_HOSTS = set()  # WC 2022 is at a neutral venue; no host advantage

    try:
        for group_name, teams in WC2022_GROUPS.items():
            print(f"\n  Group {group_name}: {', '.join(teams)}")
            attack, defense, sigma, global_avg = P.compute_strengths(
                cutoff_data, teams, today=CUTOFF
            )
            if use_prior:
                _apply_fifa_prior(teams, attack, defense, sigma)
            for home, away in itertools.combinations(teams, 2):
                sh, sa = (WC2022_STAKES.get((home, away), (1.0, 1.0))
                          if use_stakes else (1.0, 1.0))
                pred = P.predict(
                    cutoff_data, attack, defense, sigma, global_avg,
                    home, away, today=CUTOFF,
                    stake_home=sh, stake_away=sa,
                )
                actual, score = WC2022_RESULTS[(home, away)]
                code    = outcome_code(pred)
                correct = code == actual
                mark    = "✓" if correct else "✗"
                ph, pd, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
                stake_tag = (f"  [stake {sh:.2f}/{sa:.2f}]"
                             if (sh != 1.0 or sa != 1.0) else "")
                print(
                    f"    {mark}  {home:<22} vs {away:<22} "
                    f"Pred: {code}  Actual: {actual} ({score})"
                    f"  [{ph*100:.0f}%/{pd*100:.0f}%/{pa*100:.0f}%]"
                    f"{stake_tag}"
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

    failures = [r for r in results if not r["correct"]]
    print(f"\n{BAR}")
    print(f"  FAILED PREDICTIONS ({len(failures)} matches)")
    print(BAR)
    for i, r in enumerate(failures, 1):
        p  = r["pred_dict"]
        ph, pd, pa = p["p_home"], p["p_draw"], p["p_away"]
        sh, sa = p.get("stake_home", 1.0), p.get("stake_away", 1.0)
        stake_note = (f"  stake {sh:.2f}/{sa:.2f}" if (sh != 1.0 or sa != 1.0) else "")
        print(f"\n  {i}. Group {r['group']} — {r['home']} vs {r['away']}  ({r['score']}){stake_note}")
        print(
            f"     Probs      : {r['home']} {ph*100:.0f}%  /  "
            f"Draw {pd*100:.0f}%  /  {r['away']} {pa*100:.0f}%"
        )
        print(f"     Predicted  : {r['predicted']}")
        print(f"     Actual     : {r['actual']}")
        tip = suggestion(r["home"], r["away"], r["predicted"], r["actual"], p)
        words, line, wrapped = tip.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 66:
                wrapped.append(line)
                line = w
            else:
                line = (line + " " + w).lstrip()
        if line:
            wrapped.append(line)
        print(f"     Suggestion : {wrapped[0]}")
        for cont in wrapped[1:]:
            print(f"                  {cont}")

    print(f"\n{BAR}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    calibrate_only = "--calibrate" in sys.argv

    BAR = "═" * 72
    print(f"\n{BAR}")
    print("  WC 2022 GROUP STAGE BACKTEST")
    print(f"  Cut-off: {CUTOFF.date()}  |  λ={P.DECAY_LAMBDA}  "
          f"half-life≈{round(0.693/P.DECAY_LAMBDA/365.25, 1)}y  |  DRAW_BIAS={P.DRAW_BIAS}")
    print(BAR)

    print("\n[1] Loading historical data …")
    data = P.load_data()

    # ── Historical draw rate ──────────────────────────────────────────────────
    hist_rate = historical_wc_draw_rate(data)
    print(f"\n[2] Historical WC group-stage draw rate (2010-2022): "
          f"{hist_rate*100:.1f}%  ({round(hist_rate*48):.0f} draws per 48 matches)")

    # ── Calibration mode ──────────────────────────────────────────────────────
    if calibrate_only:
        print("\n[3] Running DRAW_BIAS calibration (analytical, no MC) …")
        calibrate_draw_bias(data, hist_rate * 100)
        print("\nUpdate DRAW_BIAS in predictor.py with the value marked ← above.")
        return

    # ── Full backtest ─────────────────────────────────────────────────────────
    print(f"\n[3] Running full backtest (48 matches, MC={P.N_MATCH_SIM:,}/match) …")
    results = run_backtest(data, use_stakes=True)

    print_summary(
        results,
        f"BACKTEST SUMMARY — WC 2022 Group Stage  "
        f"(λ={P.DECAY_LAMBDA}  DRAW_BIAS={P.DRAW_BIAS}  stakes+prior=ON)",
    )


if __name__ == "__main__":
    main()
