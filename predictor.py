#!/usr/bin/env python3
"""
World Cup 2026 – Group Predictor  (v2)
Models: exponential time-decay · Dixon-Coles correction · Bayesian MC
Data  : https://github.com/martj42/international_results  (CC0)
"""

import argparse, csv, difflib, io, itertools, math, pathlib, random, sys, urllib.request
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# Time-decay: half-life ≈ 2.4 years (exp(-0.0008 * 875) = 0.50)
DECAY_LAMBDA   = 0.0008
HISTORY_YEARS  = 20      # how far back to look
H2H_YEARS      = 20      # head-to-head window

# Bayesian uncertainty: BASE_SIGMA at REF_N_EFF effective recent matches
BASE_SIGMA     = 0.25
REF_N_EFF      = 40.0

# Dixon-Coles ρ: corrects Poisson over-prediction of 1-0 / 0-1 at the
# expense of 0-0 / 1-1.  Negative ρ boosts low-score draws.
DC_RHO         = -0.25
# Win must exceed P_draw by this margin; prevents 0.1% edges from always overriding a draw.
DRAW_BIAS      = 0.04

HOME_ADVANTAGE = 1.08    # neutral WC venue → modest edge for "home" ordering

N_MATCH_SIM    = 30_000  # MC runs per match  (win/draw/loss + scoreline %)
N_GROUP_SIM    = 25_000  # MC runs for group advancement probabilities

# ── Poisson primitives ─────────────────────────────────────────────────────────

_FACT = [math.factorial(k) for k in range(25)]


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k >= 25:
        return 0.0
    return math.exp(-lam) * (lam ** k) / _FACT[k]


def sample_poisson(lam: float) -> int:
    """Knuth algorithm – exact Poisson sampler."""
    threshold = math.exp(-min(lam, 30.0))
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= threshold:
            return k - 1


# ── Dixon-Coles correction ─────────────────────────────────────────────────────

def dc_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Multiplicative correction for the four low-score cells.
    Keeps the joint PMF valid (tau >= 0) for |rho| < 1 / max(lam).
    """
    if   h == 0 and a == 0: return max(0.0, 1.0 - lam_h * lam_a * rho)
    elif h == 1 and a == 0: return max(0.0, 1.0 + lam_a * rho)
    elif h == 0 and a == 1: return max(0.0, 1.0 + lam_h * rho)
    elif h == 1 and a == 1: return max(0.0, 1.0 - rho)
    return 1.0


def build_dc_grid(lam_h: float, lam_a: float, max_g: int = 10) -> dict:
    """
    DC-corrected joint PMF over (home_goals, away_goals), normalised to sum=1.
    """
    grid, total = {}, 0.0
    exp_h, exp_a = math.exp(-lam_h), math.exp(-lam_a)
    h_pows = [lam_h ** k / _FACT[k] for k in range(max_g + 1)]
    a_pows = [lam_a ** k / _FACT[k] for k in range(max_g + 1)]

    for h in range(max_g + 1):
        ph = exp_h * h_pows[h]
        for a in range(max_g + 1):
            p = max(0.0, ph * exp_a * a_pows[a] * dc_tau(h, a, lam_h, lam_a, DC_RHO))
            grid[(h, a)] = p
            total += p

    if total > 0:
        for k in grid:
            grid[k] /= total
    return grid


def probs_from_grid(grid: dict) -> tuple[float, float, float]:
    """Marginalize the joint PMF into (P_home_win, P_draw, P_away_win)."""
    p_h = p_d = p_a = 0.0
    for (h, a), p in grid.items():
        if   h > a: p_h += p
        elif h == a: p_d += p
        else:        p_a += p
    return p_h, p_d, p_a


def most_likely_score(grid: dict) -> tuple[int, int]:
    return max(grid, key=grid.__getitem__)


def most_likely_score_for_outcome(grid: dict, outcome: str) -> tuple[int, int]:
    """Most probable scoreline consistent with the predicted outcome (home/draw/away)."""
    if outcome == "home":
        filtered = {k: v for k, v in grid.items() if k[0] > k[1]}
    elif outcome == "draw":
        filtered = {k: v for k, v in grid.items() if k[0] == k[1]}
    else:
        filtered = {k: v for k, v in grid.items() if k[0] < k[1]}
    return max(filtered, key=filtered.__getitem__) if filtered else most_likely_score(grid)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> list[dict]:
    print("  Fetching match data from GitHub …")
    with urllib.request.urlopen(DATA_URL, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(content)):
        try:
            row["date"]       = datetime.strptime(row["date"], "%Y-%m-%d")
            row["home_score"] = int(row["home_score"])
            row["away_score"] = int(row["away_score"])
            rows.append(row)
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda r: r["date"])
    print(f"  Loaded {len(rows):,} matches  "
          f"({rows[0]['date'].year}–{rows[-1]['date'].year})")
    return rows


# ── Team strength estimation ───────────────────────────────────────────────────

def compute_strengths(data: list[dict], teams: list[str]):
    """
    Exponential-decay weighted attack / defense strengths.

    Returns
    -------
    attack, defense : dict[team -> float]  (relative to global average = 1.0)
    sigma           : dict[team -> float]  (log-normal uncertainty per team)
    global_avg      : float               (weighted avg goals per team per match)
    """
    today   = datetime.now()
    cutoff  = today - timedelta(days=HISTORY_YEARS * 365.25)
    recent  = [r for r in data if r["date"] >= cutoff]

    # Decay-weighted global average
    tot_wg = tot_w = 0.0
    for r in recent:
        w      = math.exp(-DECAY_LAMBDA * (today - r["date"]).days)
        tot_wg += w * (r["home_score"] + r["away_score"])
        tot_w  += w * 2          # two team slots per match
    global_avg = tot_wg / tot_w if tot_w else 1.3

    attack, defense, sigma = {}, {}, {}

    for team in teams:
        w_scored = w_conceded = n_eff = 0.0

        for r in recent:
            age = (today - r["date"]).days
            w   = math.exp(-DECAY_LAMBDA * age)

            if r["home_team"] == team:
                w_scored   += w * r["home_score"]
                w_conceded += w * r["away_score"]
                n_eff      += w
            elif r["away_team"] == team:
                w_scored   += w * r["away_score"]
                w_conceded += w * r["home_score"]
                n_eff      += w

        if n_eff > 0 and global_avg > 0:
            attack[team]  = (w_scored   / n_eff) / global_avg
            defense[team] = (w_conceded / n_eff) / global_avg
        else:
            attack[team] = defense[team] = 1.0

        # Less data (lower n_eff) → wider uncertainty → more upset potential
        sigma[team] = max(0.05, BASE_SIGMA / math.sqrt(max(1.0, n_eff / REF_N_EFF)))

    return attack, defense, sigma, global_avg


# ── Head-to-head helper ────────────────────────────────────────────────────────

def h2h_stats(data: list[dict], t1: str, t2: str) -> tuple | None:
    cutoff  = datetime.now() - timedelta(days=H2H_YEARS * 365.25)
    matches = [r for r in data
               if r["date"] >= cutoff
               and {r["home_team"], r["away_team"]} == {t1, t2}]
    if not matches:
        return None
    w1 = sum(1 for r in matches
             if (r["home_team"] == t1 and r["home_score"] > r["away_score"]) or
                (r["away_team"] == t1 and r["away_score"] > r["home_score"]))
    w2 = sum(1 for r in matches
             if (r["home_team"] == t2 and r["home_score"] > r["away_score"]) or
                (r["away_team"] == t2 and r["away_score"] > r["home_score"]))
    return len(matches), w1, len(matches) - w1 - w2, w2


# ── Per-match prediction ───────────────────────────────────────────────────────

def _noisy_lambdas(attack, defense, sigma, global_avg, home, away):
    """Sample one set of noisy expected goals (log-normal perturbation)."""
    att_h = attack[home] * math.exp(random.gauss(0, sigma[home]))
    def_a = defense[away] * math.exp(random.gauss(0, sigma[away]))
    att_a = attack[away]  * math.exp(random.gauss(0, sigma[away]))
    def_h = defense[home] * math.exp(random.gauss(0, sigma[home]))
    lh = max(0.20, min(att_h * def_a * global_avg * HOME_ADVANTAGE, 7.0))
    la = max(0.20, min(att_a * def_h * global_avg,                   7.0))
    return lh, la


def predict(data, attack, defense, sigma, global_avg, home, away) -> dict:
    """Two-track: analytical DC grid for outcome/probabilities; MC for scoreline distribution."""
    # Point-estimate lambdas (for analytical DC scoreline)
    lam_h = max(0.3, min(attack[home] * defense[away] * global_avg * HOME_ADVANTAGE, 7.0))
    lam_a = max(0.3, min(attack[away] * defense[home] * global_avg,                   7.0))

    # Analytical DC-corrected probabilities and most-likely score
    grid          = build_dc_grid(lam_h, lam_a)
    p_h, p_d, p_a = probs_from_grid(grid)
    dc_score      = most_likely_score(grid)

    # Bayesian MC: sample noisy strengths → Poisson goals → scoreline distribution
    scorelines: dict[tuple, int] = defaultdict(int)

    for _ in range(N_MATCH_SIM):
        lh, la = _noisy_lambdas(attack, defense, sigma, global_avg, home, away)
        hg     = sample_poisson(lh)
        ag     = sample_poisson(la)

        # Soft DC rejection for low-score cells: resample once when tau < 1
        tau = dc_tau(hg, ag, lh, la, DC_RHO)
        if tau < 1.0 and random.random() > tau:
            hg = sample_poisson(lh)
            ag = sample_poisson(la)

        scorelines[(hg, ag)] += 1

    top5 = sorted(scorelines.items(), key=lambda x: -x[1])[:5]

    # Analytical probs drive the outcome: MC soft-rejection never boosts tau>1 draws.
    if p_h >= p_a and p_h >= p_d + DRAW_BIAS:
        outcome, result = "home", f"{home} WIN"
    elif p_a >= p_h and p_a >= p_d + DRAW_BIAS:
        outcome, result = "away", f"{away} WIN"
    else:
        outcome, result = "draw", "DRAW"

    definitive_score = most_likely_score_for_outcome(grid, outcome)

    return {
        "home": home, "away": away,
        "lam_h": lam_h, "lam_a": lam_a,
        "dc_score": dc_score,
        "definitive_score": definitive_score,
        "p_home": p_h, "p_draw": p_d, "p_away": p_a,
        "result": result,
        "top5": top5,
        "h2h": h2h_stats(data, home, away),
        "sigma_home": sigma[home], "sigma_away": sigma[away],
    }


# ── Full-group Monte Carlo ─────────────────────────────────────────────────────

def group_advancement_mc(attack, defense, sigma, global_avg, fixtures, teams) -> dict:
    """
    Simulate the entire group N_GROUP_SIM times.
    Returns probability each team finishes top-2 (advances to round of 32).
    """
    advances: dict[str, int] = defaultdict(int)

    for _ in range(N_GROUP_SIM):
        pts: dict[str, int] = defaultdict(int)
        gf:  dict[str, int] = defaultdict(int)
        ga:  dict[str, int] = defaultdict(int)

        for home, away, _ in fixtures:
            lh, la = _noisy_lambdas(attack, defense, sigma, global_avg, home, away)
            hg = sample_poisson(lh)
            ag = sample_poisson(la)

            gf[home] += hg; ga[home] += ag
            gf[away] += ag; ga[away] += hg

            if   hg > ag: pts[home] += 3
            elif hg == ag: pts[home] += 1; pts[away] += 1
            else:          pts[away] += 3

        ranked = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
        advances[ranked[0]] += 1
        advances[ranked[1]] += 1

    return {t: advances[t] / N_GROUP_SIM for t in teams}


# ── Deterministic standings helper ────────────────────────────────────────────

def deterministic_standings(predictions: list[dict], teams: list[str]):
    pts  = defaultdict(int)
    gf   = defaultdict(int)
    ga   = defaultdict(int)
    wins = defaultdict(int)
    drws = defaultdict(int)
    loss = defaultdict(int)

    for p in predictions:
        h, a   = p["home"], p["away"]
        hg, ag = p["definitive_score"]
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        if   hg > ag: pts[h] += 3; wins[h] += 1; loss[a] += 1
        elif hg == ag: pts[h] += 1; pts[a] += 1; drws[h] += 1; drws[a] += 1
        else:          pts[a] += 3; wins[a] += 1; loss[h] += 1

    ordered = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
    return ordered, pts, wins, drws, loss, gf, ga


# ── Display ────────────────────────────────────────────────────────────────────

BAR = "═" * 68


def _pct(f: float) -> str:
    return f"{f * 100:5.1f}%"


def print_match(p: dict, label: str):
    h, a   = p["home"], p["away"]
    hg, ag = p["dc_score"]

    dhg, dag = p["definitive_score"]

    print(f"\n  {label}")
    print(f"  {'─'*64}")
    print(f"  {h:<24} vs  {a}")
    print(f"  Exp. goals (base)  : {p['lam_h']:.2f} – {p['lam_a']:.2f}"
          f"   σ = {p['sigma_home']:.2f} / {p['sigma_away']:.2f}")
    print(f"  Most-likely score  : {hg} – {ag}  (DC analytical, unconstrained)")
    print(f"  DC probability     : {h} {_pct(p['p_home'])}  │"
          f"  Draw {_pct(p['p_draw'])}  │  {a} {_pct(p['p_away'])}")
    print(f"  ► Prediction       : {p['result']}  →  {dhg} – {dag}")

    # top-5 scorelines from MC
    print(f"  Top scorelines     : ", end="")
    parts = [f"{sc[0]}-{sc[1]} ({cnt/N_MATCH_SIM*100:.1f}%)" for sc, cnt in p["top5"]]
    print("  ".join(parts))

    if p["h2h"]:
        n, w1, d, w2 = p["h2h"]
        print(f"  H2H (last {H2H_YEARS}y)    : {n} matches — "
              f"{h} {w1}W / {d}D / {w2}W {a}")


def print_standings(ordered, pts, wins, drws, loss, gf, ga, adv_pct):
    print(f"\n  {'Team':<24} {'MP':>3} {'W':>3} {'D':>3} {'L':>3}"
          f" {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}  {'Adv%':>6}")
    print(f"  {'─'*64}")
    for i, t in enumerate(ordered):
        mp   = wins[t] + drws[t] + loss[t]
        gd   = gf[t] - ga[t]
        mark = "  ✓" if i < 2 else ""
        print(f"  {t:<24} {mp:>3} {wins[t]:>3} {drws[t]:>3} {loss[t]:>3}"
              f" {gf[t]:>3} {ga[t]:>3} {gd:>+4} {pts[t]:>4}"
              f"  {adv_pct[t]*100:5.1f}%{mark}")


# ── Entry point ────────────────────────────────────────────────────────────────

def write_matches_csv(path: str, group: Optional[str], predictions: list[dict]):
    """Append match prediction rows to a CSV file."""
    file = pathlib.Path(path)
    write_header = not file.exists() or file.stat().st_size == 0
    with file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Group", "Match", "HomeTeam", "AwayTeam",
                "ExpGoalsHome", "ExpGoalsAway", "DefinitiveScore",
                "P_Home", "P_Draw", "P_Away", "Prediction",
            ])
        for i, p in enumerate(predictions, 1):
            hg, ag = p["definitive_score"]
            writer.writerow([
                group or "",
                i,
                p["home"],
                p["away"],
                f"{p['lam_h']:.2f}",
                f"{p['lam_a']:.2f}",
                f"{hg}-{ag}",
                f"{p['p_home']*100:.1f}",
                f"{p['p_draw']*100:.1f}",
                f"{p['p_away']*100:.1f}",
                p["result"],
            ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Group Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predictor.py Mexico \"South Africa\" \"South Korea\" \"Czech Republic\"",
    )
    parser.add_argument("teams", nargs="+", metavar="TEAM",
                        help="Teams in the group (at least 2)")
    parser.add_argument("--matches-csv", metavar="FILE",
                        help="Append match predictions as CSV rows to FILE")
    parser.add_argument("--group", metavar="NAME",
                        help="Group label written into the CSV (e.g. 'Group A')")
    args = parser.parse_args()
    if len(args.teams) < 2:
        parser.error("Provide at least 2 teams.")
    return args


def load_known_teams() -> list[str]:
    path = pathlib.Path(__file__).parent / "teams.txt"
    teams = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            teams.append(line)
    return teams


def check_teams(teams: list[str], known: list[str]):
    known_set = set(known)
    unknown = [t for t in teams if t not in known_set]
    if not unknown:
        return
    for t in unknown:
        suggestions = difflib.get_close_matches(t, known, n=3, cutoff=0.6)
        msg = f"Unknown team: '{t}'"
        if suggestions:
            msg += f"  —  did you mean: {', '.join(suggestions)}?"
        print(msg, file=sys.stderr)
    print(f"\nSee teams.txt for the full list of valid team names.", file=sys.stderr)
    sys.exit(1)


def generate_fixtures(teams: list[str]) -> list[tuple]:
    return [(t1, t2, f"Match {i}") for i, (t1, t2) in
            enumerate(itertools.combinations(teams, 2), 1)]


def main():
    args  = parse_args()
    teams = args.teams
    check_teams(teams, load_known_teams())
    matches_csv = args.matches_csv
    group_label = args.group
    fixtures = generate_fixtures(teams)

    print(f"\n{BAR}")
    print("  WORLD CUP 2026 — GROUP PREDICTOR  (v2)")
    print(f"  {', '.join(teams)}")
    print(BAR)

    # 1 ── load
    print("\n[1/4] Loading historical data …")
    data = load_data()

    # 2 ── strengths
    print("\n[2/4] Computing decay-weighted team strengths …")
    attack, defense, sigma, global_avg = compute_strengths(data, teams)
    print(f"  Global avg goals / team / match : {global_avg:.3f}")
    print(f"\n  {'Team':<24} {'Attack':>8} {'Defense':>9} {'σ (noise)':>10}")
    print(f"  {'─'*54}")
    for t in teams:
        n_eff = REF_N_EFF * (BASE_SIGMA / sigma[t]) ** 2
        print(f"  {t:<24} {attack[t]:>8.3f} {defense[t]:>9.3f} {sigma[t]:>10.3f}"
              f"  (n_eff ≈ {n_eff:.0f})")

    # 3 ── match predictions
    print(f"\n[3/4] Predicting matches  (Bayesian MC, {N_MATCH_SIM:,} runs / match) …")
    print(f"\n{BAR}")
    print("  MATCH PREDICTIONS")
    print(BAR)

    predictions = []
    for home, away, label in fixtures:
        p = predict(data, attack, defense, sigma, global_avg, home, away)
        predictions.append(p)
        print_match(p, label)

    if matches_csv:
        write_matches_csv(matches_csv, group_label, predictions)
        print(f"\n  Match data written to {matches_csv}")

    # 4 ── group advancement
    print(f"\n[4/4] Simulating group advancement  ({N_GROUP_SIM:,} full-group runs) …")
    adv_pct = group_advancement_mc(attack, defense, sigma, global_avg, fixtures, teams)

    ordered, pts, wins, drws, loss, gf, ga = deterministic_standings(predictions, teams)

    print(f"\n{BAR}")
    print("  PREDICTED STANDINGS")
    print("  (Adv% = probability of finishing top-2 across all MC simulations)")
    print(BAR)
    print_standings(ordered, pts, wins, drws, loss, gf, ga, adv_pct)

    print(f"\n{BAR}")
    print(f"  Decay λ={DECAY_LAMBDA}  half-life≈{round(math.log(2)/DECAY_LAMBDA/365.25,1)}y  │  "
          f"DC ρ={DC_RHO}  │  σ_base={BASE_SIGMA}")
    print(f"  {N_MATCH_SIM:,} match sims · {N_GROUP_SIM:,} group sims")
    print("  Data: martj42/international_results (CC0)")
    print(f"{BAR}\n")


if __name__ == "__main__":
    main()
