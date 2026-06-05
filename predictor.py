#!/usr/bin/env python3
"""
World Cup 2026 – Group Predictor  (v2)
Models: exponential time-decay · Dixon-Coles correction · Bayesian MC
Data  : https://github.com/martj42/international_results  (CC0)
"""

import argparse
import csv
import difflib
import io
import itertools
import json
import math
import pathlib
import random
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# Time-decay: half-life ≈ 1.5 years (exp(-0.00127 * 546) ≈ 0.50)
# Calibrated 2025-06: shorter window sharpens recent form, reduces
# strength-estimate compression in lopsided groups.
DECAY_LAMBDA  = 0.00127
HISTORY_YEARS = 20      # how far back to look
H2H_YEARS     = 20      # head-to-head window

# Bayesian uncertainty: BASE_SIGMA at REF_N_EFF effective recent matches
BASE_SIGMA = 0.25
REF_N_EFF  = 40.0

# Dixon-Coles ρ: corrects Poisson over-prediction of 1-0 / 0-1 at the
# expense of 0-0 / 1-1.  Negative ρ boosts low-score draws.
DC_RHO    = -0.25
# Win/draw decision threshold applied to the analytical probabilities.
# Positive: a win is predicted only if P_win > P_draw + DRAW_BIAS (conservative, more draws).
# Negative: a win is predicted even when P_win is up to |DRAW_BIAS| below P_draw (fewer draws).
# Calibrated 2025-06: DRAW_BIAS = -0.010 targets the ~21.9 % draw rate observed in
# WC 2010-2022 group stages.  DC_RHO=-0.25 inflates draw probabilities, so a small
# negative bias is needed to match historical draw frequency.
DRAW_BIAS = -0.010

# Tournament-type multipliers applied on top of time-decay.
# Competitive matches (WC, continental championships) are stronger signals than
# friendlies or Nations League (often rotated squads, low stakes).
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup":               2.0,
    "UEFA Euro":                    1.7,
    "Copa America":                 1.7,
    "Africa Cup of Nations":        1.4,
    "AFC Asian Cup":                1.4,
    "CONCACAF Gold Cup":            1.3,
    "FIFA World Cup qualification": 1.2,
    "UEFA Nations League":          0.8,
    "Friendly":                     0.5,
}

# Draw confidence threshold: when neither team's win probability reaches this,
# predict a draw. Poisson draw probabilities are structurally capped ~32% for
# equal teams, so DRAW_BIAS alone cannot recover the 0% draw accuracy; a
# separate classifier is needed for close-match draws.
DRAW_CONFIDENCE_THRESHOLD = 0.38

HOME_ADVANTAGE = 1.08   # applied only when a host nation plays
WC2026_HOSTS   = {"Mexico", "Canada", "United States"}

N_MATCH_SIM = 30_000  # MC runs per match  (win/draw/loss + scoreline %)
N_GROUP_SIM = 25_000  # MC runs for group advancement probabilities

# ── Poisson primitives ─────────────────────────────────────────────────────────

_FACTORIALS = [math.factorial(k) for k in range(25)]


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k >= 25:
        return 0.0
    return math.exp(-lam) * (lam ** k) / _FACTORIALS[k]


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
    """DC-corrected joint PMF over (home_goals, away_goals), normalised to sum=1."""
    grid, total = {}, 0.0
    exp_h, exp_a = math.exp(-lam_h), math.exp(-lam_a)
    h_pows = [lam_h ** k / _FACTORIALS[k] for k in range(max_g + 1)]
    a_pows = [lam_a ** k / _FACTORIALS[k] for k in range(max_g + 1)]

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
        if   h > a:  p_h += p
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


# ── Tournament weighting ──────────────────────────────────────────────────────

def _tournament_weight(tournament: str) -> float:
    """Match-importance multiplier applied alongside time-decay."""
    t = tournament.lower()
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t:
            return w
    return 1.0


# ── Team strength estimation ───────────────────────────────────────────────────

def _team_weighted_stats(
    recent: list[dict], team: str, today: datetime
) -> tuple[float, float, float]:
    """Decay- and tournament-weighted (scored, conceded, n_eff) for one team."""
    w_scored = w_conceded = n_eff = 0.0
    for r in recent:
        age = (today - r["date"]).days
        tw  = _tournament_weight(r.get("tournament", ""))
        w   = math.exp(-DECAY_LAMBDA * age) * tw
        if r["home_team"] == team:
            w_scored   += w * r["home_score"]
            w_conceded += w * r["away_score"]
            n_eff      += w
        elif r["away_team"] == team:
            w_scored   += w * r["away_score"]
            w_conceded += w * r["home_score"]
            n_eff      += w
    return w_scored, w_conceded, n_eff


def _compute_global_raw_pool(
    recent: list[dict], today: datetime
) -> tuple[dict, dict, float]:
    """
    Single-pass opponent-anchor computation for ALL teams in the dataset.
    Returns (attack_raw, defense_raw, global_avg) with tournament + decay weighting.
    Used as opponent-quality reference in the adjusted strength pass.
    """
    w_scored:   dict[str, float] = defaultdict(float)
    w_conceded: dict[str, float] = defaultdict(float)
    n_eff:      dict[str, float] = defaultdict(float)
    tot_wg = tot_w = 0.0

    for r in recent:
        age = (today - r["date"]).days
        tw  = _tournament_weight(r.get("tournament", ""))
        w   = math.exp(-DECAY_LAMBDA * age) * tw
        tot_wg += w * (r["home_score"] + r["away_score"])
        tot_w  += w * 2
        ht, at = r["home_team"], r["away_team"]
        w_scored[ht]   += w * r["home_score"]
        w_conceded[ht] += w * r["away_score"]
        n_eff[ht]      += w
        w_scored[at]   += w * r["away_score"]
        w_conceded[at] += w * r["home_score"]
        n_eff[at]      += w

    global_avg = tot_wg / tot_w if tot_w else 1.3
    att, dff = {}, {}
    for team, ne in n_eff.items():
        if ne > 0 and global_avg > 0:
            att[team] = (w_scored[team]   / ne) / global_avg
            dff[team] = (w_conceded[team] / ne) / global_avg
        else:
            att[team] = dff[team] = 1.0
    return att, dff, global_avg


def _team_opp_adjusted_stats(
    recent: list[dict], team: str, today: datetime,
    opp_att: dict, opp_def: dict,
) -> tuple[float, float, float]:
    """
    Opponent-quality-adjusted weighted stats.
    Scoring against a strong defense earns more credit; conceding to a weak
    attack is penalised less.  Uses the raw global pool as opponent anchors.
    """
    w_scored = w_conceded = n_eff = 0.0
    for r in recent:
        age = (today - r["date"]).days
        tw  = _tournament_weight(r.get("tournament", ""))
        w   = math.exp(-DECAY_LAMBDA * age) * tw
        if r["home_team"] == team:
            opp   = r["away_team"]
            adj_s = r["home_score"] / max(opp_def.get(opp, 1.0), 0.3)
            adj_c = r["away_score"] / max(opp_att.get(opp, 1.0), 0.3)
        elif r["away_team"] == team:
            opp   = r["home_team"]
            adj_s = r["away_score"] / max(opp_def.get(opp, 1.0), 0.3)
            adj_c = r["home_score"] / max(opp_att.get(opp, 1.0), 0.3)
        else:
            continue
        w_scored   += w * adj_s
        w_conceded += w * adj_c
        n_eff      += w
    return w_scored, w_conceded, n_eff


def compute_strengths(
    data: list[dict], teams: list[str], today: Optional[datetime] = None
) -> tuple[dict, dict, dict, float]:
    """
    Tournament-weighted, opponent-quality-adjusted attack/defense strengths.

    Pass 1 — global raw pool: single scan of all recent matches to build
    opponent-anchor ratings for every team in the dataset.

    Pass 2 — adjusted ratings: for each target team, re-weight each match
    contribution by the opponent's defensive/offensive quality from Pass 1.

    Returns
    -------
    attack, defense : dict[team -> float]  (relative to global average = 1.0)
    sigma           : dict[team -> float]  (log-normal uncertainty per team)
    global_avg      : float               (tournament-weighted avg goals/team/match)
    """
    today  = today or datetime.now()
    cutoff = today - timedelta(days=HISTORY_YEARS * 365.25)
    recent = [r for r in data if r["date"] >= cutoff]

    # Pass 1: global pool for tournament-weighted global_avg (and opponent anchors if needed)
    _, _, global_avg = _compute_global_raw_pool(recent, today)

    attack:  dict[str, float] = {}
    defense: dict[str, float] = {}
    sigma:   dict[str, float] = {}

    # Pass 2: per-team stats using tournament-weighted decay
    # Note: _team_opp_adjusted_stats (opponent-quality adjustment) is available
    # but disabled here — one-pass circular compression degrades same-confederation
    # groups (e.g. CONMEBOL teams all face each other, flattening inter-group spreads).
    # Tournament weighting alone captures the key signal improvements.
    for team in teams:
        w_scored, w_conceded, n_eff = _team_weighted_stats(recent, team, today)
        if n_eff > 0 and global_avg > 0:
            attack[team]  = (w_scored   / n_eff) / global_avg
            defense[team] = (w_conceded / n_eff) / global_avg
        else:
            attack[team] = defense[team] = 1.0
        sigma[team] = max(0.05, BASE_SIGMA / math.sqrt(max(1.0, n_eff / REF_N_EFF)))

    return attack, defense, sigma, global_avg


# ── Head-to-head helper ────────────────────────────────────────────────────────

def h2h_stats(data: list[dict], t1: str, t2: str, today: Optional[datetime] = None) -> tuple | None:
    today   = today or datetime.now()
    cutoff  = today - timedelta(days=H2H_YEARS * 365.25)
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

def _host_multipliers(home: str, away: str) -> tuple[float, float]:
    """Return (ha_h, ha_a): HOME_ADVANTAGE for whichever team is a WC2026 host, else 1.0."""
    return (
        HOME_ADVANTAGE if home in WC2026_HOSTS else 1.0,
        HOME_ADVANTAGE if away in WC2026_HOSTS else 1.0,
    )


def _noisy_lambdas(
    attack: dict, defense: dict, sigma: dict, global_avg: float,
    home: str, away: str,
) -> tuple[float, float]:
    """Sample one set of noisy expected goals (log-normal perturbation)."""
    ha_h, ha_a = _host_multipliers(home, away)
    att_h = attack[home] * math.exp(random.gauss(0, sigma[home]))
    def_a = defense[away] * math.exp(random.gauss(0, sigma[away]))
    att_a = attack[away]  * math.exp(random.gauss(0, sigma[away]))
    def_h = defense[home] * math.exp(random.gauss(0, sigma[home]))
    lh = max(0.20, min(att_h * def_a * global_avg * ha_h, 7.0))
    la = max(0.20, min(att_a * def_h * global_avg * ha_a, 7.0))
    return lh, la


def predict(
    data: list[dict],
    attack: dict, defense: dict, sigma: dict, global_avg: float,
    home: str, away: str,
    today: Optional[datetime] = None,
    stake_home: float = 1.0,
    stake_away: float = 1.0,
) -> dict:
    """
    Two-track: analytical DC grid for outcome/probabilities; MC for scoreline distribution.

    stake_home / stake_away (0 < x ≤ 1.0): match-importance multiplier applied to each
    team's attack strength before computing expected goals.  Use < 1.0 for dead-rubber
    fixtures (squad rotation, low motivation) — e.g. stake_home=0.80 if the home side
    has already qualified and is expected to field a rotated XI.  Both lambdas are scaled
    so the game is modelled as lower-intensity; probabilities flatten toward a draw.
    """
    # Build effective attack dict with stake factors applied
    eff_attack = dict(attack)
    if stake_home != 1.0:
        eff_attack[home] = attack[home] * stake_home
    if stake_away != 1.0:
        eff_attack[away] = attack[away] * stake_away

    ha_h, ha_a = _host_multipliers(home, away)
    lam_h = max(0.3, min(eff_attack[home] * defense[away] * global_avg * ha_h, 7.0))
    lam_a = max(0.3, min(eff_attack[away] * defense[home] * global_avg * ha_a, 7.0))

    grid           = build_dc_grid(lam_h, lam_a)
    p_h, p_d, p_a  = probs_from_grid(grid)
    dc_score       = most_likely_score(grid)

    scorelines: dict[tuple, int] = defaultdict(int)
    for _ in range(N_MATCH_SIM):
        lh, la = _noisy_lambdas(eff_attack, defense, sigma, global_avg, home, away)
        hg     = sample_poisson(lh)
        ag     = sample_poisson(la)
        # Soft DC rejection for low-score cells: resample once when tau < 1
        tau = dc_tau(hg, ag, lh, la, DC_RHO)
        if tau < 1.0 and random.random() > tau:
            hg = sample_poisson(lh)
            ag = sample_poisson(la)
        scorelines[(hg, ag)] += 1

    top5 = sorted(scorelines.items(), key=lambda x: -x[1])[:5]

    # Outcome decision:
    # 1. Draw confidence classifier — when neither team has a clear edge, call draw.
    #    Needed because Poisson P_draw is structurally capped ~32% (even equal teams),
    #    so DRAW_BIAS alone can never make draws the modal prediction.
    # 2. Otherwise fall back to DRAW_BIAS threshold on analytical probabilities.
    if max(p_h, p_a) < DRAW_CONFIDENCE_THRESHOLD:
        outcome, result = "draw", "DRAW"
    elif p_h >= p_a and p_h >= p_d + DRAW_BIAS:
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
        "h2h": h2h_stats(data, home, away, today),
        "sigma_home": sigma[home], "sigma_away": sigma[away],
        "stake_home": stake_home, "stake_away": stake_away,
    }


# ── Full-group Monte Carlo ─────────────────────────────────────────────────────

def group_advancement_mc(
    attack: dict, defense: dict, sigma: dict, global_avg: float,
    fixtures: list[tuple], teams: list[str],
) -> dict[str, float]:
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

            if   hg > ag:  pts[home] += 3
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
        if   hg > ag:  pts[h] += 3; wins[h] += 1; loss[a] += 1
        elif hg == ag: pts[h] += 1; pts[a] += 1; drws[h] += 1; drws[a] += 1
        else:          pts[a] += 3; wins[a] += 1; loss[h] += 1

    ordered = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
    return ordered, pts, wins, drws, loss, gf, ga


# ── Display ────────────────────────────────────────────────────────────────────

BAR = "═" * 68


def _pct(f: float) -> str:
    return f"{f * 100:5.1f}%"


def print_match(p: dict, label: str):
    h, a     = p["home"], p["away"]
    hg, ag   = p["dc_score"]
    dhg, dag = p["definitive_score"]

    print(f"\n  {label}")
    print(f"  {'─'*64}")
    print(f"  {h:<24} vs  {a}")
    stakes = ""
    if p.get("stake_home", 1.0) != 1.0 or p.get("stake_away", 1.0) != 1.0:
        stakes = (f"   stake = {p['stake_home']:.2f} / {p['stake_away']:.2f}"
                  f"  ⚠ dead-rubber adj.")
    print(f"  Exp. goals (base)  : {p['lam_h']:.2f} – {p['lam_a']:.2f}"
          f"   σ = {p['sigma_home']:.2f} / {p['sigma_away']:.2f}{stakes}")
    print(f"  Most-likely score  : {hg} – {ag}  (DC analytical, unconstrained)")
    print(f"  DC probability     : {h} {_pct(p['p_home'])}  │"
          f"  Draw {_pct(p['p_draw'])}  │  {a} {_pct(p['p_away'])}")
    print(f"  ► Prediction       : {p['result']}  →  {dhg} – {dag}")

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


# ── Output helpers ─────────────────────────────────────────────────────────────

def _read_json_list(path: pathlib.Path) -> list:
    """Read an existing JSON array from disk, or return an empty list."""
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists() and path.stat().st_size
        else []
    )


def write_matches_csv(
    path: pathlib.Path | str,
    group: Optional[str],
    predictions: list[dict],
):
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


def _predictions_to_records(group: Optional[str], predictions: list[dict]) -> list[dict]:
    records = []
    for i, p in enumerate(predictions, 1):
        hg, ag = p["definitive_score"]
        records.append({
            "group":        group or "",
            "match":        i,
            "homeTeam":     p["home"],
            "awayTeam":     p["away"],
            "expGoalsHome": round(p["lam_h"], 2),
            "expGoalsAway": round(p["lam_a"], 2),
            "homeScore":    hg,
            "awayScore":    ag,
            "pHome":        round(p["p_home"] * 100, 1),
            "pDraw":        round(p["p_draw"] * 100, 1),
            "pAway":        round(p["p_away"] * 100, 1),
            "prediction":   p["result"],
        })
    return records


def write_matches_json(
    path: pathlib.Path | str,
    group: Optional[str],
    predictions: list[dict],
):
    """Merge match prediction records into a JSON file (array, append-safe)."""
    file = pathlib.Path(path)
    existing = _read_json_list(file)
    existing.extend(_predictions_to_records(group, predictions))
    file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def write_matches_mpp(
    path: pathlib.Path | str,
    group: Optional[str],
    predictions: list[dict],
):
    """
    Export predictions in MPP push format: array of objects ready for
    PATCH /user-match-forecasts/entity/{scope}/match/{matchId}.
    Each entry contains the metadata needed to match against the MPP calendar.
    """
    file = pathlib.Path(path)
    existing = _read_json_list(file)
    for p in predictions:
        hg, ag = p["definitive_score"]
        existing.append({
            "group":      group or "",
            "homeTeam":   p["home"],
            "awayTeam":   p["away"],
            "homeScore":  hg,
            "awayScore":  ag,
            "originPage": "home",
        })
    file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Argument parsing and team validation ───────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Group Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predictor.py Mexico \"South Africa\" \"South Korea\" \"Czech Republic\"\n"
            "  python predictor.py --output-format csv,json --output-dir results/ "
            "--group \"Group A\" Mexico \"South Africa\" \"South Korea\" \"Czech Republic\"\n"
            "  python predictor.py --output-format mpp --output-dir results/ "
            "--group \"Group A\" Mexico \"South Africa\" \"South Korea\" \"Czech Republic\""
        ),
    )
    parser.add_argument("teams", nargs="+", metavar="TEAM",
                        help="Teams in the group (at least 2)")

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output-format", metavar="FORMAT",
        default="csv",
        help=(
            "Comma-separated list of output formats: csv, json, mpp (default: csv). "
            "'mpp' produces a JSON file ready for mpp_push.py."
        ),
    )
    out_group.add_argument(
        "--output-dir", metavar="DIR",
        help="Directory where output files are written (uses group name for filename).",
    )
    out_group.add_argument(
        "--output-stem", metavar="STEM",
        help="Base filename (without extension) for output files. "
             "Overrides the auto-name derived from --group.",
    )
    out_group.add_argument("--matches-csv", metavar="FILE", help=argparse.SUPPRESS)
    out_group.add_argument("--group", metavar="NAME",
                           help="Group label written into output files (e.g. 'Group A')")

    args = parser.parse_args()
    if len(args.teams) < 2:
        parser.error("Provide at least 2 teams.")

    args.output_formats = {f.strip().lower() for f in args.output_format.split(",")}
    unknown = args.output_formats - {"csv", "json", "mpp"}
    if unknown:
        parser.error(f"Unknown output format(s): {', '.join(sorted(unknown))}. "
                     "Choose from: csv, json, mpp")

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


def _resolve_output_paths(args) -> dict[str, pathlib.Path]:
    """
    Return a mapping of format -> Path for each requested output format.
    Priority: --output-stem > --output-dir+group > --matches-csv (legacy).
    """
    paths: dict[str, pathlib.Path] = {}
    ext_map = {"csv": ".csv", "json": ".json", "mpp": "_mpp.json"}

    if args.output_stem:
        stem = pathlib.Path(args.output_stem)
        for fmt in args.output_formats:
            paths[fmt] = stem.with_suffix("").parent / (stem.stem + ext_map[fmt])
    elif args.output_dir:
        out_dir = pathlib.Path(args.output_dir)
        slug = (args.group or "predictions").lower().replace(" ", "_")
        for fmt in args.output_formats:
            paths[fmt] = out_dir / (slug + ext_map[fmt])
    elif args.matches_csv and "csv" in args.output_formats:
        paths["csv"] = pathlib.Path(args.matches_csv)

    return paths


# ── main() helpers ─────────────────────────────────────────────────────────────

def _display_strengths(teams: list[str], attack: dict, defense: dict, sigma: dict):
    print(f"\n  {'Team':<24} {'Attack':>8} {'Defense':>9} {'σ (noise)':>10}")
    print(f"  {'─'*54}")
    for t in teams:
        n_eff = REF_N_EFF * (BASE_SIGMA / sigma[t]) ** 2
        print(f"  {t:<24} {attack[t]:>8.3f} {defense[t]:>9.3f} {sigma[t]:>10.3f}"
              f"  (n_eff ≈ {n_eff:.0f})")


def _run_match_predictions(
    data: list[dict],
    attack: dict, defense: dict, sigma: dict, global_avg: float,
    fixtures: list[tuple],
) -> list[dict]:
    predictions = []
    for home, away, label in fixtures:
        p = predict(data, attack, defense, sigma, global_avg, home, away)
        predictions.append(p)
        print_match(p, label)
    return predictions


def _write_prediction_outputs(
    output_paths: dict[str, pathlib.Path],
    group_label: Optional[str],
    predictions: list[dict],
):
    writers = {
        "csv":  lambda path: write_matches_csv(path, group_label, predictions),
        "json": lambda path: write_matches_json(path, group_label, predictions),
        "mpp":  lambda path: write_matches_mpp(path, group_label, predictions),
    }
    for fmt, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[fmt](path)
        print(f"\n  [{fmt.upper()}] written → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args         = parse_args()
    teams        = args.teams
    group_label  = args.group
    output_paths = _resolve_output_paths(args)
    fixtures     = generate_fixtures(teams)

    check_teams(teams, load_known_teams())

    print(f"\n{BAR}")
    print("  WORLD CUP 2026 — GROUP PREDICTOR  (v2)")
    print(f"  {', '.join(teams)}")
    print(BAR)

    print("\n[1/4] Loading historical data …")
    data = load_data()

    print("\n[2/4] Computing decay-weighted team strengths …")
    attack, defense, sigma, global_avg = compute_strengths(data, teams)
    print(f"  Global avg goals / team / match : {global_avg:.3f}")
    _display_strengths(teams, attack, defense, sigma)

    print(f"\n[3/4] Predicting matches  (Bayesian MC, {N_MATCH_SIM:,} runs / match) …")
    print(f"\n{BAR}")
    print("  MATCH PREDICTIONS")
    print(BAR)
    predictions = _run_match_predictions(data, attack, defense, sigma, global_avg, fixtures)
    _write_prediction_outputs(output_paths, group_label, predictions)

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
