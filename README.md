# WC 2026 Match Predictor

Predicts World Cup 2026 group-stage results using historical international
football data and a three-layer statistical model.

## Prerequisites

- **Python 3.10 or later** (uses the `X | Y` union type syntax)
- **No third-party libraries** — standard library only
- **Internet access** — match data is fetched from GitHub on each run

## Usage

Pass the group's teams as positional arguments:

```bash
python3 predictor.py "Brazil" "Argentina" "France" "Germany"
```

Any number of teams ≥ 2 is accepted. All round-robin fixtures are generated
automatically. Data is fetched automatically from GitHub on each run.

### Run all groups at once

```bash
bash run_all_groups.sh
```

Reads every file in `group_stage/`, runs the predictor for each group, and
writes one CSV per group to `results/group_<X>.csv`.

Each CSV has the columns: `Group, Rank, Team, MP, W, D, L, GF, GA, GD, Pts, Adv%`

Pre-computed results for WC 2026 are already committed under [`results/`](results/).

---

## Data source

**[martj42/international_results](https://github.com/martj42/international_results)**
— a community-maintained CSV of every international football match since 1872
(~49 000 matches, updated regularly). Released under CC0 (public domain).

The script uses the last **20 years** of data per team.

---

## Statistical model

### 1 · Exponential time-decay weighting

Every historical match is given a weight that decays exponentially with age:

```
w(t) = exp(−λ · days_ago)     λ = 0.0008
```

At this rate the half-life is ~2.4 years: a match played 2.4 years ago counts
for half as much as one played yesterday.  A match from 10 years ago counts
for ~5 %.

**Why**: national-team rosters and styles change continuously. A friendly
from 2014 tells us little about a team's current form.  Decay weighting lets
the model react to recent performance without hard-cutting old data entirely.

These decayed weights are used to compute two values per team:

| Quantity | Meaning |
|---|---|
| **Attack strength** | weighted avg goals scored / match ÷ global avg |
| **Defense strength** | weighted avg goals conceded / match ÷ global avg |

Values > 1.0 mean above-average; < 1.0 mean below-average.

---

### 2 · Expected goals (Dixon-Coles style)

For each match, expected goals are computed from both teams' attack and
defense strengths:

```
λ_home = attack[home] × defense[away] × global_avg × home_advantage
λ_away = attack[away] × defense[home] × global_avg
```

`global_avg` is the decay-weighted worldwide average goals per team per
match.  `home_advantage = 1.08` reflects the mild positional advantage of
being listed first at a neutral venue.

The most-likely scoreline is found by computing the full 11×11 joint
Poisson probability matrix and finding its peak.

---

### 3 · Dixon-Coles correction (ρ = −0.10)

Pure independent Poisson overestimates the probability of 1-0 and 0-1
results while underestimating 0-0 and 1-1.  Dixon & Coles (1997) introduced
a multiplicative correction τ for the four low-score cells:

```
τ(0,0) = 1 − λ_h · λ_a · ρ      →  boosts 0-0  (ρ < 0)
τ(1,1) = 1 − ρ                   →  boosts 1-1
τ(1,0) = 1 + λ_a · ρ             →  reduces 1-0
τ(0,1) = 1 + λ_h · ρ             →  reduces 0-1
τ(x,y) = 1              for x+y ≥ 3
```

The adjusted joint PMF is then renormalized to sum to 1.

**Why**: this correction is empirically validated on decades of football data.
It produces more realistic scoreline distributions, especially for tight,
low-scoring internationals.

---

### 4 · Bayesian Monte Carlo (uncertainty + upsets)

The attack/defense strengths are *estimated* from data, so they carry
uncertainty.  A team with few recent matches has a wider confidence interval
than one with rich data.

The uncertainty per team is modelled as a log-normal distribution:

```
attack_sample  = attack[team]  × exp(N(0, σ))
defense_sample = defense[team] × exp(N(0, σ))

σ = BASE_SIGMA / √(n_eff / REF_N_EFF)
```

Where `n_eff = Σ decay_weights` (effective number of recent matches) and
`REF_N_EFF = 40` (the "reference" data quantity).

| Effect | Consequence |
|---|---|
| Team with many recent matches | low σ → tight prediction |
| Team with sparse data | high σ → wide prediction → upsets more likely |

**30 000 simulations per match**: in each run, noisy attack/defense values
are sampled, new λ values are derived, and goals are drawn from Poisson.
A partial DC rejection step is applied to low-score cells.  Win / draw / loss
probabilities and the scoreline frequency table come from aggregating these
simulations.

**25 000 full-group simulations**: all six group matches are simulated
end-to-end with independent noisy parameters.  The fraction of runs in which
each team finishes top-2 becomes its *advancement probability* (Adv%).

---

## Output guide

| Field | Source |
|---|---|
| **Exp. goals (base)** | Point-estimate λ from decay-weighted strengths |
| **Most-likely score** | Peak of DC-corrected analytical PMF |
| **Win / Draw / Away %** | Bayesian MC aggregate |
| **Top scorelines** | MC frequency table |
| **Adv%** | Full-group MC (25 000 runs) |
| **σ (noise)** | Per-team strength uncertainty |

---

## Team name spelling

All 48 WC 2026 teams are listed in [`teams.txt`](teams.txt) with their exact
spelling. The script validates names against that file before running and
suggests corrections for typos:

```
Unknown team: 'Argentin'  —  did you mean: Argentina?
```

Copy-paste names from `teams.txt` to avoid issues with spaces and dashes
(e.g. `"South Korea"`, `"Ivory Coast"`, `"United States"`).

## Tests

The test suite covers all pure functions and requires no network access:

```bash
python3 -m unittest test_predictor -v
```

---

## Limitations

- **Friendly matches are weighted equally** to competitive ones.  Adding a
  tournament-type filter (World Cup qualifiers, Nations League) would improve
  quality.
- **No player-level data** (injuries, suspensions, form) — purely team-level
  historical aggregates.
- **Draws in knockout stage** are not modelled; this predictor is for group
  stage only (draw = 1 point each).
- **Correlation between matches** within the group is ignored; each match is
  simulated independently.
