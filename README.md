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

### Output formats

Use `--output-format` (comma-separated) and `--output-dir` to control what
gets written:

```bash
# CSV only (default)
python3 predictor.py --output-format csv --output-dir results --group "Group A" \
    Mexico "South Africa" "South Korea" "Czech Republic"

# All three formats at once
python3 predictor.py --output-format csv,json,mpp --output-dir results --group "Group A" \
    Mexico "South Africa" "South Korea" "Czech Republic"
```

| Format | File | Contents |
|---|---|---|
| `csv` | `group_a.csv` | Match predictions, expected goals, probabilities |
| `json` | `group_a.json` | Same as CSV but as a JSON array |
| `mpp` | `group_a_mpp.json` | Minimal payload ready for `mpp_push.py` |

### Run all groups at once

```bash
bash run_all_groups.sh
```

Reads every file in `group_stage/`, runs the predictor for each group, and
writes outputs to `results/`. The default formats are `csv,json,mpp`; override
with the `OUTPUT_FORMAT` environment variable:

```bash
OUTPUT_FORMAT=csv bash run_all_groups.sh
```

Results are gitignored and must be regenerated locally (see workflow above).

---

## Pushing predictions to Mon Petit Prono (MPP)

`mpp_push.py` reads the MPP-format JSON files produced above and submits them
to your [mpp.football](https://mpp.football) account via the private API.

### First-time setup

1. Log in at [mpp.football](https://mpp.football) in your browser.
2. Open DevTools → Application → Local Storage → `https://mpp.football`.
3. Find the Auth0 key containing `refresh_token` and copy its value.
4. Run the push script — it will prompt for the token and cache it in
   `.mpp_tokens.json` (gitignored):

```bash
python3 mpp_push.py results/all_groups_mpp.json --championship-id 8
```

The token file is updated automatically on every run (Auth0 rotates tokens).

### Typical workflow

```bash
# 1. Generate all group predictions (CSV + JSON + MPP) and combine them
bash run_all_groups.sh
# → writes results/group_*.csv, results/group_*.json, results/group_*_mpp.json
# → combines into results/all_groups.json and results/all_groups_mpp.json

# 2. Find your championship ID
python3 mpp_push.py --list-championships

# 3. Dry run to preview the match mapping
python3 mpp_push.py results/all_groups_mpp.json --championship-id 8 --dry-run

# 4. Submit
python3 mpp_push.py results/all_groups_mpp.json --championship-id 8
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--championship-id ID` | *(required)* | Use `--list-championships` to find it |
| `--scope SCOPE` | `general` | `general` for global leaderboard; or a contest ID |
| `--list-championships` | — | Print your active contests and exit |
| `--dry-run` | — | Preview without submitting |
| `--yes` / `-y` | — | Skip confirmation prompt |
| `--token-file FILE` | `.mpp_tokens.json` | Path to the token cache file |

---

## Data source

**[martj42/international_results](https://github.com/martj42/international_results)**
— a community-maintained CSV of every international football match since 1872
(~49 300 matches, updated regularly). Released under CC0 (public domain).

The script uses the last **20 years** of data per team (though exponential decay
makes data older than 8–10 years effectively negligible).

---

## Statistical model

### 1 · Exponential time-decay weighting

Every historical match is given a weight that decays exponentially with age:

```
w(t) = exp(−λ · days_ago)     λ = 0.00127   (half-life ≈ 1.5 years)
```

A match played 1.5 years ago counts for half as much as one played yesterday.
A match from 5 years ago counts for ~10 %.

**Why λ = 0.00127 (previously 0.0008 / 2.4 y)**: backtesting against WC 2022
showed that a 2.4-year window compressed team strength estimates — top sides
like France and Brazil showed near-identical profiles to mid-table teams in
evenly-spread groups.  Shortening to 1.5 years sharpens recent form and
produces more differentiated λ values, improving away-win prediction accuracy
from 25 % to 40 % on the WC 2022 group stage.

These weights also serve as observation weights in the MLE fit described next.

---

### 2 · Opponent-quality-adjusted strengths (Dixon-Coles MLE)

Raw goal averages ignore opponent quality — a 3-0 win over San Marino inflates
attack ratings identically to a 3-0 win over Germany.  Instead, attack and
defense multipliers for *every team in the dataset* are estimated jointly by
maximising the weighted Poisson log-likelihood:

```
log P(h, a | λ_h, λ_a)   where   λ_h = att[home] × dff[away] × μ
                                   λ_a = att[away] × dff[home] × μ
```

This is solved via 100 iterations of multiplicative EM (a closed-form Poisson
GLM update): at each step, each team's attack/defense parameter is scaled by
the ratio of its actual weighted goals to its predicted weighted goals, so the
model converges to the joint MLE.  Both `att` and `dff` are renormalised to
mean 1 after every iteration; `μ` absorbs the scale and equals the global
decay-weighted average goals per team per match.

| Quantity | Meaning |
|---|---|
| **Attack strength** | MLE estimate of goals-scored multiplier, relative to average (=1.0) |
| **Defense strength** | MLE estimate of goals-conceded multiplier, relative to average (=1.0) |

Values > 1.0 mean above-average; < 1.0 mean below-average.  Because all teams
are fitted simultaneously, cross-confederation bridges in the data (WC
qualifiers, Copa America guests, friendlies) propagate quality signals globally —
eliminating the one-pass circularity that made a simpler opponent-adjustment
approach unreliable.

**Impact on WC 2022 backtest**: replacing raw averages with MLE improved overall
accuracy from 39.6 % to **47.9 %** (+8.3 pp), with home-win accuracy rising
from 44 % to 67 % and away-win accuracy from 40 % to 50 %.

---

### 3 · Expected goals (Dixon-Coles style)

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

### 4 · Dixon-Coles correction (ρ = −0.25)

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
low-scoring internationals.  ρ = −0.25 (stronger than the original paper's
−0.13) reflects the more defensive nature of WC group-stage football
compared to general international fixtures.

#### Draw bias / decision threshold

After computing P_home, P_draw, P_away from the DC grid, the outcome is
decided by this rule:

```
if P_win_best > P_draw + DRAW_BIAS:   predict win
else:                                  predict draw
```

`DRAW_BIAS` can be positive (conservative — requires the win to clearly exceed
the draw before committing) or negative (aggressive — commits to the best
decisive outcome even when P_draw is marginally higher).

**Calibration** (June 2026, MLE strengths, against WC 2010–2022 group stages):

| Metric | Before (raw averages) | After (MLE) |
|---|---|---|
| Historical draw rate (4 WCs) | — | **21.9 %** (target) |
| Predicted draw rate | 22.9 % | **20.8 %** |
| DRAW_BIAS | −0.010 | **+0.050** |

MLE produces more extreme win probabilities than raw goal averages (because
opponent quality spreads ratings further apart), so a positive bias
(`DRAW_BIAS = +0.050`) is needed to recover the historical ~22 % WC group-stage
draw rate.  Calibration is reproducible via:

```bash
python3 backtest_2022.py --calibrate
```

---

### 5 · Bayesian Monte Carlo (uncertainty + upsets)

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

**Poisson as the upset engine**: even with Brazil expected at λ=2.5 and
Morocco at λ=0.8, the Poisson distribution gives Morocco a non-trivial chance
of scoring 2 while Brazil scores 0 in any single draw.  The noise layer
amplifies this for data-sparse teams, but the Poisson variance is what makes
upsets structurally possible regardless of the strength gap.

**30 000 simulations per match**: in each run, noisy attack/defense values
are sampled, new λ values are derived, and goals are drawn from Poisson.
A partial DC rejection step is applied to low-score cells.  These simulations
are used only for the scoreline frequency table.

**Dead-rubber / stake adjustment**: pass `stake_home` or `stake_away` (float
0 < x ≤ 1.0) to `predict()` to model squad rotation or low motivation.  A
value < 1.0 scales the team's attack output proportionally, flattening their
expected goals and shifting the outcome toward a draw or upset.  Example use:

```python
# France has already qualified and will field a rotated XI vs Tunisia
pred = predict(data, attack, defense, sigma, global_avg,
               "France", "Tunisia", stake_home=0.75)
```

| Suggested value | Situation |
|---|---|
| **0.75** | Heavy rotation — team is qualified and protecting players |
| **0.85** | Partial rest — first choice but reduced intensity |
| **1.00** | Normal (default) — both teams playing to win |

Dead-rubber detection for all WC 2022 match-day 3 games is implemented in
`backtest_2022.py` and serves as a reference for setting stakes in future
tournaments.

**25 000 full-group simulations**: all six group matches are simulated
end-to-end with independent noisy parameters.  The fraction of runs in which
each team finishes top-2 becomes its *advancement probability* (Adv%).

---

## Output guide

| Field | Source |
|---|---|
| **Exp. goals (base)** | Point-estimate λ from MLE attack/defense strengths |
| **Most-likely score** | Peak of the DC-corrected analytical PMF (unconstrained) |
| **DC probability** | Win / draw / loss from the DC-corrected analytical PMF |
| **Prediction score** | Most probable scoreline consistent with the predicted outcome |
| **Top scorelines** | MC frequency table (30 000 runs) |
| **Adv%** | Full-group MC (25 000 runs) |
| **σ (noise)** | Per-team strength uncertainty |

---

## Group stage input files

Each file in `group_stage/` lists one team per line (exact spelling from `teams.txt`):

```
Mexico
South Africa
South Korea
Czech Republic
```

The filename (e.g. `group_A.txt`) is used to derive the `--group` label and output filename
(e.g. `Group A` → `group_a.csv`). Add or edit these files to customise which groups are run
by `run_all_groups.sh`.

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

## Backtesting

`backtest_2022.py` validates the predictor against actual WC 2022 group stage
results using only pre-tournament data (cut-off: 2022-11-19).

```bash
python3 backtest_2022.py               # full backtest with dead-rubber stakes
python3 backtest_2022.py --calibrate   # fast calibration of DRAW_BIAS (no MC)
```

**WC 2022 results (λ=0.00127 · DRAW_BIAS=+0.050 · MLE · dead-rubber stakes ON):**

| Metric | Raw averages (old) | DC MLE (current) |
|---|---|---|
| Overall accuracy | 39.6 % (19/48) | **47.9 % (23/48)** |
| Home wins correct | 44 % (8/18) | **67 % (12/18)** |
| Away wins correct | 40 % (8/20) | **50 % (10/20)** |
| Draws correct | 30 % (3/10) | 10 % (1/10) |
| Predicted draw rate | 35.4 % | 10.4 % |

Draws remain structurally hard for Poisson-based models: even with correct
draw-rate calibration in the 2026 predictions, individual draw calls are
difficult when the model assigns one team a clear probabilistic edge.  The draw
accuracy trade-off is the expected cost of the large gains in decisive outcomes.

---

## Limitations

- **No player-level data** (injuries, suspensions, form) — purely team-level
  historical aggregates.
- **Draws in knockout stage** are not modelled; this predictor is for group
  stage only (draw = 1 point each).
- **Correlation between matches** within the group is ignored; each match is
  simulated independently.
- **Dead-rubber stakes require manual input**: the predictor cannot
  automatically detect when a team has already qualified; callers must supply
  `stake_home`/`stake_away` based on the group standings at match time.
