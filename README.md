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

### 1 · Dual time-window decay blend

The model blends two separate exponential decay windows and mixes their
attack/defense estimates:

```
w_long(t)  = exp(−λ_long  · days_ago)   λ_long  = 0.000347  (half-life ≈ 5.5 years)
w_short(t) = exp(−λ_short · days_ago)   λ_short = 0.00762   (half-life ≈ 3 months)

att_eff = α · att_long + (1−α) · att_short     α = 0.75
```

**Long window** (5.5 y half-life): provides a stable baseline strength that
avoids being polluted by transient slumps (Nations League rotations, poorly
motivated dead-rubber runs).

**Short window** (3 m half-life): captures tournament-peak momentum — teams
that peaked in the 3 months before the tournament (e.g. Japan, Morocco, Australia
at WC 2022) gain a brief but measurable signal.

**Blend weight α = 0.75**: calibrated analytically on the WC 2018 group stage
(`python3 backtest_2018.py --calibrate-alpha`); 75 % long / 25 % short.

**Impact**: this fixes the systematic error in the single-window (λ = 0.00127,
1.5 y half-life) where France's poor Nations League 2022 run lowered their
rating enough that the model predicted losses to Australia and Denmark — both of
which France won comfortably (4-1 and 2-1).  The longer stable window preserves
France's structural quality regardless of short-term slumps.

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
decided by:

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

Two WC group stages are used as benchmarks.  Both scripts use only
pre-tournament data for predictions and read actual results from the live
dataset (no hardcoded scores for WC 2018).

```bash
python3 backtest_2022.py               # WC 2022 — stakes + FIFA ranking prior
python3 backtest_2022.py --calibrate   # fast DRAW_BIAS calibration (no MC)
python3 backtest_2018.py               # WC 2018 — stakes, no prior
python3 backtest_2018.py --calibrate-alpha  # fast BLEND_ALPHA calibration (no MC)
```

**WC 2022** (cut-off 2022-11-19 · dead-rubber stakes ON · FIFA prior ON):

| Metric | Raw averages (old) | DC MLE single-window | DC MLE dual-window (current) |
|---|---|---|---|
| Overall accuracy | 39.6 % (19/48) | 50.0 % (24/48) | **47.9 % (23/48)** |
| Home wins correct | 44 % (8/18) | 72 % (13/18) | **67 % (12/18)** |
| Away wins correct | 40 % (8/20) | 50 % (10/20) | **45 % (9/20)** |
| Draws correct | 30 % (3/10) | 10 % (1/10) | **20 % (2/10)** |
| Predicted draw rate | 35.4 % | 8.3 % | 12.5 % |

**WC 2018** (cut-off 2018-06-14 · dead-rubber stakes ON · no prior):

| Metric | DC MLE single-window | DC MLE dual-window (current) |
|---|---|---|
| Overall accuracy | 60.4 % (29/48) | **64.6 % (31/48)** |
| Home wins correct | 84 % (16/19) | **89 % (17/19)** |
| Away wins correct | 50 % (10/20) | **60 % (12/20)** |
| Draws correct | 33 % (3/9) | 22 % (2/9) |
| Predicted draw rate | 22.9 % | 12.5 % |

**Combined** (96 matches, WC 2018 + 2022): single-window 53/96 (55.2 %) → dual-window **54/96 (56.2 %, +1.0 pp)**.

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
