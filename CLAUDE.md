# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the predictor for a single group
python3 predictor.py "Brazil" "Argentina" "France" "Germany"

# Run all groups (reads group_stage/*.txt, writes results/)
bash run_all_groups.sh

# Run tests (no network access required)
python3 -m unittest test_predictor -v

# Run a single test class
python3 -m unittest test_predictor.TestBuildDcGrid -v

# Backtest against WC 2022 group stage
python3 backtest_2022.py

# Calibrate DRAW_BIAS (fast, no MC)
python3 backtest_2022.py --calibrate

# Backtest against WC 2018 group stage
python3 backtest_2018.py

# Calibrate BLEND_ALPHA against WC 2018
python3 backtest_2018.py --calibrate-alpha

# Push predictions to Mon Petit Prono
python3 mpp_push.py results/all_groups_mpp.json --championship-id <ID>
python3 mpp_push.py --list-championships
python3 mpp_push.py results/all_groups_mpp.json --championship-id <ID> --dry-run
```

No third-party libraries — standard library only. Requires Python 3.10+ (uses `X | Y` union type syntax).

## Architecture

All prediction logic lives in a single file: `predictor.py`. Everything else calls into it.

### Data flow

1. `load_data()` — fetches `martj42/international_results` CSV from GitHub (~49k matches). Called once per run; no caching.
2. `compute_blended_strengths()` — fits attack/defense multipliers. This is the core stat layer (see below). Returns `attack`, `defense`, `sigma`, `global_avg` dicts keyed by team name.
3. `predict()` — two-track prediction for one match: analytical DC grid for outcome probabilities + 30k Monte Carlo runs for scoreline distribution.
4. `group_advancement_mc()` — 25k full-group simulations to compute Adv% per team.

### Statistical model layers (in order of application)

**Layer 1 — Dual time-window blend** (`compute_blended_strengths`): runs `compute_strengths` twice (long λ=0.000347 ≈ 5.5y half-life; short λ=0.00762 ≈ 3m half-life), then linearly blends with `BLEND_ALPHA=0.75`. Long window is the stable baseline; short window captures pre-tournament peak form. Alpha was calibrated on WC 2018 via `backtest_2018.py --calibrate-alpha`.

**Layer 2 — Dixon-Coles MLE** (`compute_strengths`): fits attack/defense multipliers for every team in the dataset simultaneously using 100 iterations of multiplicative EM. Each match observation is weighted by `exp(-λ·days) × tournament_weight`. Normalised so mean(attack) = mean(defense) = 1.0; `mu` absorbs global scale.

**Layer 3 — Expected goals** (`predict`): `λ_home = attack[home] × defense[away] × global_avg × home_advantage`. `HOME_ADVANTAGE=1.08` applies only to WC2026 host nations (Mexico, Canada, USA).

**Layer 4 — Dixon-Coles correction** (`build_dc_grid`): multiplicative correction `dc_tau` applied to the four low-score cells (0-0, 1-0, 0-1, 1-1) with `DC_RHO=-0.25`. The full 11×11 PMF is normalised to sum=1.

**Layer 5 — Draw bias threshold** (`predict`): outcome is decided by `P_win_best > P_draw + DRAW_BIAS`. `DRAW_BIAS=+0.050` is calibrated to reproduce the ~22% historical WC group-stage draw rate. Calibration is reproducible via `backtest_2022.py --calibrate`.

**Layer 6 — FIFA ranking prior** (`apply_fifa_prior`): after MLE fitting, attack/defense are shrunk toward a linear prior derived from FIFA rankings: `effective = (n_eff·data + n_prior·prior) / (n_eff + n_prior)`. `FIFA_N_PRIOR=4.0` pseudo-matches. Minimal effect on data-rich teams (large n_eff); corrects debutants and sparse qualifiers (Canada, Qatar). Applied via `FIFA_RANKS_2026` in production and tournament-specific dicts in backtests.

**Layer 7 — Bayesian uncertainty / MC** (`predict`, `group_advancement_mc`): per-team `sigma` scales with `1/√(n_eff/REF_N_EFF)` where `n_eff = Σ decay_weights`. At prediction time, attack/defense are perturbed log-normally before sampling Poisson goals.

### Key constants in `predictor.py`

| Constant | Value | Purpose |
|---|---|---|
| `BLEND_ALPHA` | 0.75 | Long-window weight in dual blend |
| `DECAY_LAMBDA_LONG` | 0.000347 | ≈ 5.5y half-life for stable baseline |
| `DECAY_LAMBDA_SHORT` | 0.00762 | ≈ 3m half-life for form peak |
| `DC_RHO` | −0.25 | Dixon-Coles low-score correction |
| `DRAW_BIAS` | +0.050 | Win/draw decision threshold |
| `BASE_SIGMA` | 0.25 | Uncertainty at `REF_N_EFF=40` effective matches |
| `TOURNAMENT_WEIGHTS` | dict | Per-tournament match importance multiplier |
| `FIFA_N_PRIOR` | 4.0 | Pseudo-match count for FIFA ranking prior |
| `FIFA_RANKS_2026` | dict | Current FIFA rankings for WC 2026 teams |
| `N_MATCH_SIM` | 30 000 | MC runs per match (scoreline %) |
| `N_GROUP_SIM` | 25 000 | MC runs for Adv% |

### Supporting files

- `group_stage/group_*.txt` — one team per line, exact spelling from `teams.txt`. Filename drives the group label and output slug.
- `teams.txt` — canonical team names; validated before each run with fuzzy-match suggestions.
- `run_all_groups.sh` — loops over `group_stage/`, runs predictor, then merges per-group JSON files into `results/all_groups.json` and `results/all_groups_mpp.json`.
- `mpp_push.py` — posts predictions to mpp.football via private API. Tokens cached in `.mpp_tokens.json` (gitignored).
- `results/` — output directory, gitignored, regenerated locally.

### Dead-rubber / stake adjustment

Pass `stake_home` or `stake_away` (float 0 < x ≤ 1.0) to `predict()` to model squad rotation. Scales the team's attack before computing λ. `backtest_2022.py` contains a reference implementation of dead-rubber detection for all WC 2022 MD3 fixtures.
