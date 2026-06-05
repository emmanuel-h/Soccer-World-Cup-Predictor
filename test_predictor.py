"""Unit tests for predictor.py — no network access required."""

import json
import math
import pathlib
import random
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

from predictor import (
    _fit_logistic,
    _sigmoid,
    build_dc_grid,
    check_teams,
    dc_tau,
    deterministic_standings,
    draw_clf_prob,
    generate_fixtures,
    h2h_stats,
    load_known_teams,
    most_likely_score,
    most_likely_score_for_outcome,
    poisson_pmf,
    probs_from_grid,
    sample_poisson,
    write_matches_csv,
    write_matches_json,
    write_matches_mpp,
)


# ── poisson_pmf ────────────────────────────────────────────────────────────────

class TestPoissonPmf(unittest.TestCase):

    def test_zero_lambda_returns_one_for_k0(self):
        self.assertEqual(poisson_pmf(0, 0.0), 1.0)

    def test_zero_lambda_returns_zero_for_k1(self):
        self.assertEqual(poisson_pmf(1, 0.0), 0.0)

    def test_k_at_limit_returns_zero(self):
        self.assertEqual(poisson_pmf(25, 2.0), 0.0)

    def test_known_value(self):
        # P(X=0 | λ=1) = e^-1
        self.assertAlmostEqual(poisson_pmf(0, 1.0), math.exp(-1), places=10)

    def test_sums_to_one(self):
        total = sum(poisson_pmf(k, 2.5) for k in range(25))
        self.assertAlmostEqual(total, 1.0, delta=0.001)


# ── sample_poisson ─────────────────────────────────────────────────────────────

class TestSamplePoisson(unittest.TestCase):

    def test_mean_close_to_lambda(self):
        random.seed(42)
        lam = 1.8
        samples = [sample_poisson(lam) for _ in range(10_000)]
        self.assertAlmostEqual(sum(samples) / len(samples), lam, delta=0.1)

    def test_returns_non_negative(self):
        random.seed(0)
        self.assertTrue(all(sample_poisson(1.5) >= 0 for _ in range(100)))


# ── dc_tau ─────────────────────────────────────────────────────────────────────

class TestDcTau(unittest.TestCase):

    def setUp(self):
        self.lh, self.la, self.rho = 1.5, 1.2, -0.10

    def test_cell_00(self):
        expected = 1.0 - self.lh * self.la * self.rho
        self.assertAlmostEqual(dc_tau(0, 0, self.lh, self.la, self.rho), expected)

    def test_cell_10(self):
        expected = 1.0 + self.la * self.rho
        self.assertAlmostEqual(dc_tau(1, 0, self.lh, self.la, self.rho), expected)

    def test_cell_01(self):
        expected = 1.0 + self.lh * self.rho
        self.assertAlmostEqual(dc_tau(0, 1, self.lh, self.la, self.rho), expected)

    def test_cell_11(self):
        expected = 1.0 - self.rho
        self.assertAlmostEqual(dc_tau(1, 1, self.lh, self.la, self.rho), expected)

    def test_other_cell_returns_one(self):
        self.assertEqual(dc_tau(2, 2, self.lh, self.la, self.rho), 1.0)

    def test_clamped_to_zero(self):
        # Large positive rho forces (0,0) negative → clamp to 0
        self.assertEqual(dc_tau(0, 0, 2.0, 2.0, 1.0), 0.0)


# ── build_dc_grid ──────────────────────────────────────────────────────────────

class TestBuildDcGrid(unittest.TestCase):

    def test_sums_to_one(self):
        grid = build_dc_grid(1.5, 1.2)
        self.assertAlmostEqual(sum(grid.values()), 1.0, places=9)

    def test_all_non_negative(self):
        grid = build_dc_grid(1.5, 1.2)
        self.assertTrue(all(v >= 0 for v in grid.values()))

    def test_contains_zero_zero(self):
        grid = build_dc_grid(1.5, 1.2)
        self.assertIn((0, 0), grid)


# ── probs_from_grid ────────────────────────────────────────────────────────────

class TestProbsFromGrid(unittest.TestCase):

    def test_sum_to_one(self):
        grid = build_dc_grid(1.5, 1.2)
        p_h, p_d, p_a = probs_from_grid(grid)
        self.assertAlmostEqual(p_h + p_d + p_a, 1.0, places=9)

    def test_symmetry_equal_lambdas(self):
        grid = build_dc_grid(1.5, 1.5)
        p_h, _, p_a = probs_from_grid(grid)
        self.assertAlmostEqual(p_h, p_a, delta=0.001)

    def test_stronger_home_favoured(self):
        grid = build_dc_grid(3.0, 0.5)
        p_h, _, p_a = probs_from_grid(grid)
        self.assertGreater(p_h, p_a)


# ── most_likely_score ──────────────────────────────────────────────────────────

class TestMostLikelyScore(unittest.TestCase):

    def test_returns_max_cell(self):
        grid = {(0, 0): 0.1, (1, 0): 0.4, (0, 1): 0.2, (1, 1): 0.3}
        self.assertEqual(most_likely_score(grid), (1, 0))


# ── generate_fixtures ──────────────────────────────────────────────────────────

class TestGenerateFixtures(unittest.TestCase):

    def test_two_teams_one_fixture(self):
        fixtures = generate_fixtures(["A", "B"])
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0][:2], ("A", "B"))

    def test_four_teams_six_fixtures(self):
        self.assertEqual(len(generate_fixtures(["A", "B", "C", "D"])), 6)

    def test_labels_sequential(self):
        fixtures = generate_fixtures(["A", "B", "C"])
        self.assertEqual([f[2] for f in fixtures], ["Match 1", "Match 2", "Match 3"])


# ── h2h_stats ─────────────────────────────────────────────────────────────────

def _row(home, away, hs, as_, days_ago):
    return {
        "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_,
        "date": datetime.now() - timedelta(days=days_ago),
    }


class TestH2hStats(unittest.TestCase):

    def test_no_matches_returns_none(self):
        self.assertIsNone(h2h_stats([], "A", "B"))

    def test_counts_wins_draws(self):
        data = [
            _row("Brazil", "Argentina", 2, 1, 100),  # Brazil win
            _row("Argentina", "Brazil",  1, 1, 200),  # draw
            _row("Brazil", "Argentina", 0, 2, 300),  # Argentina win
        ]
        n, w1, d, w2 = h2h_stats(data, "Brazil", "Argentina")
        self.assertEqual(n, 3)
        self.assertEqual(w1, 1)
        self.assertEqual(d, 1)
        self.assertEqual(w2, 1)

    def test_ignores_unrelated_matches(self):
        data = [
            _row("Brazil", "France", 1, 0, 100),
            _row("Brazil", "Argentina", 2, 0, 200),
        ]
        n, w1, _, _ = h2h_stats(data, "Brazil", "Argentina")
        self.assertEqual(n, 1)
        self.assertEqual(w1, 1)

    def test_old_matches_excluded(self):
        data = [_row("Brazil", "Argentina", 2, 0, 365 * 25)]  # 25 years ago
        self.assertIsNone(h2h_stats(data, "Brazil", "Argentina"))


# ── deterministic_standings ────────────────────────────────────────────────────

class TestDeterministicStandings(unittest.TestCase):

    def setUp(self):
        self.predictions = [
            {"home": "A", "away": "B", "definitive_score": (2, 1)},  # A wins
            {"home": "A", "away": "C", "definitive_score": (1, 1)},  # draw
            {"home": "B", "away": "C", "definitive_score": (0, 1)},  # C wins
        ]

    def test_points(self):
        _, pts, _, _, _, _, _ = deterministic_standings(self.predictions, ["A", "B", "C"])
        self.assertEqual(pts["A"], 4)  # win + draw
        self.assertEqual(pts["B"], 0)
        self.assertEqual(pts["C"], 4)  # draw + win

    def test_goals(self):
        _, _, _, _, _, gf, ga = deterministic_standings(self.predictions, ["A", "B", "C"])
        self.assertEqual(gf["A"], 3)
        self.assertEqual(ga["A"], 2)

    def test_ordering_by_points_then_gd(self):
        ordered, pts, _, _, _, gf, ga = deterministic_standings(self.predictions, ["A", "B", "C"])
        # A: 4pts, GD +1 — C: 4pts, GD +1 — both equal; then B: 0pts
        self.assertEqual(ordered[-1], "B")


# ── load_known_teams ───────────────────────────────────────────────────────────

class TestLoadKnownTeams(unittest.TestCase):

    def test_contains_expected_teams(self):
        teams = load_known_teams()
        for name in ("Brazil", "South Korea", "United States", "Ivory Coast"):
            self.assertIn(name, teams)

    def test_no_comment_lines(self):
        teams = load_known_teams()
        self.assertFalse(any(t.startswith("#") for t in teams))

    def test_no_empty_strings(self):
        teams = load_known_teams()
        self.assertNotIn("", teams)


# ── check_teams ────────────────────────────────────────────────────────────────

class TestCheckTeams(unittest.TestCase):

    KNOWN = ["Brazil", "Argentina", "France", "South Korea"]

    def test_valid_teams_pass(self):
        check_teams(["Brazil", "France"], self.KNOWN)  # must not raise

    def test_unknown_team_exits(self):
        with self.assertRaises(SystemExit):
            check_teams(["Brazil", "Argentin"], self.KNOWN)

    def test_error_message_names_bad_team(self):
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", new_callable=StringIO) as mock_err:
                check_teams(["Argentin"], self.KNOWN)
                self.assertIn("Argentin", mock_err.getvalue())

    def test_suggestion_offered_for_close_match(self):
        buf = StringIO()
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", buf):
                check_teams(["Argentin"], self.KNOWN)
        self.assertIn("Argentina", buf.getvalue())

    def test_no_suggestion_for_gibberish(self):
        buf = StringIO()
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", buf):
                check_teams(["ZZZZZZ"], self.KNOWN)
        self.assertNotIn("did you mean", buf.getvalue())


# ── most_likely_score_for_outcome ──────────────────────────────────────────────

class TestMostLikelyScoreForOutcome(unittest.TestCase):

    def test_home_outcome_yields_home_win_scoreline(self):
        grid = build_dc_grid(2.5, 0.8)  # strong home side
        h, a = most_likely_score_for_outcome(grid, "home")
        self.assertGreater(h, a)

    def test_away_outcome_yields_away_win_scoreline(self):
        grid = build_dc_grid(0.8, 2.5)  # strong away side
        h, a = most_likely_score_for_outcome(grid, "away")
        self.assertLess(h, a)

    def test_draw_outcome_yields_level_scoreline(self):
        grid = build_dc_grid(1.5, 1.5)
        h, a = most_likely_score_for_outcome(grid, "draw")
        self.assertEqual(h, a)


# ── _sigmoid ──────────────────────────────────────────────────────────────────

class TestSigmoid(unittest.TestCase):

    def test_zero_returns_half(self):
        self.assertAlmostEqual(_sigmoid(0.0), 0.5)

    def test_large_positive_approaches_one(self):
        self.assertAlmostEqual(_sigmoid(100.0), 1.0, places=5)

    def test_large_negative_approaches_zero(self):
        self.assertAlmostEqual(_sigmoid(-100.0), 0.0, places=5)

    def test_known_value(self):
        import math
        self.assertAlmostEqual(_sigmoid(1.0), 1 / (1 + math.exp(-1)), places=10)


# ── _fit_logistic ──────────────────────────────────────────────────────────────

class TestFitLogistic(unittest.TestCase):

    def test_perfectly_separable_data(self):
        # Intercept + 1 feature; positives are x>0, negatives x<0
        X = [[1.0, float(i)] for i in range(-10, 11) if i != 0]
        y = [1.0 if x[1] > 0 else 0.0 for x in X]
        coeffs = _fit_logistic(X, y, lr=0.1, n_iter=3000)
        # Intercept should be near 0, feature coeff should be positive
        self.assertAlmostEqual(coeffs[0], 0.0, delta=0.3)
        self.assertGreater(coeffs[1], 1.0)

    def test_returns_correct_length(self):
        X = [[1.0, 0.5, 0.3] for _ in range(20)]
        y = [1.0] * 10 + [0.0] * 10
        coeffs = _fit_logistic(X, y)
        self.assertEqual(len(coeffs), 3)


# ── draw_clf_prob ──────────────────────────────────────────────────────────────

class TestDrawClfProb(unittest.TestCase):

    def test_output_in_zero_one(self):
        p = draw_clf_prob(1.5, 1.5, 0.28, [-1.0, -1.0, 1.0])
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_more_equal_teams_higher_draw_prob(self):
        # With negative norm_diff coeff, equal teams (norm_diff=0) > mismatched
        coeffs = [-1.0, -2.0, 2.0]
        p_equal    = draw_clf_prob(1.5, 1.5, 0.30, coeffs)   # norm_diff = 0
        p_unequal  = draw_clf_prob(2.5, 0.8, 0.15, coeffs)   # norm_diff > 0
        self.assertGreater(p_equal, p_unequal)


# ── shared fixture for write tests ────────────────────────────────────────────

_SAMPLE_PREDICTION = {
    "home": "Brazil", "away": "France",
    "definitive_score": (2, 1),
    "lam_h": 1.80, "lam_a": 1.20,
    "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25,
    "result": "Brazil WIN",
}


# ── write_matches_csv ──────────────────────────────────────────────────────────

class TestWriteMatchesCsv(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mktemp(suffix=".csv"))

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_creates_file_with_header_and_row(self):
        write_matches_csv(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        lines = self.tmp.read_text().splitlines()
        self.assertIn("HomeTeam", lines[0])
        self.assertIn("Brazil", lines[1])

    def test_appends_without_duplicate_header(self):
        write_matches_csv(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        write_matches_csv(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        lines = self.tmp.read_text().splitlines()
        header_count = sum(1 for line in lines if "HomeTeam" in line)
        self.assertEqual(header_count, 1)
        self.assertEqual(len(lines), 3)  # header + 2 data rows

    def test_group_label_is_written(self):
        write_matches_csv(self.tmp, "Group Z", [_SAMPLE_PREDICTION])
        self.assertIn("Group Z", self.tmp.read_text())


# ── write_matches_json ─────────────────────────────────────────────────────────

class TestWriteMatchesJson(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_creates_valid_json_array(self):
        write_matches_json(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        data = json.loads(self.tmp.read_text())
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["homeTeam"], "Brazil")
        self.assertEqual(data[0]["homeScore"], 2)

    def test_appends_to_existing(self):
        write_matches_json(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        write_matches_json(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        data = json.loads(self.tmp.read_text())
        self.assertEqual(len(data), 2)


# ── write_matches_mpp ──────────────────────────────────────────────────────────

class TestWriteMatchesMpp(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_creates_mpp_format(self):
        write_matches_mpp(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        data = json.loads(self.tmp.read_text())
        self.assertEqual(len(data), 1)
        record = data[0]
        self.assertEqual(record["homeTeam"], "Brazil")
        self.assertEqual(record["homeScore"], 2)
        self.assertEqual(record["awayScore"], 1)
        self.assertEqual(record["originPage"], "home")

    def test_appends_to_existing(self):
        write_matches_mpp(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        write_matches_mpp(self.tmp, "Group A", [_SAMPLE_PREDICTION])
        data = json.loads(self.tmp.read_text())
        self.assertEqual(len(data), 2)


if __name__ == "__main__":
    unittest.main()
