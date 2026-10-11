from dataclasses import replace

import numpy as np

from src.engine import simulate


def test_seed_reproduces_finite_nonnegative_paths(config):
    first = simulate(config, max_years=1)
    second = simulate(config, max_years=1)
    assert first.shape == (24, 13)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert (first >= 0).all()


def test_zero_volatility_matches_compounded_simple_returns(config):
    fixed = replace(config, annual_volatility=(0.0, 0.0, 0.0), rebalancing="monthly")
    paths = simulate(fixed, max_years=1)
    monthly_growth = np.dot(np.asarray(fixed.weights), np.power(1 + np.asarray(fixed.annual_expected_returns), 1 / 12))
    expected = fixed.initial_capital * monthly_growth ** np.arange(13)
    np.testing.assert_allclose(paths, np.tile(expected, (fixed.simulations, 1)), rtol=1e-12)


def test_perfect_correlation_is_supported(config):
    identical = replace(config, correlation=((1., 1., 1.),) * 3)
    paths = simulate(identical, max_years=1)
    assert np.isfinite(paths).all()


def test_progress_reports_completed_months_without_changing_results(config):
    progress = []
    baseline = simulate(config, max_years=1)
    observed = simulate(config, max_years=1, progress_callback=lambda done, total: progress.append((done, total)))
    np.testing.assert_array_equal(observed, baseline)
    assert progress == [(month, 12) for month in range(1, 13)]


def test_progress_callback_is_not_called_before_first_completed_step(config):
    seen = []
    simulate(config, max_years=1, progress_callback=lambda done, total: seen.append(done))
    assert seen[0] == 1
    assert seen[-1] == 12
