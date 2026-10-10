import numpy as np

from src.config import validate_config
from src.portfolio import rebalance, should_rebalance


def test_weights_sum_to_one(raw_config):
    config = validate_config(raw_config)
    assert sum(config.weights) == 1
    raw_config["weights"]["BTC"] = 0.31
    import pytest
    with pytest.raises(ValueError, match="sum to 1"):
        validate_config(raw_config)


def test_rebalance_preserves_value_and_target_weights():
    holdings = np.array([[200., 300., 500.], [100., 400., 100.]])
    weights = np.array([.3, .4, .3])
    result = rebalance(holdings, weights)
    np.testing.assert_allclose(result, np.array([[300., 400., 300.], [180., 240., 180.]]))
    np.testing.assert_allclose(result.sum(axis=1), holdings.sum(axis=1))
    assert not should_rebalance(1, "annual")
    assert should_rebalance(12, "annual")
    assert should_rebalance(1, "monthly")
    assert not should_rebalance(12, "none")


def test_no_rebalance_differs_from_monthly_when_assets_diverge(config):
    from dataclasses import replace
    from src.engine import simulate
    fixed = replace(config, annual_volatility=(0., 0., 0.), annual_expected_returns=(.5, .0, .0))
    hold = simulate(replace(fixed, rebalancing="none"), max_years=1)
    monthly = simulate(replace(fixed, rebalancing="monthly"), max_years=1)
    assert hold[0, -1] > monthly[0, -1]
