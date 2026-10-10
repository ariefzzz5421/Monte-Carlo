import numpy as np

from src.analytics import max_drawdowns, percentile_paths, summarize
from src.config import validate_config
import pytest


@pytest.mark.parametrize("change,match", [
    ({"initial_capital": 0}, "positive"),
    ({"simulations": 0}, "integer"),
    ({"years": 3}, "1, 5, or 10"),
    ({"seed": -1}, "integer"),
    ({"rebalancing": "weekly"}, "rebalancing"),
])
def test_invalid_scalar_config(raw_config, change, match):
    raw_config.update(change)
    with pytest.raises(ValueError, match=match):
        validate_config(raw_config)


@pytest.mark.parametrize("matrix,match", [
    ([[1, .2, 0], [.3, 1, 0], [0, 0, 1]], "symmetric"),
    ([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]], "positive semidefinite"),
    ([[.9, 0, 0], [0, 1, 0], [0, 0, 1]], "diagonal"),
    ([[1, 1.2, 0], [1.2, 1, 0], [0, 0, 1]], "between -1 and 1"),
])
def test_invalid_correlations(raw_config, matrix, match):
    raw_config["correlation"] = matrix
    with pytest.raises(ValueError, match=match):
        validate_config(raw_config)


def test_valid_semidefinite_correlation(raw_config):
    raw_config["correlation"] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    validate_config(raw_config)


def test_percentiles_drawdowns_and_terminal_loss_risk():
    paths = np.array([[100., 120., 90.], [100., 100., 150.], [100., 80., 70.], [100., 100., 110.]])
    np.testing.assert_allclose(max_drawdowns(paths), [0.25, 0., 0.30, 0.])
    percentiles = percentile_paths(paths)
    assert percentiles.shape == (3, 3)
    assert percentiles[0, 1] == 100
    stats = summarize(paths, 100., 1)
    assert stats["final_value_usd"]["median"] == 100
    assert stats["probability_below_initial"] == .5
    assert stats["terminal_loss_risk_usd"]["expected_shortfall_95"] == 30
    assert stats["max_drawdown"]["median"] == .125
