from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.spot import DEFAULT_SPOT, estimate_history, parse_price_csv, simulate_spot, validate_spot_config


def _prices(days=40):
    lines = ["date,BTC,HYPE,USDT,XAUT"]
    from datetime import date, timedelta
    for i in range(days):
        day = date(2025, 1, 1) + timedelta(days=i)
        lines.append(f"{day},{100*np.exp(.001*i)},{10*np.exp(.002*i)},1,{50*np.exp(.0005*i)}")
    return "\n".join(lines) + "\n"


def test_default_weights_and_arbitrary_asset_validation():
    config = validate_spot_config(deepcopy(DEFAULT_SPOT))
    saved = json.loads((Path(__file__).resolve().parents[1] / "configs" / "spot.json").read_text())
    assert saved == DEFAULT_SPOT
    assert list(config.data["weights"].values()) == [0.5, 0.2, 0.15, 0.15]
    invalid = deepcopy(DEFAULT_SPOT)
    invalid["weights"]["BTC"] = .51
    with pytest.raises(ValueError, match="sum to 1"):
        validate_spot_config(invalid)
    invalid = deepcopy(DEFAULT_SPOT)
    invalid["future_correlation"][0][1] = 0.99
    with pytest.raises(ValueError, match="symmetric"):
        validate_spot_config(invalid)
    invalid = deepcopy(DEFAULT_SPOT)
    invalid["future_correlation"] = [[1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_spot_config(invalid)
    one_asset = deepcopy(DEFAULT_SPOT)
    one_asset.update(assets=["TEST"], weights={"TEST": 1.0}, future_expected_returns={"TEST": 0.0},
                     future_volatility={"TEST": 0.0}, future_correlation=[[1.0]],
                     years=1, runs=5)
    paths = simulate_spot(validate_spot_config(one_asset))
    np.testing.assert_array_equal(paths, np.full((5, 13), 10_000.0))


def test_price_alignment_and_historical_covariance():
    history = parse_price_csv(_prices(), tuple(DEFAULT_SPOT["assets"]))
    estimates = estimate_history(history)
    np.testing.assert_allclose(estimates["mean_daily_log_return"], [.001, .002, 0, .0005], atol=1e-12)
    assert np.asarray(estimates["annualized_covariance"]).shape == (4, 4)
    assert estimates["annualized_volatility"][2] == 0
    assert "USDT" in estimates["undefined_correlation_assets"]
    with pytest.raises(ValueError, match="contiguous"):
        parse_price_csv(_prices().replace("2025-01-02", "2025-01-03"), tuple(DEFAULT_SPOT["assets"]))
    with pytest.raises(ValueError, match="aligned price"):
        parse_price_csv(_prices().replace("2025-01-02,", "2025-01-02,,", 1), tuple(DEFAULT_SPOT["assets"]))


def test_covariance_uses_sample_daily_log_returns_and_365_day_scaling():
    from datetime import date, timedelta
    daily = np.tile(np.array([[.01, -.01], [-.02, .03], [.005, .02]]), (10, 1))
    prices = np.vstack([np.ones((1, 2)) * 100, 100 * np.exp(np.cumsum(daily, axis=0))])
    rows = ["date,A,B"]
    for index, pair in enumerate(prices):
        rows.append(f"{date(2025, 1, 1) + timedelta(days=index)},{pair[0]},{pair[1]}")
    history = parse_price_csv("\n".join(rows), ("A", "B"))
    estimated = estimate_history(history)
    np.testing.assert_allclose(estimated["annualized_covariance"], np.cov(daily, rowvar=False, ddof=1) * 365)
    np.testing.assert_allclose(estimated["historical_correlation"], np.corrcoef(daily, rowvar=False))


def test_spot_gbm_and_bootstrap_reproducibility_and_progress():
    raw = deepcopy(DEFAULT_SPOT)
    raw.update(years=1, runs=20, seed=5)
    config = validate_spot_config(raw)
    events = []
    first = simulate_spot(config, progress_callback=lambda done, total: events.append((done, total)))
    np.testing.assert_array_equal(first, simulate_spot(config))
    assert events == [(n, 12) for n in range(1, 13)]
    raw["method"] = "block_bootstrap"
    bootstrap = validate_spot_config(raw)
    history = parse_price_csv(_prices(), tuple(raw["assets"]))
    np.testing.assert_array_equal(simulate_spot(bootstrap, history), simulate_spot(bootstrap, history))
    with pytest.raises(ValueError, match="requires imported"):
        simulate_spot(bootstrap)


def test_stress_is_applied_once_before_rebalancing():
    raw = deepcopy(DEFAULT_SPOT)
    raw.update(assets=["USDT"], weights={"USDT": 1.0}, future_expected_returns={"USDT": 0.0},
               future_volatility={"USDT": 0.0}, future_correlation=[[1.0]],
               years=1, runs=3)
    raw["stablecoin_depeg"] = dict(enabled=True, asset="USDT", month=2, multiplier=0.8)
    config = validate_spot_config(raw)
    paths = simulate_spot(config)
    np.testing.assert_array_equal(paths[:, :2], np.full((3, 2), 10_000.0))
    np.testing.assert_array_equal(paths[:, 2:], np.full((3, 11), 8_000.0))
