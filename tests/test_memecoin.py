import numpy as np
import pytest
import json
from pathlib import Path

from src.memecoin import (MemecoinConfig, parse_trade_csv, simulate_memecoin,
                          summarize_memecoin, validate_memecoin_config)


def test_realized_trade_csv_and_fees_are_not_subtracted_twice():
    trades = parse_trade_csv("timestamp,entry_cost,realized_net_proceeds,fees\n"
                             "2025-01-01T00:00:00Z,100,180,20\n"
                             "2025-01-02T00:00:00Z,50,0,5\n")
    assert [trade.realized_multiple for trade in trades] == [1.8, 0.0]
    with pytest.raises(ValueError, match="sorted chronologically"):
        parse_trade_csv("timestamp,entry_cost,realized_net_proceeds,fees\n"
                        "2025-01-02,100,180,20\n2025-01-01,50,0,5\n")


def test_bankroll_accounting_fixed_reinvested_and_capital_constraint():
    config = MemecoinConfig(bankroll=100, position_size=20, trades=2, runs=1,
                            outcomes=(("realized 2x", 2.0, 1.0),))
    np.testing.assert_array_equal(simulate_memecoin(config), [[100, 120, 140]])
    reinvested = MemecoinConfig(**{**config.__dict__, "reinvest": True})
    np.testing.assert_array_equal(simulate_memecoin(reinvested), [[100, 120, 144]])
    loss = MemecoinConfig(bankroll=100, position_size=60, trades=2, runs=1,
                          outcomes=(("total loss", 0.0, 1.0),))
    paths = simulate_memecoin(loss)
    np.testing.assert_array_equal(paths, [[100, 40, 0]])
    assert summarize_memecoin(paths, loss)["risk_of_ruin"] == 1
    assert summarize_memecoin(paths, loss)["max_drawdown"]["median"] == 1


def test_sampling_reproducibility_and_empirical_bootstrap():
    config = MemecoinConfig(trades=12, runs=20, seed=7)
    first = simulate_memecoin(config)
    np.testing.assert_array_equal(first, simulate_memecoin(config))
    events = []
    simulate_memecoin(config, progress_callback=lambda done, total: events.append((done, total)))
    assert events == [(n, 12) for n in range(1, 13)]
    trades = parse_trade_csv("timestamp,entry_cost,realized_net_proceeds,fees\n"
                             "2025-01-01,100,0,2\n2025-01-02,100,200,2\n")
    empirical = MemecoinConfig(model="empirical_bootstrap", trades=5, runs=20)
    np.testing.assert_array_equal(simulate_memecoin(empirical, trades), simulate_memecoin(empirical, trades))
    with pytest.raises(ValueError, match="requires an imported"):
        simulate_memecoin(empirical)


def test_hypothetical_probabilities_are_validated():
    saved = json.loads((Path(__file__).resolve().parents[1] / "configs" / "memecoin.json").read_text())
    assert saved == MemecoinConfig().as_dict()
    raw = MemecoinConfig().as_dict()
    raw["outcomes"][0]["probability"] = 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        validate_memecoin_config(raw)
