"""Named, non-overwriting CSV/JSON exports for the additional research modes."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from src.analytics import percentile_paths
from src.memecoin import MemecoinConfig, Trade, simulate_memecoin, summarize_memecoin
from src.spot import PriceHistory, SpotConfig, simulate_spot, summarize_spot


ROOT = Path(__file__).resolve().parents[1]


def _directory(mode: str, run_name: str | None) -> Path:
    if run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run name must contain 1–64 letters, digits, underscores or hyphens")
    name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = ROOT / "results" / "runs" / f"{mode}-{name}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _export(mode: str, paths: np.ndarray, report: dict, run_name: str | None) -> Path:
    directory = _directory(mode, run_name)
    with (directory / "simulation_summary.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    initial = float(paths[0, 0])
    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = np.divide(peaks - paths, peaks, out=np.zeros_like(paths), where=peaks > 0).max(axis=1)
    with (directory / "final_portfolio_values.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("scenario", "final_value_usd", "net_profit_usd", "max_drawdown"))
        for index, (final, drawdown) in enumerate(zip(paths[:, -1], drawdowns), 1):
            writer.writerow((index, float(final), float(final - initial), float(drawdown)))
    unit = "trade" if mode == "memecoin" else "month"
    with (directory / "percentile_paths.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow((unit, "p5_usd", "median_usd", "p95_usd"))
        for step, row in enumerate(percentile_paths(paths)):
            writer.writerow((step, *map(float, row)))
    return directory


def run_memecoin_experiment(
    config: MemecoinConfig, historical_trades: tuple[Trade, ...] = (),
    run_name: str | None = None, progress_callback=None,
) -> tuple[np.ndarray, dict, Path]:
    paths = simulate_memecoin(config, historical_trades, progress_callback)
    report = summarize_memecoin(paths, config)
    report["historical_sample_size"] = len(historical_trades) if config.model == "empirical_bootstrap" else None
    report["source_label"] = ("imported realized trades, sampled with replacement" if config.model == "empirical_bootstrap"
                              else "user-defined hypothetical probabilities; not observed statistics")
    directory = _export("memecoin", paths, report, run_name)
    return paths, report, directory


def run_spot_experiment(
    config: SpotConfig, history: PriceHistory | None = None,
    run_name: str | None = None, progress_callback=None,
) -> tuple[np.ndarray, dict, Path]:
    paths = simulate_spot(config, history, progress_callback)
    report = summarize_spot(paths, config, history)
    directory = _export("spot", paths, report, run_name)
    return paths, report, directory
