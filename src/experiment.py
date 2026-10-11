"""Shared experiment and export flow for the CLI and local dashboard."""

from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from src.analytics import max_drawdowns, percentile_paths, summarize
from src.config import Config
from src.engine import simulate
from src.visualization import create_charts

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    config: Config
    paths: np.ndarray
    report: dict
    directory: Path


def _validate_run_name(run_name: str | None) -> None:
    if run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run name must be 1–64 letters, digits, underscores or hyphens")


def _output_dir(run_name: str | None) -> Path:
    results = ROOT / "results"
    if run_name:
        target = results / "runs" / run_name
        target.mkdir(parents=True, exist_ok=False)
        return target
    results.mkdir(exist_ok=True)
    existing_root_outputs = any(item.name not in (".gitkeep", "runs") for item in results.iterdir())
    if not existing_root_outputs:
        return results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = results / "runs" / timestamp
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_experiment(
    config: Config,
    *,
    run_name: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    stage_callback: Callable[[str], None] | None = None,
) -> Experiment:
    """Simulate the same 10Y scenarios for selected and comparison horizons.

    Progress callbacks report only completed simulation months. Exports keep the
    original summary/CSV/chart names and terminal-loss conventions.
    """
    _validate_run_name(run_name)
    if run_name and (ROOT / "results" / "runs" / run_name).exists():
        raise FileExistsError(f"run already exists: {run_name}")
    paths = simulate(config, max_years=10, progress_callback=progress_callback)
    selected = paths[:, :config.years * 12 + 1]
    summary = summarize(selected, config.initial_capital, config.years)
    comparison = {
        str(year): summarize(paths[:, :year * 12 + 1], config.initial_capital, year)
        for year in (1, 5, 10)
    }
    if stage_callback is not None:
        stage_callback("Computations complete. Saving JSON, CSV and PNG exports…")
    directory = _output_dir(run_name)
    (directory / "charts").mkdir()
    report = {
        "description": "Hypothetical GBM research simulation; not a forecast or financial advice.",
        "model": "Correlated GBM; monthly steps; annual input is expected simple return; month-end rebalancing",
        "assumptions": config.as_dict(),
        "selected_horizon": summary,
        "horizon_comparison": comparison,
    }
    with (directory / "simulation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    final_values = selected[:, -1]
    drawdowns = max_drawdowns(selected)
    with (directory / "final_portfolio_values.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("scenario", "final_value_usd", "cagr", "max_drawdown"))
        for index, (final, drawdown) in enumerate(zip(final_values, drawdowns), 1):
            writer.writerow((index, float(final), float((final / config.initial_capital) ** (1 / config.years) - 1), float(drawdown)))
    with (directory / "percentile_paths.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("month", "year", "p5_usd", "median_usd", "p95_usd"))
        for month, row in enumerate(percentile_paths(selected)):
            writer.writerow((month, month / 12, *map(float, row)))
    create_charts(paths, config.years, config.initial_capital, directory / "charts")
    return Experiment(config=config, paths=paths, report=report, directory=directory)
