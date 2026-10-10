"""Command-line entry point for reproducible portfolio experiments."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from src.analytics import max_drawdowns, percentile_paths, summarize
from src.config import load_config, validate_config
from src.engine import simulate
from src.visualization import create_charts

ROOT = Path(__file__).resolve().parent


def _output_dir(run_name: str | None) -> Path:
    results = ROOT / "results"
    if run_name:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
            raise ValueError("run name must be 1–64 letters, digits, underscores or hyphens")
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


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    data = config.as_dict()
    if args.years is not None:
        data["years"] = args.years
    if args.simulations is not None:
        data["simulations"] = args.simulations
    if args.seed is not None:
        data["seed"] = args.seed
    if args.rebalancing is not None:
        data["rebalancing"] = args.rebalancing
    config = validate_config(data)
    if args.run_name and (ROOT / "results" / "runs" / args.run_name).exists():
        raise FileExistsError(f"run already exists: {args.run_name}")
    # Simulate through year 10 once, then slice the identical scenarios at 1/5/10Y.
    paths = simulate(config, max_years=10)
    selected = paths[:, :config.years * 12 + 1]
    summary = summarize(selected, config.initial_capital, config.years)
    comparison = {
        str(year): summarize(paths[:, :year * 12 + 1], config.initial_capital, year)
        for year in (1, 5, 10)
    }
    directory = _output_dir(args.run_name)
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
    print(f"Portfolio Monte Carlo Lab | {config.simulations:,} scenarios | {config.years} years | seed {config.seed}")
    print(f"Assumptions: {config.assumptions_note}")
    print(f"Rebalancing: {config.rebalancing}")
    values = summary["final_value_usd"]
    print(f"Final value: mean ${values['mean']:,.2f}; median ${values['median']:,.2f}")
    print(f"P5-P95: ${values['p5']:,.2f} to ${values['p95']:,.2f}")
    print(f"Below initial: {summary['probability_below_initial']:.1%}; at least 2x: {summary['probability_at_least_double']:.1%}")
    print(f"Median CAGR: {summary['cagr']['median']:.1%}; median max drawdown: {summary['max_drawdown']['median']:.1%}")
    risk = summary["terminal_loss_risk_usd"]
    print(f"Terminal-loss VaR 95%: ${risk['var_95']:,.2f}; ES 95%: ${risk['expected_shortfall_95']:,.2f}")
    print(f"Saved: {directory}")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Correlated GBM portfolio research simulator")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json", help="JSON configuration file")
    parser.add_argument("--years", type=int, choices=(1, 5, 10), help="selected report horizon")
    parser.add_argument("--simulations", type=int, help="number of scenarios")
    parser.add_argument("--seed", type=int, help="reproducible random seed")
    parser.add_argument("--rebalancing", choices=("none", "monthly", "annual"), help="rebalance frequency")
    parser.add_argument("--run-name", help="save to results/runs/NAME; never overwrite a prior named run")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
