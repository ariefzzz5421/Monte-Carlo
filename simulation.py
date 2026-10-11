"""Command-line entry point for reproducible portfolio experiments."""

import argparse
from pathlib import Path
import sys

from src.config import load_config, validate_config
from src.experiment import run_experiment

ROOT = Path(__file__).resolve().parent


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
    experiment = run_experiment(config, run_name=args.run_name)
    summary = experiment.report["selected_horizon"]
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
    print(f"Saved: {experiment.directory}")
    return experiment.directory


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
