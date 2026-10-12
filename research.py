"""CLI for the added memecoin and arbitrary-asset spot research modes."""

import argparse
import json
from pathlib import Path

from src.memecoin import parse_trade_csv, validate_memecoin_config
from src.research_experiment import run_memecoin_experiment, run_spot_experiment
from src.spot import parse_price_csv, validate_spot_config


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local memecoin or spot portfolio research")
    parser.add_argument("mode", choices=("memecoin", "spot"))
    parser.add_argument("--config", type=Path, help="JSON configuration; defaults to configs/<mode>.json")
    parser.add_argument("--trades-csv", type=Path, help="realized trade CSV for empirical memecoin bootstrap")
    parser.add_argument("--prices-csv", type=Path, help="aligned daily USD price CSV for spot block bootstrap")
    parser.add_argument("--run-name", help="unique experiment name; existing output is never overwritten")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / f"{args.mode}.json"
    try:
        with config_path.open(encoding="utf-8") as source:
            raw = json.load(source)
        if args.mode == "memecoin":
            if args.prices_csv:
                raise ValueError("--prices-csv only applies to spot mode")
            config = validate_memecoin_config(raw)
            trades = parse_trade_csv(args.trades_csv.read_bytes()) if args.trades_csv else ()
            _, report, directory = run_memecoin_experiment(config, trades, args.run_name)
            median = report["final_bankroll_usd"]["median"]
        else:
            if args.trades_csv:
                raise ValueError("--trades-csv only applies to memecoin mode")
            config = validate_spot_config(raw)
            history = parse_price_csv(args.prices_csv.read_bytes(), tuple(config.data["assets"])) if args.prices_csv else None
            _, report, directory = run_spot_experiment(config, history, args.run_name)
            median = report["selected_horizon"]["final_value_usd"]["median"]
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"{args.mode} | median final value ${median:,.2f} | saved {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
