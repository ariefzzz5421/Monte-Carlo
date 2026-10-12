"""Sequential bankroll research with realized trade multiples.

A multiple is net proceeds after fees divided by entry cost. Token ATH is not
an input: a price reaching 100x does not mean a trader realized a 100x exit.
"""

from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import math

import numpy as np

DEFAULT_OUTCOMES = (
    ("Total loss", 0.0, 0.35),
    ("Partial loss", 0.5, 0.35),
    ("2x realized", 2.0, 0.22),
    ("10x realized", 10.0, 0.06),
    ("50x realized", 50.0, 0.015),
    ("100x realized", 100.0, 0.005),
)


@dataclass(frozen=True)
class Trade:
    timestamp: datetime
    entry_cost: float
    realized_net_proceeds: float
    fees: float

    @property
    def realized_multiple(self) -> float:
        return self.realized_net_proceeds / self.entry_cost


@dataclass(frozen=True)
class MemecoinConfig:
    bankroll: float = 10_000.0
    position_size: float = 100.0
    trades: int = 100
    runs: int = 10_000
    reinvest: bool = False
    ruin_fraction: float = 0.1
    seed: int = 42
    model: str = "hypothetical"
    outcomes: tuple[tuple[str, float, float], ...] = DEFAULT_OUTCOMES

    def as_dict(self) -> dict:
        return {
            "bankroll": self.bankroll, "position_size": self.position_size,
            "trades": self.trades, "runs": self.runs, "reinvest": self.reinvest,
            "ruin_fraction": self.ruin_fraction, "seed": self.seed,
            "model": self.model,
            "outcomes": [dict(label=name, realized_multiple=multiple, probability=probability)
                         for name, multiple, probability in self.outcomes],
        }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def validate_memecoin_config(data: dict) -> MemecoinConfig:
    if not isinstance(data, dict):
        raise ValueError("memecoin configuration must be a JSON object")
    expected = set(MemecoinConfig().as_dict())
    if set(data) != expected:
        raise ValueError(f"memecoin configuration keys: missing={sorted(expected-set(data))}, unknown={sorted(set(data)-expected)}")
    bankroll = _finite(data["bankroll"], "bankroll")
    position = _finite(data["position_size"], "position_size")
    ruin = _finite(data["ruin_fraction"], "ruin_fraction")
    if bankroll <= 0 or position <= 0 or position > bankroll:
        raise ValueError("bankroll and position_size must be positive; position_size cannot exceed bankroll")
    if not 0 <= ruin < 1:
        raise ValueError("ruin_fraction must be between 0 (inclusive) and 1 (exclusive)")
    trades = _integer(data["trades"], "trades", 1, 10_000)
    runs = _integer(data["runs"], "runs", 1, 100_000)
    seed = _integer(data["seed"], "seed", 0, 2**32 - 1)
    if not isinstance(data["reinvest"], bool):
        raise ValueError("reinvest must be true or false")
    model = data["model"]
    if model not in ("hypothetical", "empirical_bootstrap"):
        raise ValueError("model must be hypothetical or empirical_bootstrap")
    raw = data["outcomes"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("outcomes must be a nonempty list")
    outcomes = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) != {"label", "realized_multiple", "probability"}:
            raise ValueError(f"outcomes[{index}] needs label, realized_multiple and probability")
        label = row["label"]
        multiple = _finite(row["realized_multiple"], f"outcomes[{index}].realized_multiple")
        probability = _finite(row["probability"], f"outcomes[{index}].probability")
        if not isinstance(label, str) or not label.strip() or multiple < 0 or not 0 <= probability <= 1:
            raise ValueError(f"outcomes[{index}] has an invalid label, multiple or probability")
        outcomes.append((label.strip(), multiple, probability))
    if not math.isclose(sum(row[2] for row in outcomes), 1.0, abs_tol=1e-8):
        raise ValueError("outcome probabilities must sum to 1")
    return MemecoinConfig(bankroll, position, trades, runs, data["reinvest"], ruin,
                          seed, model, tuple(outcomes))


def parse_trade_csv(content: str | bytes) -> tuple[Trade, ...]:
    """Read realized trades; required columns are timestamp, entry_cost,
    realized_net_proceeds and fees. Net proceeds already subtract the listed
    fees; fees are retained for audit and are never subtracted twice.
    """
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("trade CSV must be UTF-8") from exc
    reader = csv.DictReader(StringIO(content))
    required = {"timestamp", "entry_cost", "realized_net_proceeds", "fees"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("trade CSV needs timestamp, entry_cost, realized_net_proceeds, fees")
    trades = []
    for line, row in enumerate(reader, 2):
        try:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"trade CSV line {line}: invalid ISO timestamp") from exc
        entry = _finite(row["entry_cost"], f"line {line} entry_cost")
        proceeds = _finite(row["realized_net_proceeds"], f"line {line} realized_net_proceeds")
        fees = _finite(row["fees"], f"line {line} fees")
        if entry <= 0 or proceeds < 0 or fees < 0:
            raise ValueError(f"trade CSV line {line}: entry must be positive; net proceeds and fees nonnegative")
        trades.append(Trade(timestamp, entry, proceeds, fees))
    if not trades:
        raise ValueError("trade CSV must contain at least one realized trade")
    if any(a.timestamp.tzinfo is None for a in trades) != all(a.timestamp.tzinfo is None for a in trades):
        raise ValueError("trade CSV timestamps must use one consistent timezone convention")
    if any(a.timestamp > b.timestamp for a, b in zip(trades, trades[1:])):
        raise ValueError("trade CSV timestamps must be sorted chronologically")
    return tuple(trades)


def simulate_memecoin(
    config: MemecoinConfig,
    historical_trades: tuple[Trade, ...] = (),
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Return bankroll paths after each trade, including trade zero.

    Reinvest=true scales position size with the current bankroll using the
    initial position/bankroll fraction; otherwise position size stays fixed.
    Both are capped by available capital. Each trade settles before the next.
    """
    if config.model == "empirical_bootstrap":
        if not historical_trades:
            raise ValueError("empirical bootstrap requires an imported trade CSV")
        multiples = np.asarray([trade.realized_multiple for trade in historical_trades])
        probabilities = None
    else:
        multiples = np.asarray([row[1] for row in config.outcomes])
        probabilities = np.asarray([row[2] for row in config.outcomes])
    rng = np.random.default_rng(config.seed)
    paths = np.empty((config.runs, config.trades + 1), dtype=float)
    paths[:, 0] = config.bankroll
    bankroll = np.full(config.runs, config.bankroll, dtype=float)
    fraction = config.position_size / config.bankroll
    for trade_number in range(1, config.trades + 1):
        sampled = rng.choice(multiples, size=config.runs, p=probabilities)
        requested = bankroll * fraction if config.reinvest else config.position_size
        stake = np.minimum(bankroll, requested)
        bankroll = bankroll + stake * (sampled - 1)
        paths[:, trade_number] = bankroll
        if progress_callback is not None:
            progress_callback(trade_number, config.trades)
    if not np.all(np.isfinite(paths)) or np.any(paths < -1e-8):
        raise ValueError("bankroll overflowed; reduce outcome multiples or trade count")
    return np.maximum(paths, 0)


def summarize_memecoin(paths: np.ndarray, config: MemecoinConfig) -> dict:
    final = paths[:, -1]
    peaks = np.maximum.accumulate(paths, axis=1)
    relative_loss = np.divide(peaks - paths, peaks, out=np.zeros_like(paths), where=peaks > 0)
    drawdown = relative_loss.max(axis=1)
    losses = config.bankroll - final
    var = float(np.percentile(losses, 95))
    return {
        "mode": "memecoin_trading", "model": config.model,
        "assumptions": config.as_dict(),
        "historical_sample_size": None,  # Caller fills this for empirical bootstrap.
        "risk_conventions": {
            "realized_multiple": "realized net proceeds after fees / entry cost; theoretical ATH is excluded",
            "ruin": f"bankroll ever <= {config.ruin_fraction:.1%} of starting bankroll",
            "terminal_loss": "starting bankroll - final bankroll; negative denotes a gain",
            "var_es": "95th percentile terminal USD loss and mean loss at or above that percentile",
        },
        "final_bankroll_usd": {
            "mean": float(np.mean(final)), "median": float(np.median(final)),
            "p5": float(np.percentile(final, 5)), "p95": float(np.percentile(final, 95)),
            "p99": float(np.percentile(final, 99)),
        },
        "probability_of_loss": float(np.mean(final < config.bankroll)),
        "risk_of_ruin": float(np.mean(np.any(paths <= config.bankroll * config.ruin_fraction, axis=1))),
        "max_drawdown": {"median": float(np.median(drawdown)), "p95": float(np.percentile(drawdown, 95))},
        "terminal_loss_risk_usd": {
            "var_95": var, "expected_shortfall_95": float(np.mean(losses[losses >= var])),
        },
    }
