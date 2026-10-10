"""Headless, consistently styled Matplotlib experiment charts."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

from src.analytics import max_drawdowns, percentile_paths

NAVY = "#172b4d"
BLUE = "#2478b8"
GOLD = "#d69b22"


def _style() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 5.3),
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "font.size": 10,
    })


def _usd(value: float, _position: float) -> str:
    return f"${value:,.0f}"


def _save(fig, path) -> None:
    fig.tight_layout(pad=1.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def create_charts(paths: np.ndarray, selected_years: int, initial: float, directory) -> None:
    _style()
    chosen = paths[:, :selected_years * 12 + 1]
    time = np.arange(chosen.shape[1]) / 12
    bands = percentile_paths(chosen)

    fig, ax = plt.subplots()
    # A fixed subset avoids adding RNG draws to the experiment.
    for path in chosen[:min(100, len(chosen))]:
        ax.plot(time, path, color=BLUE, alpha=0.10, linewidth=0.7)
    ax.plot(time, bands[:, 1], color=NAVY, linewidth=2.5, label="Median")
    ax.axhline(initial, color=GOLD, linestyle="--", label="Initial capital")
    ax.set(title=f"Portfolio scenarios · {selected_years} years (up to 100 shown)",
           xlabel="Years", ylabel="Portfolio value (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax.legend()
    _save(fig, directory / "simulation_paths.png")

    fig, ax = plt.subplots()
    ax.fill_between(time, bands[:, 0], bands[:, 2], color=BLUE, alpha=0.22,
                    label="P5–P95 of simulated scenarios")
    ax.plot(time, bands[:, 1], color=NAVY, linewidth=2.5, label="Median")
    ax.axhline(initial, color=GOLD, linestyle="--", label="Initial capital")
    ax.set(title=f"Simulated outcome band · {selected_years} years",
           xlabel="Years", ylabel="Portfolio value (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax.legend()
    _save(fig, directory / "percentile_band.png")

    fig, ax = plt.subplots()
    finals = chosen[:, -1]
    display_max = max(float(np.percentile(finals, 99)), float(np.median(finals)) * 1.01)
    visible = finals[finals <= display_max]
    ax.hist(visible, bins=50, color=BLUE, alpha=0.82, edgecolor="white")
    ax.axvline(np.median(finals), color=NAVY, linewidth=2, label="Median")
    ax.axvline(initial, color=GOLD, linestyle="--", linewidth=2, label="Initial capital")
    ax.set(title=f"Final portfolio values · {selected_years} years\nTop {len(finals) - len(visible):,} outcomes beyond axis; all in CSV",
           xlabel="Final value (USD)", ylabel="Scenario count")
    ax.xaxis.set_major_formatter(FuncFormatter(_usd))
    ax.legend()
    _save(fig, directory / "final_values_histogram.png")

    fig, ax = plt.subplots()
    ax.hist(max_drawdowns(chosen), bins=35, color=BLUE, alpha=0.82, edgecolor="white")
    ax.set(title=f"Maximum drawdown across scenarios · {selected_years} years",
           xlabel="Largest peak-to-trough loss", ylabel="Scenario count")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, directory / "drawdown_distribution.png")

    fig, ax = plt.subplots()
    horizons = (1, 5, 10)
    end_values = [paths[:, years * 12] for years in horizons]
    low = np.array([np.percentile(values, 5) for values in end_values])
    mid = np.array([np.median(values) for values in end_values])
    high = np.array([np.percentile(values, 95) for values in end_values])
    x = np.arange(len(horizons))
    ax.vlines(x, low, high, color=BLUE, linewidth=10, alpha=0.45,
              label="P5–P95 of simulated outcomes")
    ax.scatter(x, mid, color=NAVY, s=70, zorder=3, label="Median")
    ax.axhline(initial, color=GOLD, linestyle="--", label="Initial capital")
    ax.set_xticks(x, [f"{years}Y" for years in horizons])
    ax.set(title="Comparison of simulated final outcomes", xlabel="Horizon",
           ylabel="Final portfolio value (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax.legend()
    _save(fig, directory / "horizon_comparison.png")
