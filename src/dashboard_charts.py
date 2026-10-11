"""Small Plotly payloads derived from all simulated scenarios."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from src.analytics import max_drawdowns, percentile_paths
from src.experiment import Experiment

CYAN = "#59D9CF"
BLUE = "#70A8FF"
GOLD = "#F2C572"
INK = "#E7F0F6"
GRID = "#27364D"


@dataclass(frozen=True)
class DashboardData:
    bands: np.ndarray
    sampled_paths: np.ndarray
    final_values: np.ndarray
    drawdowns: np.ndarray
    cagrs: np.ndarray
    comparison: dict
    initial_capital: float
    years: int
    scenarios: int


def prepare_dashboard_data(experiment: Experiment, max_paths: int = 100) -> DashboardData:
    """Keep all-scenario analytics, but retain at most 100 full paths for UI."""
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    config = experiment.config
    selected = experiment.paths[:, :config.years * 12 + 1]
    indexes = np.linspace(0, config.simulations - 1, min(max_paths, config.simulations), dtype=int)
    finals = selected[:, -1].copy()  # Copy: a view would retain the full simulation array.
    return DashboardData(
        bands=percentile_paths(selected),
        sampled_paths=selected[indexes].copy(),
        final_values=finals,
        drawdowns=max_drawdowns(selected),
        cagrs=np.power(finals / config.initial_capital, 1 / config.years) - 1,
        comparison=experiment.report["horizon_comparison"],
        initial_capital=config.initial_capital,
        years=config.years,
        scenarios=config.simulations,
    )


def _base(title: str, *, height: int = 405) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        margin=dict(l=78, r=16, t=50, b=50),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    return fig


def paths_figure(data: DashboardData, month: int) -> go.Figure:
    end = min(max(int(month), 0), data.years * 12)
    time = np.arange(end + 1) / 12
    sample = data.sampled_paths[:, :end + 1]
    # One trace with NaN separators is much lighter than 100 Plotly traces.
    xs = np.full((len(sample), end + 2), np.nan)
    ys = np.full_like(xs, np.nan)
    xs[:, :end + 1] = time
    ys[:, :end + 1] = sample
    fig = _base("Scenario paths · sampled for display", height=470)
    fig.add_trace(go.Scattergl(
        x=xs.ravel(), y=ys.ravel(), mode="lines", line=dict(color="rgba(112,168,255,0.23)", width=1),
        name=f"{len(sample)} sampled paths", hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=time, y=data.bands[:end + 1, 1], mode="lines",
        line=dict(color=CYAN, width=3), name="Median of all scenarios",
        hovertemplate="Year %{x:.2f}<br>Median $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=data.initial_capital, line_color=GOLD, line_dash="dot", opacity=.75)
    fig.update_xaxes(title="Years", range=[0, data.years])
    fig.update_yaxes(title="Portfolio value (USD)", tickprefix="$", tickformat=",.0f", rangemode="tozero")
    return fig


def band_figure(data: DashboardData) -> go.Figure:
    time = np.arange(len(data.bands)) / 12
    fig = _base("Median and P5–P95 scenario band")
    fig.add_trace(go.Scatter(x=time, y=data.bands[:, 2], mode="lines",
                             line=dict(color=BLUE, width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time, y=data.bands[:, 0], mode="lines", fill="tonexty",
                             fillcolor="rgba(112,168,255,0.20)", line=dict(color=BLUE, width=0),
                             name="P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time, y=data.bands[:, 1], mode="lines",
                             line=dict(color=CYAN, width=3), name="Median",
                             hovertemplate="Year %{x:.2f}<br>Median $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=data.initial_capital, line_color=GOLD, line_dash="dot", opacity=.75)
    fig.update_xaxes(title="Years")
    fig.update_yaxes(title="Portfolio value (USD)", tickprefix="$", tickformat=",.0f")
    return fig


def _histogram(values: np.ndarray, title: str, x_title: str, *, cap_p99: bool = False) -> go.Figure:
    fig = _base(title)
    upper = float(np.percentile(values, 99)) if cap_p99 else float(np.max(values))
    visible = values[values <= upper] if cap_p99 else values
    counts, edges = np.histogram(visible, bins=48)
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges), marker_color=BLUE,
                         hovertemplate="%{x:,.2f}<br>%{y:,} scenarios<extra></extra>", showlegend=False))
    if cap_p99:
        omitted = len(values) - len(visible)
        fig.add_annotation(text=f"Top {omitted:,} outcomes beyond axis · all retained in CSV",
                           xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                           xanchor="right", font=dict(size=11, color=GOLD))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Scenario count", rangemode="tozero")
    return fig


def final_values_figure(data: DashboardData) -> go.Figure:
    fig = _histogram(data.final_values, "Final portfolio value distribution", "Final value (USD)", cap_p99=True)
    fig.add_vline(x=float(np.median(data.final_values)), line_color=CYAN, line_width=2)
    fig.add_vline(x=data.initial_capital, line_color=GOLD, line_dash="dot")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


def drawdown_figure(data: DashboardData) -> go.Figure:
    fig = _histogram(data.drawdowns * 100, "Maximum drawdown distribution", "Largest monthly-observed peak-to-trough loss (%)")
    fig.update_xaxes(ticksuffix="%")
    return fig


def cagr_figure(data: DashboardData) -> go.Figure:
    fig = _histogram(data.cagrs * 100, "Annualized CAGR distribution", "Compound annual growth rate (%)")
    fig.update_xaxes(ticksuffix="%")
    return fig


def horizon_figure(data: DashboardData) -> go.Figure:
    years = (1, 5, 10)
    stats = [data.comparison[str(year)]["final_value_usd"] for year in years]
    median = np.array([row["median"] for row in stats])
    low = np.array([row["p5"] for row in stats])
    high = np.array([row["p95"] for row in stats])
    fig = _base("1Y / 5Y / 10Y final values")
    fig.add_trace(go.Scatter(
        x=[f"{year}Y" for year in years], y=median, mode="markers+lines",
        marker=dict(color=CYAN, size=11), line=dict(color=CYAN, width=2),
        error_y=dict(type="data", symmetric=False, array=high - median,
                     arrayminus=median - low, color=BLUE, thickness=4, width=9),
        name="Median · P5–P95",
        hovertemplate="%{x}<br>Median $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=data.initial_capital, line_color=GOLD, line_dash="dot", opacity=.75)
    fig.update_yaxes(title="Final value (USD)", tickprefix="$", tickformat=",.0f")
    return fig
