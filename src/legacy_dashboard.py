"""Local interactive research dashboard. Launch with python -m streamlit run dashboard.py."""

from io import BytesIO
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st

from src.config import ASSETS, load_config, validate_config
from src.dashboard_charts import (
    band_figure, cagr_figure, drawdown_figure, final_values_figure,
    horizon_figure, paths_figure, prepare_dashboard_data,
)
from src.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.json"
CHARTS = (
    "simulation_paths.png", "percentile_band.png", "final_values_histogram.png",
    "drawdown_distribution.png", "horizon_comparison.png",
)

def _inputs() -> tuple[dict, bool]:
    defaults = load_config(DEFAULT).as_dict()
    st.sidebar.markdown("### Experiment inputs")
    st.sidebar.caption("Hypothetical educational assumptions · no market data feed")
    with st.sidebar.form("assumptions_form"):
        initial = st.number_input("Initial capital (USD)", min_value=1.0,
                                  value=float(defaults["initial_capital"]), step=1000.0)
        st.markdown("**Target allocation**")
        weights = {
            asset: st.number_input(f"{asset} weight (%)", min_value=0.0, max_value=100.0,
                                   value=float(defaults["weights"][asset] * 100), step=1.0)
            for asset in ASSETS
        }
        st.caption(f"Total allocation: {sum(weights.values()):.1f}%")
        st.markdown("**Expected annual simple returns**")
        returns = {
            asset: st.number_input(f"{asset} expected return (%)", min_value=-99.0, max_value=500.0,
                                   value=float(defaults["annual_expected_returns"][asset] * 100), step=1.0)
            for asset in ASSETS
        }
        st.markdown("**Annual GBM volatility**")
        volatility = {
            asset: st.number_input(f"{asset} volatility (%)", min_value=0.0, max_value=500.0,
                                   value=float(defaults["annual_volatility"][asset] * 100), step=1.0)
            for asset in ASSETS
        }
        st.markdown("**Pairwise correlations**")
        btc_sp = st.number_input("BTC / S&P 500", min_value=-1.0, max_value=1.0,
                                 value=float(defaults["correlation"][0][1]), step=.05, format="%.2f")
        btc_gold = st.number_input("BTC / Gold", min_value=-1.0, max_value=1.0,
                                   value=float(defaults["correlation"][0][2]), step=.05, format="%.2f")
        sp_gold = st.number_input("S&P 500 / Gold", min_value=-1.0, max_value=1.0,
                                  value=float(defaults["correlation"][1][2]), step=.05, format="%.2f")
        st.markdown("**Run settings**")
        simulations = st.number_input("Simulation scenarios", min_value=100, max_value=100_000,
                                      value=int(defaults["simulations"]), step=100)
        years = st.selectbox("Selected horizon", (1, 5, 10), index=(1, 5, 10).index(defaults["years"]),
                             format_func=lambda value: f"{value} year{'s' if value > 1 else ''}")
        rebalancing = st.selectbox("Rebalancing", ("none", "monthly", "annual"),
                                   index=("none", "monthly", "annual").index(defaults["rebalancing"]))
        seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1,
                               value=int(defaults["seed"]), step=1)
        submitted = st.form_submit_button("Run Simulation", type="primary", width="stretch")
    data = {
        "assumptions_note": defaults["assumptions_note"],
        "initial_capital": initial,
        "weights": {asset: weights[asset] / 100 for asset in ASSETS},
        "annual_expected_returns": {asset: returns[asset] / 100 for asset in ASSETS},
        "annual_volatility": {asset: volatility[asset] / 100 for asset in ASSETS},
        "correlation": [[1.0, btc_sp, btc_gold], [btc_sp, 1.0, sp_gold], [btc_gold, sp_gold, 1.0]],
        "years": years, "simulations": simulations, "rebalancing": rebalancing, "seed": seed,
    }
    return data, submitted


def _start_playback() -> None:
    state = st.session_state
    if state.playback_month >= state.playback_max:
        state.playback_month = 0
    state.playing = True
    state.last_tick = time.monotonic()


def _pause_playback() -> None:
    st.session_state.playing = False


def _reset_playback() -> None:
    st.session_state.playing = False
    st.session_state.playback_month = 0


def _speed_changed() -> None:
    st.session_state.last_tick = time.monotonic()


def _playback(data) -> None:
    state = st.session_state
    run_every = .25 if state.playing else None

    @st.fragment(run_every=run_every)
    def view() -> None:
        state = st.session_state
        if state.playing != (run_every is not None):
            st.rerun()  # Recreate the fragment with the correct timer state.
        if state.playing:
            now = time.monotonic()
            advanced = int((now - state.last_tick) * state.playback_speed)
            if advanced > 0:
                state.playback_month = min(state.playback_max, state.playback_month + advanced)
                state.last_tick += advanced / state.playback_speed
            if state.playback_month >= state.playback_max:
                state.playing = False
                st.rerun()
        play, pause, reset = st.columns(3)
        play.button("▶ Play", key="play_paths", on_click=_start_playback,
                    disabled=state.playing, width="stretch")
        pause.button("Ⅱ Pause", key="pause_paths", on_click=_pause_playback,
                     disabled=not state.playing, width="stretch")
        reset.button("↺ Reset", key="reset_paths", on_click=_reset_playback, width="stretch")
        st.select_slider("Playback speed · months per second", options=(1, 2, 4, 8),
                         key="playback_speed", on_change=_speed_changed)
        chosen_month = st.slider("Timeline · completed month", min_value=0, max_value=state.playback_max,
                                 value=state.playback_month, key=f"timeline_at_{state.playback_month}")
        if chosen_month != state.playback_month:
            state.playing = False
            state.playback_month = chosen_month
            st.rerun()
        st.caption(f"Playback of completed results · month {state.playback_month} / {state.playback_max}. "
                   f"Showing {len(data.sampled_paths):,} sampled paths; statistics use all {data.scenarios:,} scenarios.")
        st.plotly_chart(paths_figure(data, state.playback_month), width="stretch", theme=None,
                        config={"displaylogo": False, "scrollZoom": False},
                        alt="Sampled portfolio scenario paths appearing month by month with an all-scenario median")

    view()


def _metric_cards(report: dict) -> None:
    stats = report["selected_horizon"]
    final = stats["final_value_usd"]
    risk = stats["terminal_loss_risk_usd"]
    cards = (
        ("Median final", f"${final['median']:,.0f}", "selected horizon"),
        ("Mean final", f"${final['mean']:,.0f}", "selected horizon"),
        ("Median CAGR", f"{stats['cagr']['median']:.1%}", "compound annual growth"),
        ("Median max drawdown", f"{stats['max_drawdown']['median']:.1%}", "monthly-observed peak loss"),
        ("Finish below initial", f"{stats['probability_below_initial']:.1%}", "simulated share"),
        ("Terminal-loss VaR 95%", f"${risk['var_95']:,.0f}", "USD, selected horizon"),
        ("Expected shortfall 95%", f"${risk['expected_shortfall_95']:,.0f}", "USD, selected horizon"),
    )
    markup = "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'
        for label, value, note in cards
    )
    st.markdown(f'<div class="metric-grid">{markup}</div>', unsafe_allow_html=True)


def _export_section(directory: Path) -> None:
    st.subheader("Export this experiment")
    st.caption("The files use the same JSON, CSV and PNG structure as the command-line simulator.")
    names = ["simulation_summary.json", "final_portfolio_values.csv", "percentile_paths.csv"]
    names.extend(f"charts/{chart}" for chart in CHARTS)
    selected = st.selectbox("Choose one file", names)
    mime = "application/json" if selected.endswith(".json") else (
        "text/csv" if selected.endswith(".csv") else "image/png"
    )
    st.download_button("Download selected file", data=(directory / selected).read_bytes(),
                       file_name=Path(selected).name, mime=mime, on_click="ignore", width="stretch")
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(directory / name, arcname=name)
    st.download_button("Download full experiment ZIP", data=archive.getvalue(),
                       file_name=f"{directory.name}-portfolio-monte-carlo.zip", mime="application/zip",
                       on_click="ignore", width="stretch")
    st.caption(f"Also saved locally: {directory}")




def render_legacy() -> None:
    st.markdown('<div class="lab-eyebrow">LOCAL RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="lab-title">Portfolio Monte Carlo Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="lab-subtitle">Explore how portfolio weights, return assumptions, volatility and correlation change modeled outcomes across BTC, the S&amp;P 500 and Gold.</div>', unsafe_allow_html=True)
    st.caption("Educational GBM scenarios · no live data, predictions or financial advice")

    raw_inputs, run_clicked = _inputs()
    if run_clicked:
        try:
            config = validate_config(raw_inputs)
        except ValueError as exc:
            st.error(f"Please correct the inputs: {exc}")
        else:
            status = st.empty()
            status.info("Computing 120 monthly steps for the shared 1Y / 5Y / 10Y comparison…")
            bar = st.progress(0.0, text="Computation: 0 / 120 months completed")

            def update_progress(done: int, total: int) -> None:
                bar.progress(done / total, text=f"Computation: {done} / {total} months completed")

            try:
                experiment = run_experiment(config, progress_callback=update_progress,
                                            stage_callback=status.info)
                display_data = prepare_dashboard_data(experiment)
            except (ValueError, FileExistsError, OSError) as exc:
                bar.empty()
                status.error(f"Simulation failed: {exc}")
            else:
                st.session_state.result = {
                    "report": experiment.report,
                    "data": display_data,
                    "directory": experiment.directory,
                }
                st.session_state.playback_month = 0
                st.session_state.playback_max = config.years * 12
                st.session_state.playback_speed = 2
                st.session_state.playing = False
                st.session_state.last_tick = time.monotonic()
                bar.empty()
                status.success(f"Calculation and exports complete · {config.simulations:,} scenarios · seed {config.seed}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Choose assumptions in the sidebar and select **Run Simulation**. The initial view performs no calculation.")
        st.markdown("**Research conventions:** annual input returns are expected simple returns; GBM volatility is the annual log-return shock parameter. Correlations must form a valid matrix. Simulated percentile bands are not forecast confidence intervals.")
    else:
        report = result["report"]
        data = result["data"]
        directory = result["directory"]
        config = report["assumptions"]
        st.markdown(f"### Completed experiment · {config['years']}Y · {config['simulations']:,} scenarios")
        st.caption(f"Seed {config['seed']} · {config['rebalancing']} rebalancing · annual return and volatility inputs are hypothetical")
        _metric_cards(report)
        st.caption("Terminal loss = initial capital − final value over the selected horizon; negative values are gains. VaR 95% is the 95th loss percentile; expected shortfall is the average loss at or above it. Simulated probabilities are not real-world forecasts.")

        st.subheader("Scenario playback")
        st.caption("Animation advances only through already computed monthly results. It is separate from the computation progress shown during a run.")
        _playback(data)

        outcomes, risk, horizons = st.tabs(("Outcome bands & final values", "Drawdown & CAGR", "Horizon comparison"))
        with outcomes:
            st.plotly_chart(band_figure(data), width="stretch", theme=None, config={"displaylogo": False},
                            alt="Median simulated portfolio value and fifth to ninety-fifth percentile band")
            st.plotly_chart(final_values_figure(data), width="stretch", theme=None, config={"displaylogo": False},
                            alt="Histogram of final portfolio values, with upper one percent noted separately")
        with risk:
            st.plotly_chart(drawdown_figure(data), width="stretch", theme=None, config={"displaylogo": False},
                            alt="Distribution of scenario maximum drawdowns")
            st.plotly_chart(cagr_figure(data), width="stretch", theme=None, config={"displaylogo": False},
                            alt="Distribution of scenario compound annual growth rates")
        with horizons:
            st.plotly_chart(horizon_figure(data), width="stretch", theme=None, config={"displaylogo": False},
                            alt="One, five and ten year final value medians and scenario percentile ranges")
            st.caption("The comparison uses the same 10-year paths, sliced at months 12, 60 and 120.")
        _export_section(directory)
