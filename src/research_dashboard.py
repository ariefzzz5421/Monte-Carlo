"""Streamlit controls and visual results for the two additional local modes."""

from copy import deepcopy
from io import BytesIO
import json
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import percentile_paths
from src.memecoin import MemecoinConfig, parse_trade_csv, validate_memecoin_config
from src.research_experiment import run_memecoin_experiment, run_spot_experiment
from src.spot import DEFAULT_SPOT, parse_price_csv, validate_spot_config


CYAN, BLUE, GOLD = "#59D9CF", "#70A8FF", "#F2C572"


def _memecoin_controls() -> tuple[dict | None, bool, bytes | None]:
    default = MemecoinConfig().as_dict()
    if st.sidebar.radio("Configuration format", ("Controls", "JSON"), key="memecoin_source", horizontal=True) == "JSON":
        uploaded = st.sidebar.file_uploader("Load configuration JSON", type="json", key="memecoin_json_file")
        text = st.sidebar.text_area("Configuration JSON", value=json.dumps(default, indent=2),
                                    height=310, key="memecoin_json_text")
        csv_file = st.sidebar.file_uploader("Realized trades CSV", type="csv", key="memecoin_json_csv")
        run = st.sidebar.button("Run simulation", type="primary", key="memecoin_json_run", width="stretch")
        if not run:
            return None, False, None
        try:
            return json.loads(uploaded.getvalue() if uploaded else text), True, csv_file.getvalue() if csv_file else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid configuration JSON: {exc}") from exc
    with st.expander("Configure memecoin experiment", expanded="memecoin_result" not in st.session_state), st.form("memecoin_form"):
        bankroll = st.number_input("Starting bankroll (USD)", min_value=1.0, value=10_000.0, step=100.0)
        position = st.number_input("Initial position size (USD)", min_value=0.01, value=100.0, step=10.0)
        trades = st.number_input("Sequential trades", min_value=1, max_value=10_000, value=100)
        runs = st.number_input("Monte Carlo runs", min_value=100, max_value=100_000, value=10_000, step=100)
        model = st.selectbox("Outcome model", ("hypothetical", "empirical_bootstrap"),
                             format_func=lambda item: "User-defined hypothetical" if item == "hypothetical" else "Imported trades · empirical bootstrap")
        csv_file = st.file_uploader("Realized trades CSV · needed for bootstrap", type="csv")
        reinvest = st.checkbox("Reinvest: position scales with current bankroll", value=False)
        ruin = st.number_input("Ruin threshold (% of starting bankroll)", min_value=0.0,
                               max_value=99.0, value=10.0, step=1.0)
        seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
        st.markdown("**Hypothetical realized outcomes**")
        st.caption("These are net exit multiples, never theoretical token ATHs. Probabilities must sum to 100%.")
        outcomes = []
        for item in default["outcomes"]:
            label = item["label"]
            with st.expander(label):
                multiple = st.number_input(f"{label} · realized multiple", min_value=0.0,
                                           value=float(item["realized_multiple"]), step=0.5)
                probability = st.number_input(f"{label} · probability (%)", min_value=0.0,
                                              max_value=100.0, value=item["probability"] * 100, step=0.5)
            outcomes.append(dict(label=label, realized_multiple=multiple, probability=probability / 100))
        submitted = st.form_submit_button("Run simulation", type="primary", width="stretch")
    return dict(bankroll=bankroll, position_size=position, trades=int(trades), runs=int(runs),
                reinvest=reinvest, ruin_fraction=ruin / 100, seed=int(seed), model=model,
                outcomes=outcomes), submitted, csv_file.getvalue() if csv_file else None


def _spot_controls() -> tuple[dict | None, bool, bytes | None]:
    default = deepcopy(DEFAULT_SPOT)
    if st.sidebar.radio("Configuration format", ("Controls", "JSON"), key="spot_source", horizontal=True) == "JSON":
        uploaded = st.sidebar.file_uploader("Load configuration JSON", type="json", key="spot_json_file")
        text = st.sidebar.text_area("Configuration JSON", value=json.dumps(default, indent=2),
                                    height=310, key="spot_json_text")
        csv_file = st.sidebar.file_uploader("Aligned daily USD price CSV", type="csv", key="spot_json_csv")
        run = st.sidebar.button("Run simulation", type="primary", key="spot_json_run", width="stretch")
        if not run:
            return None, False, None
        try:
            return json.loads(uploaded.getvalue() if uploaded else text), True, csv_file.getvalue() if csv_file else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid configuration JSON: {exc}") from exc
    with st.expander("Configure spot portfolio", expanded="spot_result" not in st.session_state), st.form("spot_form"):
        capital = st.number_input("Initial capital (USD)", min_value=1.0, value=10_000.0, step=1000.0)
        st.markdown("**Assets and hypothetical future inputs**")
        rows = pd.DataFrame([
            {"Asset": asset, "Weight %": default["weights"][asset] * 100,
             "Future return %": default["future_expected_returns"][asset] * 100,
             "Future volatility %": default["future_volatility"][asset] * 100}
            for asset in default["assets"]
        ])
        edited = st.data_editor(rows, num_rows="dynamic", hide_index=True, width="stretch",
                                key="spot_asset_editor")
        st.caption("Add or remove asset rows. Correlation matrix must match their displayed order. Swipe the table sideways on a phone to edit return and volatility.")
        corr_text = st.text_area("Future correlation matrix · JSON", value=json.dumps(default["future_correlation"]),
                                 height=125)
        method = st.selectbox("Simulation method", ("gbm", "block_bootstrap"),
                              format_func=lambda value: "Correlated GBM · future assumptions" if value == "gbm" else "Historical daily block bootstrap")
        csv_file = st.file_uploader("Aligned daily USD prices CSV · required for bootstrap", type="csv")
        block_days = st.number_input("Bootstrap block length (days)", min_value=1, max_value=365, value=7)
        years = st.selectbox("Years", (1, 5, 10), index=1)
        runs = st.number_input("Monte Carlo runs", min_value=100, max_value=100_000, value=10_000, step=100)
        rebalance = st.selectbox("Rebalancing", ("none", "monthly", "annual"), index=1)
        seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
        st.markdown("**Optional one-time stress shocks**")
        depeg_on = st.checkbox("Stablecoin depeg")
        depeg_asset = st.text_input("Depeg asset", value="USDT")
        depeg_month = st.number_input("Depeg month", min_value=1, max_value=120, value=12)
        depeg_loss = st.number_input("Depeg loss (%)", min_value=0.0, max_value=100.0, value=20.0)
        crash_on = st.checkbox("Crypto crash")
        crash_assets = st.text_input("Crash assets · comma separated", value="BTC, HYPE")
        crash_month = st.number_input("Crash month", min_value=1, max_value=120, value=12)
        crash_loss = st.number_input("Crash loss (%)", min_value=0.0, max_value=100.0, value=50.0)
        submitted = st.form_submit_button("Run simulation", type="primary", width="stretch")
    if not submitted:
        return None, False, None
    try:
        corr = json.loads(corr_text)
        assets = [str(value).strip() for value in edited["Asset"].tolist()]
        weights = dict(zip(assets, (edited["Weight %"] / 100).tolist()))
        expected = dict(zip(assets, (edited["Future return %"] / 100).tolist()))
        volatility = dict(zip(assets, (edited["Future volatility %"] / 100).tolist()))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"asset table or correlation matrix is invalid: {exc}") from exc
    data = dict(initial_capital=capital, assets=assets, weights=weights, years=years,
                runs=int(runs), seed=int(seed), rebalancing=rebalance, method=method,
                block_days=int(block_days), future_assumptions_note=default["future_assumptions_note"],
                future_expected_returns=expected, future_volatility=volatility, future_correlation=corr,
                stablecoin_depeg=dict(enabled=depeg_on, asset=depeg_asset.strip(), month=int(depeg_month),
                                      multiplier=1 - depeg_loss / 100),
                crypto_crash=dict(enabled=crash_on, assets=[x.strip() for x in crash_assets.split(",") if x.strip()],
                                  month=int(crash_month), multiplier=1 - crash_loss / 100))
    return data, True, csv_file.getvalue() if csv_file else None


def _base(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=390, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#E7F0F6",
                      margin=dict(l=75, r=15, t=55, b=55), hovermode="x unified")
    fig.update_xaxes(gridcolor="#27364D", automargin=True)
    fig.update_yaxes(gridcolor="#27364D", automargin=True)
    return fig


def _path_figure(result: dict, step: int) -> go.Figure:
    paths, bands = result["sampled"], result["bands"]
    end = min(step, paths.shape[1] - 1)
    x = np.arange(end + 1)
    sample = paths[:, :end + 1]
    xs = np.full((len(sample), end + 2), np.nan)
    ys = np.full_like(xs, np.nan)
    xs[:, :end + 1], ys[:, :end + 1] = x, sample
    fig = _base("Sampled bankroll paths" if result["mode"] == "memecoin" else "Sampled portfolio paths")
    fig.add_trace(go.Scattergl(x=xs.ravel(), y=ys.ravel(), mode="lines", name="Sampled scenarios",
                               line=dict(color="rgba(112,168,255,0.22)", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[:end + 1, 1], mode="lines", name="All-scenario median",
                             line=dict(color=CYAN, width=3)))
    fig.update_xaxes(title="Completed trade" if result["mode"] == "memecoin" else "Completed month",
                     range=[0, paths.shape[1] - 1])
    fig.update_yaxes(title="USD", tickprefix="$")
    return fig


def _playback(result: dict) -> None:
    mode = result["mode"]
    state = st.session_state
    key = f"{mode}_playback"
    if key not in state:
        state[key] = dict(step=0, playing=False, speed=2, tick=time.monotonic())
    run_every = .25 if state[key]["playing"] else None

    @st.fragment(run_every=run_every)
    def view() -> None:
        controls = st.session_state[key]
        if controls["playing"] != (run_every is not None):
            st.rerun()
        maximum = result["sampled"].shape[1] - 1
        if controls["playing"]:
            now = time.monotonic()
            advance = int((now - controls["tick"]) * controls["speed"])
            if advance:
                controls["step"] = min(maximum, controls["step"] + advance)
                controls["tick"] += advance / controls["speed"]
            if controls["step"] == maximum:
                controls["playing"] = False
                st.rerun()
        a, b, c = st.columns(3)
        if a.button("▶ Play", key=f"{mode}_play", disabled=controls["playing"], width="stretch"):
            controls.update(step=0 if controls["step"] == maximum else controls["step"],
                            playing=True, tick=time.monotonic())
            st.rerun()
        if b.button("Ⅱ Pause", key=f"{mode}_pause", disabled=not controls["playing"], width="stretch"):
            controls["playing"] = False
            st.rerun()
        if c.button("↺ Reset", key=f"{mode}_reset", width="stretch"):
            controls.update(step=0, playing=False)
            st.rerun()
        speed = st.select_slider("Playback speed · steps per second", options=(1, 2, 4, 8),
                                 value=controls["speed"], key=f"{mode}_speed")
        if speed != controls["speed"]:
            controls.update(speed=speed, tick=time.monotonic())
        chosen = st.slider("Timeline · completed step", min_value=0, max_value=maximum,
                           value=controls["step"], key=f"{mode}_timeline_{controls['step']}")
        if chosen != controls["step"]:
            controls.update(step=chosen, playing=False)
            st.rerun()
        st.caption(f"Playback of completed results · step {controls['step']} / {maximum}. "
                   f"Showing {len(result['sampled'])} paths; metrics use all {result['runs']:,} runs.")
        st.plotly_chart(_path_figure(result, controls["step"]), width="stretch", theme=None,
                        alt="Sampled scenario paths revealed progressively after computation")

    view()


def _band_chart(result: dict) -> go.Figure:
    bands = result["bands"]
    x = np.arange(len(bands))
    fig = _base("Median and P5–P95 simulation band")
    fig.add_trace(go.Scatter(x=x, y=bands[:, 2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[:, 0], fill="tonexty", fillcolor="rgba(112,168,255,.20)",
                             line=dict(width=0), name="P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[:, 1], name="Median", line=dict(color=CYAN, width=3)))
    fig.update_xaxes(title="Trade" if result["mode"] == "memecoin" else "Month")
    fig.update_yaxes(title="USD", tickprefix="$")
    return fig


def _hist(values: np.ndarray, title: str, x_title: str) -> go.Figure:
    fig = _base(title)
    upper = np.percentile(values, 99)
    visible = values[values <= upper]
    counts, edges = np.histogram(visible, bins=45)
    fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
                         marker_color=BLUE, showlegend=False))
    fig.add_annotation(text=f"Top {len(values)-len(visible):,} outcomes beyond axis · retained in CSV",
                       xref="paper", yref="paper", x=1, y=1.12, xanchor="right", showarrow=False,
                       font=dict(color=GOLD, size=10))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Scenarios")
    return fig


def _exports(result: dict) -> None:
    directory = result["directory"]
    names = ("simulation_summary.json", "final_portfolio_values.csv", "percentile_paths.csv")
    st.subheader("Export this experiment")
    st.caption(f"Saved locally: {directory}")
    selected = st.selectbox("Choose an output file", names, key=f"{result['mode']}_download_name")
    st.download_button("Download selected file", (directory / selected).read_bytes(), file_name=selected,
                       mime="application/json" if selected.endswith(".json") else "text/csv",
                       on_click="ignore", width="stretch", key=f"{result['mode']}_download")
    config_bytes = json.dumps(result["report"]["assumptions"], indent=2).encode("utf-8")
    st.download_button("Download reusable configuration JSON", config_bytes,
                       file_name=f"{result['mode']}-config.json", mime="application/json",
                       on_click="ignore", width="stretch", key=f"{result['mode']}_config_download")
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(directory / name, arcname=name)
    st.download_button("Download all results ZIP", archive.getvalue(),
                       file_name=f"{directory.name}.zip", mime="application/zip", on_click="ignore",
                       width="stretch", key=f"{result['mode']}_zip")


def render_research(mode: str) -> None:
    is_trade = mode == "memecoin"
    st.markdown("### Memecoin trading simulator" if is_trade else "### Spot portfolio simulator")
    st.caption("All probabilities and prices come from your inputs. No live data or invented history is used.")
    try:
        raw, submitted, csv_bytes = _memecoin_controls() if is_trade else _spot_controls()
    except ValueError as exc:
        st.error(str(exc))
        return
    if submitted:
        try:
            if is_trade:
                config = validate_memecoin_config(raw)
                history = parse_trade_csv(csv_bytes) if csv_bytes else ()
                if config.model == "empirical_bootstrap" and not history:
                    raise ValueError("upload realized trades CSV to use empirical bootstrap")
                total = config.trades
            else:
                config = validate_spot_config(raw)
                history = parse_price_csv(csv_bytes, tuple(config.data["assets"])) if csv_bytes else None
                if config.data["method"] == "block_bootstrap" and history is None:
                    raise ValueError("upload aligned daily USD price CSV to use block bootstrap")
                total = config.data["years"] * 12
        except ValueError as exc:
            st.error(f"Please correct the inputs: {exc}")
        else:
            status = st.empty()
            status.info(f"Computing {total} {'trades' if is_trade else 'months'} across all scenarios…")
            bar = st.progress(0.0, text=f"Computation: 0 / {total} completed")

            def progress(done: int, count: int) -> None:
                if done == count or done % max(1, count // 100) == 0:
                    bar.progress(done / count, text=f"Computation: {done} / {count} completed")

            try:
                if is_trade:
                    paths, report, directory = run_memecoin_experiment(config, history, progress_callback=progress)
                else:
                    paths, report, directory = run_spot_experiment(config, history, progress_callback=progress)
                bands = percentile_paths(paths)
                indexes = np.linspace(0, len(paths) - 1, min(100, len(paths)), dtype=int)
                result = dict(mode=mode, sampled=paths[indexes].copy(), bands=bands,
                              finals=paths[:, -1].copy(),
                              drawdowns=np.divide(np.maximum.accumulate(paths, axis=1) - paths,
                                                  np.maximum.accumulate(paths, axis=1),
                                                  out=np.zeros_like(paths),
                                                  where=np.maximum.accumulate(paths, axis=1) > 0).max(axis=1),
                              runs=len(paths), report=report, directory=directory)
            except (ValueError, OSError, FileExistsError) as exc:
                bar.empty()
                status.error(f"Simulation failed: {exc}")
            else:
                st.session_state[f"{mode}_result"] = result
                st.session_state[f"{mode}_playback"] = dict(step=0, playing=False, speed=2,
                                                             tick=time.monotonic())
                bar.empty()
                status.success("Calculation and exports complete")
    result = st.session_state.get(f"{mode}_result")
    if result is None:
        st.info("Choose inputs above (or switch to JSON in the sidebar) and select Run simulation. No calculation runs until you click it.")
        if is_trade:
            st.markdown("A realized 10x means the **actual net exit proceeds** were ten times entry cost. Token all-time highs do not prove a trader sold there. Hypothetical probabilities are examples, not observed trading statistics.")
        else:
            st.markdown("The default BTC / HYPE / USDT / XAUT allocation is a strategy example. GBM uses explicitly hypothetical future assumptions; block bootstrap requires your own aligned daily USD prices. Imported historical estimates never silently replace future inputs.")
        return
    report = result["report"]
    st.caption(report["source_label"])
    if is_trade:
        final = report["final_bankroll_usd"]
        with st.container(horizontal=True):
            st.metric("Median final bankroll", f"${final['median']:,.0f}", border=True)
            st.metric("Mean final bankroll", f"${final['mean']:,.0f}", border=True)
            st.metric("Probability of loss", f"{report['probability_of_loss']:.1%}", border=True)
            st.metric("Risk of ruin", f"{report['risk_of_ruin']:.1%}", border=True)
            st.metric("Median max drawdown", f"{report['max_drawdown']['median']:.1%}", border=True)
        st.caption(report["risk_conventions"]["ruin"] + ". Theoretical token ATH is excluded.")
        if report["historical_sample_size"] is not None:
            st.info(f"Empirical bootstrap sampled {report['historical_sample_size']} realized trades with replacement. A small or biased sample cannot establish true future probabilities.")
    else:
        stats = report["selected_horizon"]
        with st.container(horizontal=True):
            st.metric("Median final value", f"${stats['final_value_usd']['median']:,.0f}", border=True)
            st.metric("Mean final value", f"${stats['final_value_usd']['mean']:,.0f}", border=True)
            st.metric("Probability of loss", f"{stats['probability_below_initial']:.1%}", border=True)
            st.metric("Median max drawdown", f"{stats['max_drawdown']['median']:.1%}", border=True)
            st.metric("Expected shortfall 95%", f"${stats['terminal_loss_risk_usd']['expected_shortfall_95']:,.0f}", border=True)
        if report["historical_estimates"]:
            with st.expander("Historical estimates from imported prices"):
                st.json(report["historical_estimates"])
        st.caption("Terminal USD loss = initial capital − final value. VaR 95% and expected shortfall use the selected horizon and all runs.")
    st.subheader("Scenario playback")
    st.caption("Animation uses completed results; the computation progress bar only appears while the model is running.")
    _playback(result)
    outcomes, risk = st.tabs(("Outcome paths and final values", "Drawdown and tail risk"))
    with outcomes:
        st.plotly_chart(_band_chart(result), width="stretch", theme=None,
                        alt="Median path and fifth to ninety-fifth percentile band")
        st.plotly_chart(_hist(result["finals"], "Final bankroll distribution" if is_trade else "Final portfolio distribution",
                              "Final USD value"), width="stretch", theme=None,
                        alt="Distribution of final values across all simulations")
    with risk:
        st.plotly_chart(_hist(result["drawdowns"] * 100, "Maximum drawdown distribution", "Drawdown (%)"),
                        width="stretch", theme=None, alt="Distribution of maximum drawdowns")
        losses = result["report"]["terminal_loss_risk_usd"] if is_trade else report["selected_horizon"]["terminal_loss_risk_usd"]
        st.metric("Terminal-loss VaR 95%", f"${losses['var_95']:,.0f}", border=True)
        st.metric("Expected shortfall 95%", f"${losses['expected_shortfall_95']:,.0f}", border=True)
    _exports(result)
