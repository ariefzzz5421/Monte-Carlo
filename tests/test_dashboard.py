from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from streamlit.testing.v1 import AppTest

from src.dashboard_charts import (
    band_figure, cagr_figure, drawdown_figure, final_values_figure,
    horizon_figure, paths_figure, prepare_dashboard_data,
)
from src.experiment import run_experiment


def _remove_test_output(directory: Path) -> None:
    if directory.name == "results":
        for name in ("simulation_summary.json", "final_portfolio_values.csv", "percentile_paths.csv"):
            (directory / name).unlink(missing_ok=True)
        shutil.rmtree(directory / "charts", ignore_errors=True)
    else:
        shutil.rmtree(directory, ignore_errors=True)


def test_dashboard_data_uses_all_scenarios_and_caps_display_paths(config):
    events = []
    experiment = run_experiment(
        config,
        run_name=f"pytest-{uuid4().hex}",
        progress_callback=lambda done, total: events.append((done, total)),
        stage_callback=lambda stage: events.append(stage),
    )
    try:
        assert events[:2] == [(1, 120), (2, 120)]
        assert events[-2] == (120, 120)
        assert "Saving JSON" in events[-1]
        data = prepare_dashboard_data(experiment, max_paths=10)
        assert data.sampled_paths.shape == (10, config.years * 12 + 1)
        assert len(data.final_values) == config.simulations
        assert len(data.drawdowns) == config.simulations
        assert len(data.cagrs) == config.simulations
        assert not np.shares_memory(data.final_values, experiment.paths)
        assert len(paths_figure(data, 0).data) == 2
        assert len(paths_figure(data, 12).data[0].x) > len(paths_figure(data, 0).data[0].x)
        for figure in (band_figure(data), final_values_figure(data), drawdown_figure(data),
                       cagr_figure(data), horizon_figure(data)):
            assert figure.data
    finally:
        _remove_test_output(experiment.directory)


def test_streamlit_run_and_playback_controls():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py", default_timeout=90).run()
    assert not app.exception
    assert "result" not in app.session_state
    scenario_input = next(field for field in app.number_input if field.label == "Simulation scenarios")
    scenario_input.set_value(100)
    next(button for button in app.button if button.label == "Run Simulation").click().run()
    directory = app.session_state.result["directory"]
    try:
        assert not app.exception
        assert app.success
        assert len(app.session_state.result["data"].sampled_paths) == 100
        assert (directory / "simulation_summary.json").exists()
        assert len(list((directory / "charts").glob("*.png"))) == 5
        next(button for button in app.button if "Play" in button.label).click().run()
        assert app.session_state.playing is True
        next(button for button in app.button if "Pause" in button.label).click().run()
        assert app.session_state.playing is False
        app.slider[-1].set_value(30).run()
        assert app.session_state.playback_month == 30
        assert app.session_state.playing is False
        next(button for button in app.button if "Reset" in button.label).click().run()
        assert app.session_state.playback_month == 0
        assert not app.exception
    finally:
        _remove_test_output(directory)


def test_streamlit_rejects_invalid_weights():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py", default_timeout=20).run()
    next(field for field in app.number_input if field.label == "BTC weight (%)").set_value(31.0)
    next(button for button in app.button if button.label == "Run Simulation").click().run()
    assert app.error
    assert "result" not in app.session_state
