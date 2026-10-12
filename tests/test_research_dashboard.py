from pathlib import Path
import shutil
import json

from streamlit.testing.v1 import AppTest
from src.memecoin import MemecoinConfig


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard.py"


def test_memecoin_dashboard_mode_runs_and_exports():
    app = AppTest.from_file(DASHBOARD, default_timeout=90).run()
    app.selectbox(key="research_mode").select("Memecoin trading").run()
    assert not app.exception
    for field in app.number_input:
        if field.label == "Monte Carlo runs":
            field.set_value(100)
        if field.label == "Sequential trades":
            field.set_value(5)
    app.button[0].click().run()
    result = app.session_state["memecoin_result"]
    try:
        assert not app.exception
        assert result["sampled"].shape == (100, 6)
        assert result["report"]["source_label"].startswith("user-defined hypothetical")
        assert (result["directory"] / "simulation_summary.json").exists()
        assert (result["directory"] / "final_portfolio_values.csv").exists()
    finally:
        shutil.rmtree(result["directory"])


def test_spot_dashboard_mode_runs_and_exports():
    app = AppTest.from_file(DASHBOARD, default_timeout=90).run()
    app.selectbox(key="research_mode").select("Spot portfolio").run()
    assert not app.exception
    for field in app.number_input:
        if field.label == "Monte Carlo runs":
            field.set_value(100)
    app.button[0].click().run()
    result = app.session_state["spot_result"]
    try:
        assert not app.exception
        assert result["sampled"].shape == (100, 61)
        assert result["report"]["historical_estimates"] is None
        assert (result["directory"] / "percentile_paths.csv").exists()
    finally:
        shutil.rmtree(result["directory"])


def test_memecoin_json_configuration_runs():
    app = AppTest.from_file(DASHBOARD, default_timeout=90).run()
    app.selectbox(key="research_mode").select("Memecoin trading").run()
    app.radio(key="memecoin_source").set_value("JSON").run()
    raw = MemecoinConfig().as_dict()
    raw.update(trades=3, runs=10)
    app.text_area(key="memecoin_json_text").set_value(json.dumps(raw))
    app.button(key="memecoin_json_run").click().run()
    result = app.session_state["memecoin_result"]
    try:
        assert not app.exception
        assert result["runs"] == 10
        assert result["report"]["assumptions"]["trades"] == 3
    finally:
        shutil.rmtree(result["directory"])
