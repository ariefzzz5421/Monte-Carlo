from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest

from src.memecoin import MemecoinConfig
from src.spot import DEFAULT_SPOT


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ("memecoin", "spot"))
def test_new_mode_cli_smoke(tmp_path, mode):
    raw = MemecoinConfig().as_dict() if mode == "memecoin" else deepcopy(DEFAULT_SPOT)
    raw["runs"] = 10
    if mode == "memecoin":
        raw["trades"] = 3
    else:
        raw["years"] = 1
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    name = f"pytest-cli-{uuid4().hex}"
    directory = ROOT / "results" / "runs" / f"{mode}-{name}"
    try:
        process = subprocess.run(
            (sys.executable, str(ROOT / "research.py"), mode, "--config", str(config),
             "--run-name", name), cwd=ROOT, capture_output=True, text=True,
        )
        assert process.returncode == 0, process.stderr
        assert "median final value" in process.stdout
        assert (directory / "simulation_summary.json").exists()
        assert (directory / "final_portfolio_values.csv").exists()
        assert (directory / "percentile_paths.csv").exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("mode", ("memecoin", "spot"))
def test_bootstrap_cli_uses_imported_synthetic_fixture(tmp_path, mode):
    """Synthetic inputs exercise file plumbing; they are not packaged market data."""
    raw = MemecoinConfig().as_dict() if mode == "memecoin" else deepcopy(DEFAULT_SPOT)
    raw["runs"] = 10
    if mode == "memecoin":
        raw.update(model="empirical_bootstrap", trades=3)
        csv_text = ("timestamp,entry_cost,realized_net_proceeds,fees\n"
                    "2025-01-01,100,0,2\n2025-01-02,100,200,2\n")
        flag = "--trades-csv"
    else:
        from datetime import date, timedelta
        raw.update(method="block_bootstrap", years=1)
        rows = ["date,BTC,HYPE,USDT,XAUT"]
        for index in range(40):
            rows.append(f"{date(2025, 1, 1) + timedelta(days=index)},{100+index},{10+index},1,{50+index}")
        csv_text = "\n".join(rows) + "\n"
        flag = "--prices-csv"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    csv_path = tmp_path / "synthetic.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    name = f"pytest-bootstrap-{uuid4().hex}"
    directory = ROOT / "results" / "runs" / f"{mode}-{name}"
    try:
        completed = subprocess.run(
            (sys.executable, str(ROOT / "research.py"), mode, "--config", str(config),
             flag, str(csv_path), "--run-name", name), cwd=ROOT, capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads((directory / "simulation_summary.json").read_text())
        if mode == "memecoin":
            assert report["historical_sample_size"] == 2
            assert report["source_label"].startswith("imported realized trades")
        else:
            assert report["historical_estimates"]["observations"] == 40
            assert report["method"] == "block_bootstrap"
    finally:
        shutil.rmtree(directory, ignore_errors=True)
