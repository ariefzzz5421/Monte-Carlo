import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4


def test_cli_smoke():
    root = Path(__file__).resolve().parents[1]
    name = f"pytest-{uuid4().hex}"
    target = root / "results" / "runs" / name
    try:
        run = subprocess.run(
            [sys.executable, "simulation.py", "--years", "1", "--simulations", "12", "--run-name", name],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        assert run.returncode == 0, run.stderr
        assert "Saved:" in run.stdout
        assert json.loads((target / "simulation_summary.json").read_text())["assumptions"]["seed"] == 42
        assert len((target / "final_portfolio_values.csv").read_text().splitlines()) == 13
        assert len(list((target / "charts").glob("*.png"))) == 5
        repeated = subprocess.run(
            [sys.executable, "simulation.py", "--simulations", "1", "--run-name", name],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        assert repeated.returncode == 2
        assert "exists" in repeated.stderr
    finally:
        shutil.rmtree(target, ignore_errors=True)
