import copy
import json
from pathlib import Path

import pytest

from src.config import validate_config


@pytest.fixture
def raw_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def config(raw_config):
    raw_config["simulations"] = 24
    return validate_config(raw_config)
