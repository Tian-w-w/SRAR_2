"""Allow tests from a checkout without installing into the host Python."""
import sys
import uuid
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    # Keep test artifacts isolated and avoid walking unrelated system-temp
    # pytest directories left by earlier projects on this Windows workstation.
    if not config.option.basetemp:
        parent = Path(__file__).resolve().parents[1] / "runs/test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)
