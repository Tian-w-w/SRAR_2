"""Reproducible CPU verification with a saved result; does not run a Backbone."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["PYTHONPATH"] = str(ROOT / "src")
started = time.perf_counter()
result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q"], cwd=ROOT, env=env, capture_output=True, text=True)
(ROOT / "reports/pytest.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
record = {"command": "python -B -m pytest -q", "exit_code": result.returncode,
          "elapsed_seconds": time.perf_counter() - started, "python": sys.version,
          "real_backbone_used": False, "real_experiment_completed": False}
(ROOT / "reports/cpu_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
for args in (["--help"], ["audit"], ["generate-counterfactuals", "--dry-run"], ["smoke", "--dry-run"]):
    subprocess.run([sys.executable, "-B", "-m", "srar2", *args], cwd=ROOT, env=env, check=True, capture_output=True)
print("CPU tests, CLI help, audit and dry-run: PASS. No real Qwen inference performed.")
