"""Runs the Node golden-trace test (tests/js/engine_golden.test.mjs): web/engine.js must reproduce the 50 Python traces in tests/golden/traces.json exactly."""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_engine_matches_all_50_golden_traces():
    r = subprocess.run(["node", "--test", "tests/js/engine_golden.test.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    passed = int(re.search(r"# pass (\d+)|ℹ pass (\d+)", r.stdout).group(1) or re.search(r"ℹ pass (\d+)", r.stdout).group(1))
    assert passed == 51  # 50 traces + the file-has-50-traces check
