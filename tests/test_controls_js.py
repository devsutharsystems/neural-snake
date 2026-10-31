"""Runs the Node test of web/controls.js (tests/js/controls.test.mjs): touch / swipe / button input only reports an action index 0..3."""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_controls_js_node_tests_pass():
    r = subprocess.run(["node", "--test", "tests/js/controls.test.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    assert int(re.search(r"ℹ pass (\d+)", r.stdout).group(1)) == 6
