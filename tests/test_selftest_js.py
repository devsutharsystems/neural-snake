"""Runs the Node test of the runtime self-test (tests/js/selftest.test.mjs): real model on the vendored WASM backend, plus the fallback logic."""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_selftest_js_node_tests_pass():
    r = subprocess.run(["node", "--test", "tests/js/selftest.test.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    assert int(re.search(r"ℹ pass (\d+)", r.stdout).group(1)) == 13
