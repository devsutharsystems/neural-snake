"""PyTorch vs onnxruntime parity for web/model/snake.onnx. Skips if the git-ignored checkpoint/data or the ONNX file is missing.
Validation rows only (seed % 5 == 1); held-out rows are never loaded."""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
needed = [os.path.join(ROOT, p) for p in ("checkpoints/run2_best.pt", "web/model/snake.onnx", "web/model/start_state.json", "data/spike_a/act.npy")]
pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in needed), reason="checkpoint/onnx/data missing (git-ignored or not built)")


@pytest.fixture(scope="module")
def setup():
    import onnxruntime as ort
    import export_onnx
    import parity_onnx
    return parity_onnx, export_onnx.load_net(), ort.InferenceSession(parity_onnx.ONNX, providers=["CPUExecutionProvider"])


def test_validation_rows_identical_grid_and_decisions(setup):
    p, net, sess = setup
    r = p.validation_parity(net, sess, 1000)
    assert r["rows"] == 1000
    assert r["grid_mismatches"] == 0 and r["ate_decision_mismatches"] == 0 and r["over_decision_mismatches"] == 0
    assert max(r["max_abs_prob_diff"].values()) < 1e-3


def test_closed_loop_rollouts_identical_states(setup):
    p, net, sess = setup
    r = p.closed_loop(net, sess, 20, 200)
    assert r["rollouts"] == 20 and r["steps_each"] == 200
    assert r["grid_mismatches"] == 0 and r["state_divergences"] == 0
    assert r["ate_decision_mismatches"] == 0 and r["over_decision_mismatches"] == 0
