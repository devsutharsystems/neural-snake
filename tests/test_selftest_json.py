"""web/model/selftest.json must belong to web/model/snake.onnx and be reproducible from it."""
import hashlib
import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
ST = json.load(open(os.path.join(ROOT, "web", "model", "selftest.json")))
MODEL = os.path.join(ROOT, "web", "model", "snake.onnx")


def test_selftest_json_matches_the_shipped_model_file():
    assert ST["model_sha256"] == hashlib.sha256(open(MODEL, "rb").read()).hexdigest(), "model re-exported? run scripts/make_selftest.py"


def test_cases_cover_the_required_situations():
    cases = ST["cases"]
    assert len(cases) == 8
    start = [v for row in json.load(open(os.path.join(ROOT, "web", "model", "start_state.json")))["grid"] for v in row]
    assert any(c["grid"] == start for c in cases)
    assert any(c["ate"] for c in cases) and any(c["over"] for c in cases)
    assert any(sum(1 for v in c["grid"] if v >= 2) >= 20 for c in cases), "no long snake"
    for c in cases:
        assert len(c["grid"]) == len(c["next_grid"]) == len(c["food"]) == 144 and c["action"] in range(5)
        assert c["ate"] == (c["ate_prob"] > 0.5) and c["over"] == (c["over_prob"] > 0.5)


def test_recorded_outputs_reproduce_with_onnxruntime():
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    for c in ST["cases"]:
        g, f, a, o = sess.run(None, {"grid": np.array(c["grid"], np.int32).reshape(1, 12, 12), "action": np.array([c["action"]], np.int32)})
        assert g.reshape(-1).tolist() == c["next_grid"], c["name"]
        assert abs(float(a[0]) - c["ate_prob"]) < 1e-5 and abs(float(o[0]) - c["over_prob"]) < 1e-5
        assert float(np.abs(f[0] - np.array(c["food"])).max()) < 1e-5
