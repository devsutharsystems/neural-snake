"""Compare the vendored onnxruntime-web WASM backend (run in Node, scripts/ort_wasm_node.mjs) with onnxruntime-python on the start state, 5 actions.
This is NOT a browser result and says nothing about WebGPU. Run: .venv/bin/python scripts/check_wasm_node.py -> web/model/wasm_node_check.json"""
import json
import os
import subprocess

import numpy as np
import onnxruntime as ort

ROOT = os.path.join(os.path.dirname(__file__), "..")
out = json.loads(subprocess.run(["node", os.path.join(ROOT, "scripts", "ort_wasm_node.mjs")], capture_output=True, text=True, check=True, cwd=ROOT).stdout)
sess = ort.InferenceSession(os.path.join(ROOT, "web", "model", "snake.onnx"), providers=["CPUExecutionProvider"])
grid = np.array(json.load(open(os.path.join(ROOT, "web", "model", "start_state.json")))["grid"], np.int32)[None]
rows = []
for o in out:
    g, f, a, ov = sess.run(None, {"grid": grid, "action": np.array([o["action"]], np.int32)})
    rows.append({"action": o["action"], "grid_identical": bool(np.array_equal(g.flatten(), np.array(o["next_grid"]))),
                 "max_abs_food_diff": float(np.abs(f.flatten() - np.array(o["food"])).max()), "ate_diff": float(abs(a[0] - o["ate"])),
                 "over_diff": float(abs(ov[0] - o["over"])), "wasm_output_types": o["types"]})
res = {"command": "scripts/check_wasm_node.py (node scripts/ort_wasm_node.mjs, vendored ort.wasm.min.mjs 1.31.1, executionProviders=['wasm'])", "node_not_browser": True, "rows": rows}
json.dump(res, open(os.path.join(ROOT, "web", "model", "wasm_node_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
