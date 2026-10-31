"""Write web/model/selftest.json: 8 fixed golden cases for the in-page runtime self-test.

Each case is (14-class grid, action) -> the model's OWN recorded outputs, computed here with web/model/snake.onnx in onnxruntime-python:
the argmax next grid, the ate / game-over probabilities and decisions (threshold 0.5) and the 144 food probabilities.
The page re-runs the cases on every backend it creates and compares: a backend whose outputs differ is discarded (web/selftest.js).
This compares the model with its own recorded outputs; it contains no game rules. The grids come from the Python engine only to get realistic inputs.
The file records the sha256 of snake.onnx; tests fail if the model is re-exported without regenerating this file.
Run: .venv/bin/python scripts/make_selftest.py
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spikes", "spike_a"))
from engine.snake import Snake, DELTA, UP, DOWN, LEFT, RIGHT, NOOP, N  # noqa: E402
import common  # noqa: E402
import gen  # noqa: E402

ONNX = os.path.join(ROOT, "web", "model", "snake.onnx")
OUT = os.path.join(ROOT, "web", "model", "selftest.json")
PROB_TOLERANCE = 0.05  # a backend fails if any recorded probability differs by more than this (genuine fp32 differences are ~1e-6)


def candidates():
    """Yield (category, grid (list of 144 ints), action) for engine states in a fixed order; categories name what the step is."""
    for seed in range(4000, 4400):
        for pol in ("bfs", "greedy", "bfs_eps20", "greedy_eps30", "random"):
            s = Snake(seed=seed)
            rng = random.Random(seed + 7)
            for _ in range(300):
                if s.game_over:
                    break
                a = gen.pick(pol, s, rng)
                grid = common.grid14(s.body, s.direction, s.food).reshape(-1).tolist()
                length, before_dir = len(s.body), s.direction
                hr, hc = s.body[0]
                d = s.direction if (a == NOOP or (length > 1 and a == {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}[s.direction])) else a
                nr, nc = hr + DELTA[d][0], hc + DELTA[d][1]
                wall = not (0 <= nr < N and 0 <= nc < N)
                eaten = s.step(a)
                if s.game_over and wall:
                    yield "game-over-wall", grid, int(a)
                elif s.game_over:
                    yield "game-over-self-collision", grid, int(a)
                elif eaten and length >= 20:
                    yield "eat-step-long-snake", grid, int(a)
                elif eaten and 3 <= length <= 10:
                    yield "eat-step-short-snake", grid, int(a)
                elif length == 8 and a != NOOP and a != before_dir:
                    yield "turn-mid-length-snake", grid, int(a)
                elif length >= 30 and a in (NOOP, before_dir):
                    yield "long-snake-straight", grid, int(a)


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

    def run(grid, action):
        g, f, a, o = sess.run(None, {"grid": np.array(grid, np.int32).reshape(1, N, N), "action": np.array([action], np.int32)})
        return g.reshape(-1).astype(int).tolist(), f[0].astype(float).tolist(), float(a[0]), float(o[0])

    start = json.load(open(os.path.join(ROOT, "web", "model", "start_state.json")))["grid"]
    start = [v for row in start for v in row]
    want = ["turn-mid-length-snake", "long-snake-straight", "eat-step-short-snake", "eat-step-long-snake", "game-over-wall", "game-over-self-collision"]
    found = {}
    for cat, grid, action in candidates():
        if cat in found or cat not in want:
            continue
        nxt, food, ate, over = run(grid, action)
        # keep a candidate only if the model's recorded outputs show what the case is meant to exercise
        if cat.startswith("eat") and not ate > 0.5:
            continue
        if cat.startswith("game-over") and not over > 0.5:
            continue
        if not cat.startswith(("eat", "game-over")) and (ate > 0.5 or over > 0.5):
            continue
        found[cat] = (grid, action)
        if len(found) == len(want):
            break
    missing = [c for c in want if c not in found]
    assert not missing, f"no candidate found for {missing}"
    plan = [("start-state-noop", start, NOOP), ("start-state-down", start, DOWN)] + [(c, *found[c]) for c in want]
    cases = []
    for name, grid, action in plan:
        nxt, food, ate, over = run(grid, action)
        cases.append({"name": name, "action": int(action), "grid": grid, "next_grid": nxt, "ate_prob": round(ate, 6), "over_prob": round(over, 6),
                      "ate": bool(ate > 0.5), "over": bool(over > 0.5), "food": [round(x, 6) for x in food]})
    out = {"description": "Golden cases for the runtime self-test: the model's own recorded outputs (onnxruntime-python, CPU) for fixed inputs. No game rules.",
           "generated_by": "scripts/make_selftest.py", "model_file": "snake.onnx", "model_sha256": hashlib.sha256(open(ONNX, "rb").read()).hexdigest(),
           "prob_tolerance": PROB_TOLERANCE,
           "pass_rule": "next_grid identical in all 144 cells, ate and game-over decisions (probability > 0.5) identical, and every probability (ate, over, 144 food) within prob_tolerance",
           "cases": cases}
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"{len(cases)} cases, {os.path.getsize(OUT)} bytes, model sha256 {out['model_sha256']}")
    for c in cases:
        print(f"  {c['name']:28s} action={c['action']} snake cells={sum(1 for v in c['grid'] if v >= 2):3d} ate={c['ate']!s:5} over={c['over']!s:5}")


if __name__ == "__main__":
    main()
