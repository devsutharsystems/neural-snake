"""PyTorch (original Net3 path, own one-hot) vs onnxruntime-python on web/model/snake.onnx.
 1. 1,000 Spike A VALIDATION rows (seed % 5 == 1; held-out rows are dropped by common.load_trainval): identical argmax grid,
    identical ate > 0.5 and game-over > 0.5 decisions.
 2. 20 closed-loop rollouts x 200 steps: same actions, same sampled food cell (sampled from the PyTorch food probabilities with a seeded RNG and written
    into BOTH grids, never checked for emptiness, exactly like web/main.js), states must stay identical; game over > 0.5 resets to the start state.
Run: .venv/bin/python scripts/parity_onnx.py -> web/model/parity_results.json
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "spikes", "spike_a"))
import common  # noqa: E402
import export_onnx  # noqa: E402

ONNX = os.path.join(ROOT, "web", "model", "snake.onnx")
START = os.path.join(ROOT, "web", "model", "start_state.json")
N = 12


def torch_ref(net, grid, act):
    """grid int [B,12,12], act int [B] -> (next_grid int32, food probs, ate prob, over prob) from the original forward (eval)."""
    x = torch.cat([F.one_hot(torch.as_tensor(grid).long(), 14).permute(0, 3, 1, 2).float(),
                   F.one_hot(torch.as_tensor(act).long(), 5).float().view(-1, 5, 1, 1).expand(-1, 5, N, N)], 1)
    with torch.no_grad():
        o = net(x)
    return (o["cls"].argmax(1).numpy().astype(np.int32), torch.softmax(o["food"], 1).numpy(), torch.sigmoid(o["ate"]).numpy(),
            torch.sigmoid(o["over"]).numpy())


def ort_run(sess, grid, act):
    g, f, a, o = sess.run(None, {"grid": grid.astype(np.int32), "action": act.astype(np.int32)})
    return g, f, a, o


def validation_parity(net, sess, n=1000):
    _, va = common.load_trainval()
    idx = np.sort(np.random.default_rng(0).choice(len(va["act"]), n, replace=False))
    d = {k: v[idx] for k, v in va.items()}
    x14, _ = common.make_14(d)
    bad_grid = bad_ate = bad_over = 0
    max_diff = {"food": 0.0, "ate": 0.0, "over": 0.0}
    for i in range(n):
        g, act = x14[i][None], d["act"][i:i + 1]
        rg, rf, ra, ro = torch_ref(net, g, act)
        og, of, oa, oo = ort_run(sess, g, act)
        bad_grid += int(not np.array_equal(rg, og))
        bad_ate += int((ra[0] > 0.5) != (oa[0] > 0.5))
        bad_over += int((ro[0] > 0.5) != (oo[0] > 0.5))
        max_diff["food"] = max(max_diff["food"], float(np.abs(rf - of).max()))
        max_diff["ate"] = max(max_diff["ate"], float(np.abs(ra - oa).max()))
        max_diff["over"] = max(max_diff["over"], float(np.abs(ro - oo).max()))
    return {"rows": n, "grid_mismatches": bad_grid, "ate_decision_mismatches": bad_ate, "over_decision_mismatches": bad_over,
            "max_abs_prob_diff": max_diff}


def pick_action(grid, rng):
    """Food-seeking action from the current grid (85%), random otherwise: the rollouts must contain eat steps. Same action goes to both runtimes."""
    head = np.argwhere((grid >= 2) & (grid <= 5))
    food = np.argwhere(grid == 1)
    if rng.random() < 0.15 or len(head) != 1 or len(food) != 1:
        return int(rng.integers(0, 5))
    (hr, hc), (fr, fc) = head[0], food[0]
    cur = int(grid[hr, hc]) - 2  # 0 U, 1 D, 2 L, 3 R
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    cands = [a for a in range(4) if a != opposite[cur]]
    dist = {a: abs(hr + delta[a][0] - fr) + abs(hc + delta[a][1] - fc) for a in cands}
    best = min(dist.values())
    return int(rng.choice([a for a in cands if dist[a] == best]))


def closed_loop(net, sess, n_roll=20, steps=200):
    start = np.array(json.load(open(START))["grid"], np.int32)
    res = {"rollouts": n_roll, "steps_each": steps, "grid_mismatches": 0, "state_divergences": 0, "ate_decision_mismatches": 0,
           "over_decision_mismatches": 0, "ate_events": 0, "game_over_resets": 0, "food_on_snake_writes": 0, "max_abs_prob_diff": 0.0}
    for r in range(n_roll):
        rng = np.random.default_rng(1000 + r)
        gt = start.copy()
        go = start.copy()
        for t in range(steps):
            act = np.array([pick_action(gt, rng)], np.int32)
            rg, rf, ra, ro = torch_ref(net, gt[None], act)
            og, of, oa, oo = ort_run(sess, go[None], act)
            res["grid_mismatches"] += int(not np.array_equal(rg, og))
            res["ate_decision_mismatches"] += int((ra[0] > 0.5) != (oa[0] > 0.5))
            res["over_decision_mismatches"] += int((ro[0] > 0.5) != (oo[0] > 0.5))
            res["max_abs_prob_diff"] = max(res["max_abs_prob_diff"], float(np.abs(rf - of).max()), float(np.abs(ra - oa).max()), float(np.abs(ro - oo).max()))
            gt, go = rg[0].copy(), og[0].copy()
            if ra[0] > 0.5:
                p = rf[0].astype(np.float64)
                c = int(rng.choice(N * N, p=p / p.sum()))
                res["ate_events"] += 1
                res["food_on_snake_writes"] += int(gt.flat[c] >= 2)
                gt.flat[c] = 1
                go.flat[c] = 1
            if ro[0] > 0.5:
                res["game_over_resets"] += 1
                gt, go = start.copy(), start.copy()
            res["state_divergences"] += int(not np.array_equal(gt, go))  # state after food write / reset, compared at every step
    return res


def main():
    net = export_onnx.load_net()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    out = {"command": "scripts/parity_onnx.py", "onnxruntime_py": ort.__version__, "onnx_file": "web/model/snake.onnx",
           "validation_1000": validation_parity(net, sess), "closed_loop": closed_loop(net, sess)}
    json.dump(out, open(os.path.join(ROOT, "web", "model", "parity_results.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
