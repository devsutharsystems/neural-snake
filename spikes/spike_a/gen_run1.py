"""Run 1 data: new episodes with new seeds (policy_index*100000 + 50000 + k; the Phase 0 / Spike A data used k in 0..3999, so no overlap).
Same split rule: seed % 5 == 0 held-out (written, flagged, never loaded), == 1 validation, otherwise train.
Each policy runs until it has TARGET_TRAIN train transitions (equal share per policy, same mix as before). Writes uint8 .npy to data/run1/.

Run: .venv/bin/python spikes/spike_a/gen_run1.py
"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from engine.snake import Snake  # noqa: E402
from gen import POLICIES, CAP, pick, order_grid, cls_grid  # noqa: E402

OUT = os.path.join(ROOT, "data", "run1")
TARGET_TRAIN = 80_000  # per policy -> ~400k train transitions in total
SEED_BASE = 50_000
KEYS = ["cls", "order", "dir", "act", "prev_cls", "n_cls", "n_order", "n_dir", "eaten", "food", "over", "ep", "pol", "held", "length"]


def gen_policy(pi):
    p = POLICIES[pi]
    rows = {k: [] for k in KEYS}
    n_train, k = 0, 0
    while n_train < TARGET_TRAIN:
        sd = pi * 100000 + SEED_BASE + k
        k += 1
        s = Snake(seed=sd)
        rng = random.Random(sd + 7)
        prev = cls_grid(s)
        steps = 0
        while not s.game_over and steps < CAP:
            cur_cls, cur_ord, cur_dir = cls_grid(s), order_grid(s), s.direction
            a = pick(p, s, rng)
            eaten = s.step(a)
            det = cls_grid(s)
            if eaten and s.food is not None:
                det[s.food[0], s.food[1]] = 0
            rows["cls"].append(cur_cls); rows["order"].append(cur_ord); rows["dir"].append(cur_dir)
            rows["act"].append(a); rows["prev_cls"].append(prev)
            rows["n_cls"].append(det); rows["n_order"].append(order_grid(s)); rows["n_dir"].append(s.direction)
            rows["eaten"].append(int(eaten))
            rows["food"].append(s.food[0] * 12 + s.food[1] if (eaten and s.food is not None) else 255)
            rows["over"].append(int(s.game_over))
            rows["ep"].append(sd); rows["pol"].append(pi); rows["held"].append(int(sd % 5 == 0))
            rows["length"].append(len(s.body) - (1 if eaten else 0))
            prev = cur_cls
            steps += 1
            if sd % 5 not in (0, 1):
                n_train += 1
    return {k_: np.array(v, dtype=np.int32 if k_ == "ep" else np.uint8) for k_, v in rows.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(len(POLICIES)) as pool:
        parts = pool.map(gen_policy, range(len(POLICIES)))
    n = sum(len(p["act"]) for p in parts)
    for k in KEYS:
        shape = (n,) + parts[0][k].shape[1:]
        mm = np.lib.format.open_memmap(os.path.join(OUT, k + ".npy"), mode="w+", dtype=parts[0][k].dtype, shape=shape)
        i = 0
        for p in parts:
            mm[i:i + len(p[k])] = p[k]
            i += len(p[k])
        mm.flush()
        del mm
    ep, held = np.load(os.path.join(OUT, "ep.npy")), np.load(os.path.join(OUT, "held.npy")).astype(bool)
    length = np.load(os.path.join(OUT, "length.npy"))
    tr, va = ~held & (ep % 5 != 1), ~held & (ep % 5 == 1)
    old_ep = np.load(os.path.join(ROOT, "data", "spike_a", "ep.npy"))
    old_pol = np.load(os.path.join(ROOT, "data", "spike_a", "pol.npy"))
    pol = np.load(os.path.join(OUT, "pol.npy"))
    overlap = len(set(zip(old_pol.tolist(), old_ep.tolist())) & set(zip(pol.tolist(), ep.tolist())))
    bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 999)]
    hist = {f"{a}-{b}": int(((length[tr] >= a) & (length[tr] <= b)).sum()) for a, b in bins}
    disk = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    stats = {"command": "gen_run1.py", "seconds": round(time.time() - t0, 1), "total_transitions": int(n), "train": int(tr.sum()),
             "validation": int(va.sum()), "train_by_policy": {POLICIES[i]: int((tr & (pol == i)).sum()) for i in range(len(POLICIES))},
             "train_length_hist(length before the step)": hist, "episode_seed_overlap_with_spike_a": overlap,
             "disk_MB": round(disk / 1e6, 1)}
    json.dump(stats, open(os.path.join(HERE, "gen_run1_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
