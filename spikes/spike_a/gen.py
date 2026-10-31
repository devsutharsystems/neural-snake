"""Spike A data generation. Writes uint8 arrays to data/spike_a/ (gitignored).

Run: .venv/bin/python spikes/spike_a/gen.py
The episode-seed split is written to spikes/spike_a/split.json BEFORE any data is generated.
"""
import json
import os
import random
import sys
from collections import deque

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from engine.snake import Snake, UP, DOWN, LEFT, RIGHT, NOOP, DELTA, N  # noqa: E402

OUT = os.path.join(ROOT, "data", "spike_a")
CAP = 500
POLICIES = ["random", "greedy", "bfs", "bfs_eps20", "greedy_eps30"]
TARGET_PER_POLICY = 10_000


def safe_moves(s):
    out = []
    for a in (UP, DOWN, LEFT, RIGHT):
        if len(s.body) > 1 and (a == {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}[s.direction]):
            continue
        dr, dc = DELTA[a]
        r, c = s.body[0][0] + dr, s.body[0][1] + dc
        if not (0 <= r < N and 0 <= c < N):
            continue
        blocking = s.body if (r, c) == s.food else s.body[:-1]
        if (r, c) in blocking:
            continue
        out.append(a)
    return out


def greedy(s, rng):
    safe = safe_moves(s)
    if not safe:
        return NOOP
    fr, fc = s.food
    hr, hc = s.body[0]
    best = min(safe, key=lambda a: abs(hr + DELTA[a][0] - fr) + abs(hc + DELTA[a][1] - fc))
    return best


def bfs(s, rng):
    safe = safe_moves(s)
    if not safe:
        return NOOP
    blocked = set(s.body[:-1])
    start = s.body[0]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == s.food:
            break
        for a in (UP, DOWN, LEFT, RIGHT):
            nxt = (cur[0] + DELTA[a][0], cur[1] + DELTA[a][1])
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in blocked and nxt not in prev:
                prev[nxt] = (cur, a)
                q.append(nxt)
    if s.food in prev:
        node = s.food
        while prev[node][0] != start:
            node = prev[node][0]
        a = prev[node][1]
        if a in safe:
            return a
    return rng.choice(safe)


def random_policy(s, rng):
    return rng.choice([UP, DOWN, LEFT, RIGHT, NOOP])


def pick(policy, s, rng):
    if policy == "random":
        return random_policy(s, rng)
    if policy == "greedy":
        return greedy(s, rng)
    if policy == "bfs":
        return bfs(s, rng)
    if policy == "bfs_eps20":
        return random_policy(s, rng) if rng.random() < 0.2 else bfs(s, rng)
    if policy == "greedy_eps30":
        return random_policy(s, rng) if rng.random() < 0.3 else greedy(s, rng)
    raise ValueError(policy)


def order_grid(s):
    o = np.zeros((N, N), np.uint8)
    L = len(s.body)
    for i, (r, c) in enumerate(s.body):
        o[r, c] = L - i
    return o


def cls_grid(s):
    return np.array(s.grid(), np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    # --- split by episode seed BEFORE generating any data
    n_ep_per_policy = 4000  # upper bound; generation stops at TARGET_PER_POLICY transitions
    seeds = {p: list(range(i * 100000, i * 100000 + n_ep_per_policy)) for i, p in enumerate(POLICIES)}
    held = {p: [sd for sd in v if sd % 5 == 0] for p, v in seeds.items()}
    split = {"rule": "episode seed % 5 == 0 -> held-out", "heldout_mod": 5}
    with open(os.path.join(os.path.dirname(__file__), "split.json"), "w") as f:
        json.dump(split, f)

    rows = {k: [] for k in ["cls", "order", "dir", "act", "prev_cls", "n_cls", "n_order", "n_dir",
                            "eaten", "food", "over", "ep", "pol", "held", "length"]}
    ep_lens = {p: [] for p in POLICIES}
    for pi, p in enumerate(POLICIES):
        total = 0
        for sd in seeds[p]:
            if total >= TARGET_PER_POLICY:
                break
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
                rows["food"].append(s.food[0] * N + s.food[1] if (eaten and s.food is not None) else 255)
                rows["over"].append(int(s.game_over))
                rows["ep"].append(sd); rows["pol"].append(pi); rows["held"].append(int(sd % 5 == 0))
                rows["length"].append(len(s.body) - (1 if eaten else 0))
                prev = cur_cls
                steps += 1
                total += 1
            ep_lens[p].append(steps)
    for k, v in rows.items():
        dt = np.int32 if k == "ep" else np.uint8
        np.save(os.path.join(OUT, k + ".npy"), np.array(v, dtype=dt))
    stats = {p: {"episodes": len(v), "median_len": float(np.median(v)), "max_len": int(max(v))}
             for p, v in ep_lens.items()}
    n = len(rows["ep"])
    stats["total_transitions"] = n
    stats["heldout_transitions"] = int(sum(rows["held"]))
    with open(os.path.join(os.path.dirname(__file__), "gen_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
