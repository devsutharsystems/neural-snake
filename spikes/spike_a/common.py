"""Shared helpers for the Spike A debug iteration (numpy only, no torch). Never touches held-out rows."""
import json
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
from engine.snake import Snake, N, UP, DOWN, LEFT, RIGHT  # noqa: E402

DATA = os.path.join(ROOT, "data", "spike_a")
KEYS = ["cls", "order", "dir", "act", "prev_cls", "n_cls", "n_order", "n_dir", "eaten", "food", "over", "ep", "pol", "length"]
DIR_OF = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}
# 14 classes: 0 empty, 1 food, 2-5 head(dir of travel U,D,L,R), 6-9 body(dir to next segment toward head), 10-13 tail (same)
TO4 = np.array([0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3], np.uint8)


def write_split_v2():
    p = os.path.join(HERE, "split_v2.json")
    json.dump({"rule": "episode seed % 5 == 0 -> held-out (NOT USED in v2); seed % 5 == 1 -> validation; others -> train"},
              open(p, "w"))


def load_trainval():
    """Return (train, val) dicts of in-memory arrays. Held-out rows (seed % 5 == 0) are dropped immediately and never used."""
    held = np.asarray(np.load(os.path.join(DATA, "held.npy"), mmap_mode="r")).astype(bool)
    ep_all = np.asarray(np.load(os.path.join(DATA, "ep.npy"), mmap_mode="r"))
    keep = ~held
    out_t, out_v = {}, {}
    is_val_all = (ep_all % 5 == 1)
    tr_mask, va_mask = keep & ~is_val_all, keep & is_val_all
    for k in KEYS:
        a = np.load(os.path.join(DATA, k + ".npy"), mmap_mode="r")
        out_t[k] = np.ascontiguousarray(a[tr_mask])
        out_v[k] = np.ascontiguousarray(a[va_mask])
    assert not (set(out_t["ep"] % 5) & {0}) and not (set(out_v["ep"] % 5) - {1})
    return out_t, out_v


def body_from_state(cls, order):
    cells = [(r, c) for r in range(N) for c in range(N) if order[r, c] > 0]
    cells.sort(key=lambda rc: -int(order[rc]))
    return cells


def grid14(body, direction, food):
    g = np.zeros((N, N), np.uint8)
    if food is not None:
        g[food[0], food[1]] = 1
    L = len(body)
    for i, (r, c) in enumerate(body):
        if i == 0:
            g[r, c] = 2 + int(direction)
        else:
            pr, pc = body[i - 1]
            d = DIR_OF[(pr - r, pc - c)]
            g[r, c] = (10 if i == L - 1 else 6) + d
    return g


def food_of(cls):
    fr = np.argwhere(cls == 1)
    return (int(fr[0][0]), int(fr[0][1])) if len(fr) else None


def make_14(d):
    """x14 (current), y14 (true deterministic next grid) for all rows of d."""
    n = len(d["act"])
    x = np.zeros((n, N, N), np.uint8)
    y = np.zeros((n, N, N), np.uint8)
    for i in range(n):
        x[i] = grid14(body_from_state(d["cls"][i], d["order"][i]), d["dir"][i], food_of(d["cls"][i]))
        y[i] = grid14(body_from_state(d["n_cls"][i], d["n_order"][i]), d["n_dir"][i], food_of(d["n_cls"][i]))
    return x, y


def cls4_grid(s):
    return np.array(s.grid(), np.uint8)


def order_grid(s):
    o = np.zeros((N, N), np.uint8)
    L = len(s.body)
    for i, (r, c) in enumerate(s.body):
        o[r, c] = L - i
    return o


def engine_from_state(cls, order, direc):
    s = Snake(seed=12345)
    s.body = body_from_state(cls, order)
    s.direction = int(direc)
    s.food = food_of(cls)
    return s


def valid_state(det4, food_cell, expected_len):
    heads = int((det4 == 2).sum())
    bodyc = int((det4 == 3).sum())
    if heads != 1:
        return "heads!=1"
    if heads + bodyc != expected_len:
        return "wrong_length"
    cells = {(int(r), int(c)) for r, c in np.argwhere(det4 >= 2)}
    start = tuple(int(v) for v in np.argwhere(det4 == 2)[0])
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    if seen != cells:
        return "disconnected"
    if food_cell is not None and det4.reshape(-1)[food_cell] >= 2:
        return "food_on_snake"
    return None


def ascii14(g):
    ch14 = [".", "F", "^", "v", "<", ">", "↑", "↓", "←", "→", "T", "T", "T", "T"]
    return ["".join(ch14[v] for v in g[r]) for r in range(N)]


def ascii_pair14(eng, mod):
    a, b = ascii14(eng), ascii14(mod)
    return "engine (head ^v<>, body arrows, tail T)   model\n" + "\n".join(x + "   " + y for x, y in zip(a, b))


def info_limit_tail(d, cap=20000):
    """Fraction of transitions where frames t-1 and t do not uniquely identify which cell vacates at t+1 (the tail of state t).
    Method (see results_v2.md): enumerate every self-avoiding head-to-end path over exactly the snake cells of frame t, 4-connected,
    starting at the head, with the neck forced to be the head of frame t-1 and (if the snake did not grow) the new end adjacent to the
    cell that left the snake between t-1 and t. If the valid paths end in more than one distinct cell, the tail is ambiguous."""
    n = len(d["act"])
    ambiguous = capped = counted = 0
    for i in range(n):
        cur, prev = d["cls"][i], d["prev_cls"][i]
        S = {(int(r), int(c)) for r, c in np.argwhere(cur >= 2)}
        Sp = {(int(r), int(c)) for r, c in np.argwhere(prev >= 2)}
        counted += 1
        if len(S) <= 2:
            continue  # length 1-2: tail is the only non-head cell or the head itself
        head = tuple(int(v) for v in np.argwhere(cur == 2)[0])
        phead = tuple(int(v) for v in np.argwhere(prev == 2)[0])
        left = Sp - S  # cell that left the snake (empty set if the snake grew)
        neck = phead if phead != head and phead in S else None
        ends, nodes = set(), [0]
        stop = [False]

        def dfs(cell, visited, path_len):
            if stop[0]:
                return
            nodes[0] += 1
            if nodes[0] > cap:
                stop[0] = True
                return
            if path_len == len(S):
                if left and not any(abs(cell[0] - l[0]) + abs(cell[1] - l[1]) == 1 for l in left):
                    return
                ends.add(cell)
                if len(ends) > 1:
                    stop[0] = True
                return
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cell[0] + dr, cell[1] + dc)
                if nb in S and nb not in visited:
                    if path_len == 1 and neck is not None and nb != neck:
                        continue
                    visited.add(nb)
                    dfs(nb, visited, path_len + 1)
                    visited.discard(nb)

        dfs(head, {head}, 1)
        if nodes[0] > cap and len(ends) <= 1:
            capped += 1
        elif len(ends) > 1:
            ambiguous += 1
    return {"transitions": counted, "ambiguous_tail": ambiguous, "search_capped_undetermined": capped,
            "pct_ambiguous": round(100 * ambiguous / counted, 3),
            "pct_ambiguous_or_capped_upper_bound": round(100 * (ambiguous + capped) / counted, 3)}
