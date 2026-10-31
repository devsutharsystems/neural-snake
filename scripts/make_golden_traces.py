"""Write 50 Python-engine golden traces to tests/golden/traces.json for the JS engine port (web/engine.js).
Each trace: n, seed, policy, initial state, actions, food spawn positions (one per eat, in order; null = board full) and the state after EVERY step:
grid14 as a string of base-16 digits (14 classes -> '0'..'d'), plus direction, food, game_over, won, score, eaten.
Run: .venv/bin/python scripts/make_golden_traces.py
Coverage: 40 traces on the real 12x12 board with the 5 Spike A policies (random, greedy, bfs, bfs_eps20, greedy_eps30), 5 reversal/NOOP-heavy random traces,
3 traces on a 5x5 board, and 2 traces on tiny boards that end in a WIN (board full, food=None) if such seeds are found.
"""
import json
import os
import random
import sys
from collections import deque

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spikes", "spike_a"))
from engine.snake import Snake, DELTA, OPPOSITE, UP, DOWN, LEFT, RIGHT, NOOP  # noqa: E402
import common  # noqa: E402
import gen  # noqa: E402

DIR_OF = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}
HEX = "0123456789abcdef"
MAX_STEPS = 150


def grid14(s):
    n = s.n
    g = [[0] * n for _ in range(n)]
    if s.food is not None:
        g[s.food[0]][s.food[1]] = 1
    for i, (r, c) in enumerate(s.body):
        if i == 0:
            g[r][c] = 2 + s.direction
        else:
            pr, pc = s.body[i - 1]
            g[r][c] = (10 if i == len(s.body) - 1 else 6) + DIR_OF[(pr - r, pc - c)]
    return g


def encode(g):
    return "".join(HEX[v] for row in g for v in row)


def generic_bfs(s, rng):
    """BFS toward food on any board size (gen.pick assumes 12x12); falls back to a random safe move."""
    def safe(a):
        if len(s.body) > 1 and a == OPPOSITE[s.direction]:
            return False
        r, c = s.body[0][0] + DELTA[a][0], s.body[0][1] + DELTA[a][1]
        if not (0 <= r < s.n and 0 <= c < s.n):
            return False
        return (r, c) not in (s.body if (r, c) == s.food else s.body[:-1])
    moves = [a for a in (UP, DOWN, LEFT, RIGHT) if safe(a)]
    if not moves:
        return NOOP
    if s.food is not None:
        blocked, start = set(s.body[:-1]), s.body[0]
        prev, q = {start: None}, deque([start])
        while q:
            cur = q.popleft()
            if cur == s.food:
                break
            for a in (UP, DOWN, LEFT, RIGHT):
                nb = (cur[0] + DELTA[a][0], cur[1] + DELTA[a][1])
                if 0 <= nb[0] < s.n and 0 <= nb[1] < s.n and nb not in blocked and nb not in prev:
                    prev[nb] = (cur, a)
                    q.append(nb)
        if s.food in prev:
            node = s.food
            while prev[node][0] != start:
                node = prev[node][0]
            if prev[node][1] in moves:
                return prev[node][1]
    return rng.choice(moves)


def run(seed, n, policy, max_steps=MAX_STEPS):
    s = Snake(seed=seed, n=n)
    rng = random.Random(seed + 7)
    init = {"body": [list(c) for c in s.body], "direction": s.direction, "food": list(s.food), "grid14": encode(grid14(s))}
    actions, spawns, states = [], [], []
    for _ in range(max_steps):
        if s.game_over:
            break
        if policy == "reverse_noop":
            a = rng.choice([UP, DOWN, LEFT, RIGHT, NOOP, NOOP, OPPOSITE[s.direction]])
        elif policy == "bfs_generic":
            a = generic_bfs(s, rng)
        elif policy == "random5":
            a = rng.choice([UP, DOWN, LEFT, RIGHT, NOOP])
        else:
            a = gen.pick(policy, s, rng)
        eaten = s.step(a)
        actions.append(int(a))
        if eaten:
            spawns.append(list(s.food) if s.food is not None else None)
        states.append({"grid14": encode(grid14(s)), "direction": s.direction, "food": list(s.food) if s.food is not None else None,
                       "game_over": s.game_over, "won": s.won, "score": s.score, "eaten": bool(eaten)})
    return {"n": n, "seed": seed, "policy": policy, "initial": init, "actions": actions, "spawns": spawns, "states": states}


def main():
    traces = []
    policies = ["random", "greedy", "bfs", "bfs_eps20", "greedy_eps30"]
    for i in range(40):
        traces.append(run(5000 + i, 12, policies[i % 5]))
    for i in range(5):
        traces.append(run(6000 + i, 12, "reverse_noop"))
    for i in range(3):
        traces.append(run(7000 + i, 5, "bfs_generic" if i else "random5"))
    won = 0
    for n in (3, 4):  # search for seeds whose generic-BFS game fills the board (win), up to a bounded number of tries
        for seed in range(8000, 8000 + 20000):
            t = run(seed, n, "bfs_generic", max_steps=200)
            if t["states"] and t["states"][-1]["won"]:
                traces.append(t)
                won += 1
                break
    # fall back to non-winning small-board traces so that the total is exactly 50
    k = 0
    while len(traces) < 50:
        traces.append(run(9000 + k, 4, "bfs_generic", max_steps=60))
        k += 1
    for t in traces:  # the generalised grid14 above must equal spikes/spike_a/common.py grid14 on the 12x12 board
        if t["n"] == 12:
            s = Snake(seed=t["seed"])
            assert t["initial"]["grid14"] == encode(common.grid14(s.body, s.direction, s.food).tolist()), t["seed"]
    out = {"description": "Golden traces from engine/snake.py (scripts/make_golden_traces.py). grid14: base-16 digit string, row-major.",
           "classes": "0 empty,1 food,2-5 head U/D/L/R,6-9 body,10-13 tail", "action_codes": "0 UP,1 DOWN,2 LEFT,3 RIGHT,4 NOOP", "traces": traces}
    path = os.path.join(ROOT, "tests", "golden", "traces.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{len(traces)} traces, {sum(len(t['actions']) for t in traces)} steps, {sum(len(t['spawns']) for t in traces)} eats, "
          f"win traces: {sum(1 for t in traces if t['states'] and t['states'][-1]['won'])}, game-over traces: {sum(1 for t in traces if t['states'] and t['states'][-1]['game_over'])}, "
          f"file bytes: {os.path.getsize(path)}")


if __name__ == "__main__":
    main()
