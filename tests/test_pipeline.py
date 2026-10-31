"""Pipeline tests: stored transitions must match the engine exactly. Skips if data/ is absent (it is gitignored).
Uses only train/validation rows (episode seed % 5 != 0); held-out rows are never loaded into the test."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "spikes", "spike_a"))
import common  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(common.DATA, "act.npy")), reason="data/spike_a missing")
N_SAMPLE = 1000


@pytest.fixture(scope="module")
def data():
    tr, va = common.load_trainval()
    cat = {k: np.concatenate([tr[k], va[k]]) for k in common.KEYS}
    return cat


def sample_idx(d, need_prev=False):
    rng = np.random.default_rng(0)
    pool = np.arange(1, len(d["act"])) if need_prev else np.arange(len(d["act"]))
    if need_prev:
        pool = pool[d["ep"][pool] == d["ep"][pool - 1]]
    return rng.choice(pool, N_SAMPLE, replace=False)


def test_r1_next_state_matches_engine(data):
    d, bad = data, []
    for i in sample_idx(data):
        eng = common.engine_from_state(d["cls"][i], d["order"][i], d["dir"][i])
        eaten = eng.step(int(d["act"][i]))
        det = common.cls4_grid(eng)
        if eaten and eng.food is not None:
            det[eng.food] = 0
        ok = (eaten == bool(d["eaten"][i]) and eng.game_over == bool(d["over"][i]) and eng.direction == d["n_dir"][i]
              and np.array_equal(det, d["n_cls"][i]) and np.array_equal(common.order_grid(eng), d["n_order"][i]))
        if eaten:  # new food comes from the seeded RNG of the original episode; check it is on an empty cell
            f = int(d["food"][i])
            ok = ok and f < 144 and d["n_cls"][i].reshape(-1)[f] == 0
        if not ok:
            bad.append(int(i))
    print(f"R1 next-state matches engine: {N_SAMPLE - len(bad)}/{N_SAMPLE}; bad rows {bad[:10]}")
    assert not bad


def test_r2_frame_stack_matches_engine(data):
    d, bad = data, []
    for i in sample_idx(data, need_prev=True):
        j = i - 1
        eng = common.engine_from_state(d["cls"][j], d["order"][j], d["dir"][j])
        prev_ok = np.array_equal(common.cls4_grid(eng), d["prev_cls"][i])
        eaten = eng.step(int(d["act"][j]))
        if eaten:
            eng.food = (int(d["food"][j]) // 12, int(d["food"][j]) % 12)
        if not (prev_ok and np.array_equal(common.cls4_grid(eng), d["cls"][i])):
            bad.append(int(i))
    print(f"R2 frame stack (t-1, t) matches engine replay: {N_SAMPLE - len(bad)}/{N_SAMPLE}; bad rows {bad[:10]}")
    assert not bad


def test_rollout_start_rebuild_roundtrip(data):
    d, bad = data, []
    x14, _ = common.make_14({k: v[sample_idx(data)] for k, v in data.items()})
    for n, i in enumerate(sample_idx(data)):
        eng = common.engine_from_state(d["cls"][i], d["order"][i], d["dir"][i])
        food = eng.food
        ok = (np.array_equal(common.cls4_grid(eng), d["cls"][i]) and np.array_equal(common.order_grid(eng), d["order"][i])
              and eng.direction == d["dir"][i] and np.array_equal(common.grid14(eng.body, eng.direction, food), x14[n]))
        if not ok:
            bad.append(int(i))
    print(f"rollout start-state rebuild: {N_SAMPLE - len(bad)}/{N_SAMPLE}; bad rows {bad[:10]}")
    assert not bad
