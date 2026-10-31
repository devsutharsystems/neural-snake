"""Run 1 diagnosis. For checkpoints 10/20/30/40/50/60 min: per-head loss on a fixed 5,000-row TRAIN sample vs the 10,374 Spike A validation rows,
whole-board val accuracy, the food-head uniform floor, and exact-duplicate (state, action) counts in train. Held-out rows never loaded.
Run: .venv/bin/python spikes/spike_a/diag_run1.py   -> spikes/spike_a/diag_run1.json
Losses are computed in train mode (no BatchNorm/Dropout exists in Net3, so train mode only differs by the eval-time prior shifts, which would change over/ate logits).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import common  # noqa: E402
import run1  # noqa: E402
import v3  # noqa: E402

DEV = run1.DEV


@torch.no_grad()
def head_losses(net, T, n, bs=1000):
    net.train()
    acc = {"grid": 0.0, "game_over": 0.0, "ate": 0.0, "food_sum": 0.0, "food_n": 0.0}
    for i in range(0, n, bs):
        Tb = {k: v[i:i + bs] for k, v in T.items()}
        out = net(v3.build_x(Tb))
        tgt, cur = Tb["y14"].long(), Tb["x14"].long()
        w = 1.0 + 3.0 * (tgt != cur).float()
        m = len(tgt)
        acc["grid"] += float((F.cross_entropy(out["cls"], tgt, reduction="none") * w).mean()) * m
        acc["game_over"] += float(F.binary_cross_entropy_with_logits(out["over"], Tb["over"].float())) * m
        acc["ate"] += float(F.binary_cross_entropy_with_logits(out["ate"], Tb["eaten"].float())) * m
        valid = ((Tb["eaten"] == 1) & (Tb["food"] < 144)).float()
        ce = F.cross_entropy(out["food"], Tb["food"].long().clamp(max=143), reduction="none")
        acc["food_sum"] += float((ce * valid).sum())
        acc["food_n"] += float(valid.sum())
    return {"grid": round(acc["grid"] / n, 4), "game_over": round(acc["game_over"] / n, 4), "ate": round(acc["ate"] / n, 4),
            "food": round(acc["food_sum"] / max(acc["food_n"], 1), 4), "n_food_rows": int(acc["food_n"])}


def food_floor(y14, d):
    """mean ln(#empty cells after the step) over eat rows with a spawn: the cross-entropy of a perfect uniform-over-empty-cells predictor."""
    m = (d["eaten"] == 1) & (d["food"] < 144)
    empty = (y14[m] == 0).reshape(m.sum(), -1).sum(1)
    return round(float(np.log(empty).mean()), 4), int(m.sum())


def main():
    tr, va, n_old, n_new = run1.load_all()
    x14t, y14t = common.make_14(tr)
    x14v, y14v = common.make_14(va)
    # duplicates among TRAIN rows: identical (current 14-class grid, action)
    key = np.concatenate([x14t.reshape(len(x14t), -1), tr["act"][:, None]], 1)
    _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    n_dup_rows = int(len(key) - len(first))
    # among eat rows: same (state, action) but different spawn label
    eat = np.where((tr["eaten"] == 1) & (tr["food"] < 144))[0]
    seen, conflicts = {}, 0
    for i in eat:
        k = key[i].tobytes()
        if k in seen and seen[k] != int(tr["food"][i]):
            conflicts += 1
        seen.setdefault(k, int(tr["food"][i]))
    # same (state, action) with different target grid (non-eat steps only) -> irreducible noise check
    tgt_conf = 0
    seen_t = {}
    for i in np.where(tr["eaten"] == 0)[0][:200000]:
        k = key[i].tobytes()
        t = y14t[i].tobytes()
        if k in seen_t and seen_t[k] != t:
            tgt_conf += 1
        seen_t.setdefault(k, t)
    dup = {"train_rows": int(len(key)), "unique_state_action": int(len(first)), "duplicate_rows_(rows_beyond_first)": n_dup_rows,
           "pct_duplicate": round(100 * n_dup_rows / len(key), 2), "rows_in_a_group_of_size>1": int((cnt[inv] > 1).sum()),
           "eat_rows_with_same_state_action_but_different_spawn_cell": conflicts, "eat_rows": int(len(eat)),
           "non_eat_rows_(first 200k)_with_same_state_action_but_different_target_grid": tgt_conf}
    rng = np.random.default_rng(1)
    s = np.sort(rng.choice(len(tr["act"]), 5000, replace=False))
    ds = {k: v[s] for k, v in tr.items()}
    Ts = v3.make_device_data(ds, x14t[s], y14t[s], DEV)
    Tv = v3.make_device_data(va, x14v, y14v, DEV)
    fl_tr = food_floor(y14t[s], ds)
    fl_va = food_floor(y14v, va)
    out = {"command": "diag_run1.py", "train_sample_rows": 5000, "val_rows": int(len(va["act"])),
           "food_uniform_floor": {"train_sample": {"mean_ln_empty": fl_tr[0], "eat_rows": fl_tr[1]}, "validation": {"mean_ln_empty": fl_va[0], "eat_rows": fl_va[1]}},
           "duplicates": dup, "checkpoints": {}}
    for tag in ("10min", "20min", "30min", "40min", "50min", "final"):
        net = run1.new_net()
        net.load_state_dict(torch.load(os.path.join(run1.CKPT_DIR, f"run1_{tag}.pt"))["state_dict"])
        P = v3.predict(net, Tv, len(va["act"]))
        acc = float((P["cls"].numpy() == y14v).reshape(len(y14v), -1).all(1).mean())
        Pt = v3.predict(net, Ts, 5000)
        acc_t = float((Pt["cls"].numpy() == y14t[s]).reshape(5000, -1).all(1).mean())
        out["checkpoints"][tag] = {"train_losses": head_losses(net, Ts, 5000), "val_losses": head_losses(net, Tv, len(va["act"])),
                                   "val_whole_board_acc": round(acc, 4), "train_sample_whole_board_acc": round(acc_t, 4)}
        print(tag, json.dumps(out["checkpoints"][tag]), flush=True)
    json.dump(out, open(os.path.join(HERE, "diag_run1.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
