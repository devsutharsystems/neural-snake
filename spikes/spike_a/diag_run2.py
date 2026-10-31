"""Run 2 evaluation-only diagnostics on checkpoints/run2_best.pt (no training; held-out rows never loaded):
 1. food-head cross-entropy on the 10,374 Spike A validation rows vs the uniform-over-empty-cells floor
 2. how many validation (state, action) pairs also occur in the train set (old + new), and accuracy on the pairs that do not
Run: .venv/bin/python spikes/spike_a/diag_run2.py  -> spikes/spike_a/diag_run2.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import common  # noqa: E402
import run2  # noqa: E402
import v3  # noqa: E402
from diag_run1 import head_losses, food_floor  # noqa: E402

DEV = run2.DEV


def main():
    tr, va, _ = run2.load_all2()
    x14t, _ = common.make_14(tr)
    x14v, y14v = common.make_14(va)
    net = run2.new_net()
    net.load_state_dict(torch.load(run2.BEST)["state_dict"])
    Tv = v3.make_device_data(va, x14v, y14v, DEV)
    n = len(y14v)
    hl = head_losses(net, Tv, n)
    floor, n_eat = food_floor(y14v, va)
    P = v3.predict(net, Tv, n)
    ok = (P["cls"].numpy() == y14v).reshape(n, -1).all(1)
    pk = lambda x, a: np.concatenate([x.reshape(len(x), -1), a[:, None]], 1)
    train_keys = {r.tobytes() for r in pk(x14t, tr["act"])}
    seen = np.array([r.tobytes() in train_keys for r in pk(x14v, va["act"])])
    L = va["length"]
    out = {"command": "diag_run2.py", "checkpoint": "checkpoints/run2_best.pt", "val_rows": n,
           "val_head_losses_(train-mode raw logits)": hl, "food_uniform_floor_mean_ln_empty": floor, "eat_rows": n_eat,
           "val_pairs_also_in_train": {"count": int(seen.sum()), "pct": round(100 * float(seen.mean()), 2),
                                       "accuracy_on_seen": round(float(ok[seen].mean()), 5),
                                       "accuracy_on_unseen": round(float(ok[~seen].mean()), 5), "n_unseen": int((~seen).sum()),
                                       "pct_seen_by_length_bucket": {b: round(100 * float(seen[(L >= lo) & (L <= hi)].mean()), 1)
                                                                      for b, lo, hi in [("1-3", 1, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 999)]},
                                       "accuracy_on_unseen_16plus": round(float(ok[~seen & (L >= 16)].mean()), 5), "n_unseen_16plus": int((~seen & (L >= 16)).sum())}}
    json.dump(out, open(os.path.join(HERE, "diag_run2.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
