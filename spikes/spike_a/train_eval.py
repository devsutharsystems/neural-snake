"""Spike A: train R1 (Markov) and R2 (frame stack) tiny CNNs, evaluate on held-out episode seeds only.

Run: .venv/bin/python spikes/spike_a/train_eval.py [train_seconds]
"""
import json
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from engine.snake import Snake, N, NOOP  # noqa: E402
from gen import pick, safe_moves, cls_grid, order_grid  # noqa: E402

DATA = os.path.join(ROOT, "data", "spike_a")
TRAIN_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BUCKETS = [(1, 3), (4, 7), (8, 15), (16, 999)]
ORD_SCALE = float(N * N)


def load():
    d = {k: np.load(os.path.join(DATA, k + ".npy"), mmap_mode="r") for k in
         ["cls", "order", "dir", "act", "prev_cls", "n_cls", "n_order", "n_dir", "eaten", "food",
          "over", "ep", "pol", "held", "length"]}
    return d


def onehot_planes(cls):  # cls: [B,12,12] long -> [B,4,12,12]
    return F.one_hot(cls, 4).permute(0, 3, 1, 2).float()


def build_input(rep, cls, order, direc, act, prev):
    B = cls.shape[0]
    a = F.one_hot(act, 5).float().view(B, 5, 1, 1).expand(B, 5, N, N)
    if rep == "R1":
        d = F.one_hot(direc, 4).float().view(B, 4, 1, 1).expand(B, 4, N, N)
        return torch.cat([onehot_planes(cls), (order.float() / ORD_SCALE).unsqueeze(1), d, a], 1)
    return torch.cat([onehot_planes(cls), onehot_planes(prev), a], 1)


class Net(nn.Module):
    """Receptive field: 7 3x3 convs with dilations 1,2,4,1,2,4,1 -> RF = 1+2*(15)=31 >= 23 (covers 12x12 from any cell),
    plus a global mean/max pooled feature broadcast back after layer 3."""
    DIL = [1, 2, 4, 1, 2, 4, 1]

    def __init__(self, cin, C=48):
        super().__init__()
        self.stem = nn.Conv2d(cin, C, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(C, C, 3, padding=d, dilation=d) for d in self.DIL])
        self.glob = nn.Linear(2 * C, C)
        self.cls_head = nn.Conv2d(C, 4, 1)
        self.ord_head = nn.Conv2d(C, 1, 1)
        self.food_head = nn.Conv2d(C, 1, 1)
        self.dir_head = nn.Linear(2 * C, 4)
        self.over_head = nn.Linear(2 * C, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for i, conv in enumerate(self.convs):
            h = h + F.relu(conv(h))
            if i == 2:
                g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
                h = h + F.relu(self.glob(g)).unsqueeze(-1).unsqueeze(-1)
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return (self.cls_head(h), self.ord_head(h).squeeze(1), self.food_head(h).flatten(1),
                self.dir_head(g), self.over_head(g).squeeze(1))


def tens(a, idx, dtype=torch.long):
    return torch.from_numpy(np.ascontiguousarray(a[idx])).to(dtype).to(DEV)


def train(rep, d, train_idx, log):
    cin = 14 if rep == "R1" else 13
    torch.manual_seed(0)
    net = Net(cin).to(DEV)
    nparams = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lens = np.asarray(d["length"])[train_idx]
    bucket = np.digitize(lens, [4, 8, 16])
    cnt = np.bincount(bucket, minlength=4).astype(float)
    w = 1.0 / np.sqrt(cnt[bucket])  # sqrt-inverse-frequency: long snakes up-weighted
    w /= w.sum()
    t0 = time.time()
    step = 0
    cw = torch.tensor([1.0, 5.0, 5.0, 5.0], device=DEV)
    while time.time() - t0 < TRAIN_SECONDS:
        frac = (time.time() - t0) / TRAIN_SECONDS
        for g in opt.param_groups:
            g["lr"] = 2e-3 * (1 - 0.9 * frac)
        idx = np.sort(np.random.choice(train_idx, 256, p=w))
        x = build_input(rep, tens(d["cls"], idx), tens(d["order"], idx), tens(d["dir"], idx),
                        tens(d["act"], idx), tens(d["prev_cls"], idx))
        lc, lo, lf, ld, lg = net(x)
        t_cls = tens(d["n_cls"], idx)
        loss = F.cross_entropy(lc, t_cls, weight=cw)
        t_over = tens(d["over"], idx, torch.float)
        loss = loss + F.binary_cross_entropy_with_logits(lg, t_over)
        eat = tens(d["eaten"], idx, torch.bool)
        food = tens(d["food"], idx)
        ok = eat & (food < 144)
        if ok.any():
            loss = loss + F.cross_entropy(lf[ok], food[ok])
        if rep == "R1":
            t_ord = tens(d["n_order"], idx, torch.float) / ORD_SCALE
            loss = loss + 20 * F.l1_loss(lo, t_ord) + F.cross_entropy(ld, tens(d["n_dir"], idx))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 500 == 0:
            log(f"{rep} step {step} t={time.time()-t0:.0f}s loss={loss.item():.4f}")
    wall = time.time() - t0
    return net, {"params": nparams, "steps": step, "train_wall_s": round(wall, 1), "device": str(DEV)}


@torch.no_grad()
def predict(net, rep, cls, order, direc, act, prev):
    """numpy in -> dict of decoded numpy out. cls/order/prev: [B,12,12]; direc/act: [B]."""
    net.eval()
    x = build_input(rep, torch.from_numpy(cls).long().to(DEV), torch.from_numpy(order).long().to(DEV),
                    torch.from_numpy(direc).long().to(DEV), torch.from_numpy(act).long().to(DEV),
                    torch.from_numpy(prev).long().to(DEV))
    lc, lo, lf, ld, lg = net(x)
    out = {
        "cls": lc.argmax(1).cpu().numpy().astype(np.uint8),
        "order": np.clip(np.rint(lo.cpu().numpy() * ORD_SCALE), 0, 144).astype(np.uint8),
        "dir": ld.argmax(1).cpu().numpy(),
        "over": (lg > 0).cpu().numpy(),
        "food_probs": F.softmax(lf, 1).cpu(),
    }
    return out


def sample_food(probs):
    return torch.multinomial(probs, 1).squeeze(1).numpy()


def eval_onestep(net, rep, d, held_idx):
    res = {}
    cls, order = np.asarray(d["cls"])[held_idx], np.asarray(d["order"])[held_idx]
    direc, act = np.asarray(d["dir"])[held_idx], np.asarray(d["act"])[held_idx]
    prev = np.asarray(d["prev_cls"])[held_idx]
    ncls = np.asarray(d["n_cls"])[held_idx]
    nord, ndir = np.asarray(d["n_order"])[held_idx], np.asarray(d["n_dir"])[held_idx]
    over = np.asarray(d["over"])[held_idx].astype(bool)
    eaten = np.asarray(d["eaten"])[held_idx].astype(bool)
    length = np.asarray(d["length"])[held_idx]
    preds = [predict(net, rep, cls[i:i + 512], order[i:i + 512], direc[i:i + 512], act[i:i + 512], prev[i:i + 512])
             for i in range(0, len(held_idx), 512)]
    pc = np.concatenate([p["cls"] for p in preds])
    po = np.concatenate([p["order"] for p in preds])
    pd_ = np.concatenate([p["dir"] for p in preds])
    pov = np.concatenate([p["over"] for p in preds])
    probs = torch.cat([p["food_probs"] for p in preds])
    base_match = (cls == ncls).reshape(len(cls), -1).all(1)
    grid_match = (pc == ncls).reshape(len(cls), -1).all(1)
    body = ncls >= 2
    full_match = grid_match & ((po * body == nord * body).reshape(len(cls), -1).all(1)) & (pd_ == ndir)
    for name, (lo_, hi_) in zip(["1-3", "4-7", "8-15", "16+"], BUCKETS):
        m = (length >= lo_) & (length <= hi_)
        res[name] = {"n": int(m.sum()), "baseline_copy_prev": float(base_match[m].mean()) if m.any() else None,
                     "model_grid_exact": float(grid_match[m].mean()) if m.any() else None}
        if rep == "R1":
            res[name]["model_full_state_exact(grid+order+dir)"] = float(full_match[m].mean()) if m.any() else None
    res["all"] = {"n": int(len(cls)), "baseline_copy_prev": float(base_match.mean()),
                  "model_grid_exact": float(grid_match.mean())}
    if rep == "R1":
        res["all"]["model_full_state_exact(grid+order+dir)"] = float(full_match.mean())
    tp, fp, fn = (pov & over).sum(), (pov & ~over).sum(), (~pov & over).sum()
    res["game_over"] = {"n_true_over": int(over.sum()), "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1)),
                        "baseline_precision_recall": "copy-prev never predicts game-over (0/0)"}
    # food validity: sample on eat steps only
    em = np.where(eaten)[0]
    sampled = sample_food(probs[em]) if len(em) else np.array([])
    true_empty = [ncls[i].reshape(-1)[s] == 0 for i, s in zip(em, sampled)]
    pred_empty = [pc[i].reshape(-1)[s] == 0 for i, s in zip(em, sampled)]
    res["food"] = {"n_eat_steps": int(len(em)),
                   "pct_on_empty_cell_true_grid": float(100 * np.mean(true_empty)) if len(em) else None,
                   "pct_on_empty_cell_model_grid": float(100 * np.mean(pred_empty)) if len(em) else None}
    return res


def r2_info_limit(d, idx):
    cls, prev = np.asarray(d["cls"])[idx], np.asarray(d["prev_cls"])[idx]
    act, ncls = np.asarray(d["act"])[idx], np.asarray(d["n_cls"])[idx]
    groups = {}
    for i in range(len(idx)):
        key = cls[i].tobytes() + prev[i].tobytes() + bytes([act[i]])
        groups.setdefault(key, []).append(ncls[i].tobytes())
    conflicting = [v for v in groups.values() if len(set(v)) > 1]
    n_in_conflict = sum(len(v) for v in conflicting)
    repeated = sum(1 for v in groups.values() if len(v) > 1)
    return {"transitions": int(len(idx)), "distinct_inputs": len(groups), "inputs_seen_more_than_once": repeated,
            "inputs_with_conflicting_true_next_state": len(conflicting),
            "pct_transitions_in_conflicting_inputs": round(100 * n_in_conflict / len(idx), 3)}


# ---------------------------------------------------------------- rollouts
def engine_from_state(cls, order, direc):
    s = Snake(seed=12345)
    cells = [(r, c) for r in range(N) for c in range(N) if order[r, c] > 0]
    cells.sort(key=lambda rc: -order[rc])
    s.body = cells
    s.direction = int(direc)
    fr = np.argwhere(cls == 1)
    s.food = tuple(int(v) for v in fr[0]) if len(fr) else None
    return s


def engine_det(s, eaten):
    g = cls_grid(s)
    if eaten and s.food is not None:
        g[s.food[0], s.food[1]] = 0
    return g


def valid_state(det, food_cell, expected_len):
    heads = int((det == 2).sum())
    bodyc = int((det == 3).sum())
    if heads != 1:
        return "heads!=1"
    if heads + bodyc != expected_len:
        return "wrong_length"
    cells = {(int(r), int(c)) for r, c in np.argwhere(det >= 2)}
    start = tuple(int(v) for v in np.argwhere(det == 2)[0])
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
    if food_cell is not None and det.reshape(-1)[food_cell] >= 2:
        return "food_on_snake"
    return None


def ascii_pair(eng, mod):
    ch = ".Fho"
    rows = []
    for r in range(N):
        rows.append("".join(ch[v] for v in eng[r]) + "   " + "".join(ch[v] for v in mod[r]))
    return "engine           model\n" + "\n".join(rows)


def rollouts(net, rep, d, held_idx, kind, n=50, horizon=20, seed=0):
    rng = random.Random(seed)
    lens = np.asarray(d["length"])[held_idx]
    over = np.asarray(d["over"])[held_idx]
    cands = [[i for i in held_idx[(lens >= lo) & (lens <= hi) & (over == 0)]] for lo, hi in BUCKETS]
    starts = []
    for k in range(n):
        pool = cands[k % 4] or cands[0]
        starts.append(rng.choice(pool))
    first_mis, matched_end, first_invalid, fails, invalid_reasons = [], 0, [], [], []
    for k, ti in enumerate(starts):
        cls, order, direc = np.asarray(d["cls"][ti]), np.asarray(d["order"][ti]), int(d["dir"][ti])
        prev = np.asarray(d["prev_cls"][ti])
        eng = engine_from_state(cls, order, direc)
        m_cls, m_ord, m_dir, m_prev = cls.copy(), order.copy(), direc, prev.copy()
        mis, inval, ended, m_len = None, None, False, len(eng.body)
        arng = random.Random(seed * 1000 + k)
        for t in range(1, horizon + 1):
            if kind == "bfs":
                a = pick("bfs", eng, arng)
            else:
                sm = safe_moves(eng)
                a = arng.choice(sm) if sm else arng.choice([0, 1, 2, 3, 4])
            eaten = eng.step(a)
            e_det = engine_det(eng, eaten)
            p = predict(net, rep, m_cls[None], m_ord[None], np.array([m_dir]), np.array([a]), m_prev[None])
            det = p["cls"][0].copy()
            food_cell = None
            if eaten:
                food_cell = int(sample_food(p["food_probs"])[0])
            m_len = m_len + (1 if eaten else 0)
            if not (eng.game_over and not eaten):
                reason = valid_state(det, food_cell, m_len)
                if reason and inval is None:
                    inval = t
                    invalid_reasons.append(reason)
            if mis is None and (not np.array_equal(det, e_det) or bool(p["over"][0]) != eng.game_over):
                mis = t
                if len(fails) < 2 + 8:
                    fails.append({"rollout": k, "kind": kind, "step": t, "action": int(a),
                                  "ascii": ascii_pair(e_det, det)})
            if eng.game_over:
                ended = True
                break
            # feedback: model's own prediction, food synced to engine after each eat
            nxt = det.copy()
            if eaten and eng.food is not None:
                nxt[eng.food[0], eng.food[1]] = 1
            m_prev, m_cls, m_ord, m_dir = m_cls, nxt, p["order"][0], int(p["dir"][0])
        if mis is None and ended:
            matched_end += 1
        first_mis.append(mis if mis is not None else horizon + 1)  # censored: no mismatch within horizon
        first_invalid.append(inval if inval is not None else horizon + 1)
    fm = np.array(first_mis)
    fi = np.array(first_invalid)
    return {"kind": kind, "n": n, "horizon": horizon,
            "first_mismatch_step_median": float(np.median(fm)), "first_mismatch_step_p10": float(np.percentile(fm, 10)),
            "no_mismatch_within_horizon": int(((fm == horizon + 1)).sum()),
            "matched_to_end(game over matches engine)": matched_end,
            "first_invalid_step_median": float(np.median(fi)), "first_invalid_step_p10": float(np.percentile(fi, 10)),
            "no_invalid_within_horizon": int((fi == horizon + 1).sum()),
            "invalid_reasons": {r: invalid_reasons.count(r) for r in set(invalid_reasons)},
            "note": f"values of {horizon+1} mean censored (no event within {horizon} steps)"}, fails


def main():
    log = lambda s: print(s, flush=True)
    d = load()
    held = np.asarray(d["held"]).astype(bool)
    train_idx = np.where(~held)[0]
    held_idx = np.where(held)[0]
    assert set(np.asarray(d["ep"])[train_idx]).isdisjoint(set(np.asarray(d["ep"])[held_idx]))
    hist = {}
    for name, idx in [("train", train_idx), ("heldout", held_idx)]:
        l = np.asarray(d["length"])[idx]
        hist[name] = {b: int(((l >= lo) & (l <= hi)).sum()) for b, (lo, hi) in zip(["1-3", "4-7", "8-15", "16+"], BUCKETS)}
    results = {"device": str(DEV), "train_seconds_budget": TRAIN_SECONDS, "length_hist": hist,
               "n_train": int(len(train_idx)), "n_heldout": int(len(held_idx)),
               "gen_stats": json.load(open(os.path.join(HERE, "gen_stats.json")))}
    results["R2_information_limit_heldout"] = r2_info_limit(d, held_idx)
    results["R2_information_limit_all_data"] = r2_info_limit(d, np.arange(len(held)))
    all_fails = {}
    for rep in ["R1", "R2"]:
        net, info = train(rep, d, train_idx, log)
        r = {"train": info, "one_step": eval_onestep(net, rep, d, held_idx)}
        r["rollout_bfs"], f1 = rollouts(net, rep, d, held_idx, "bfs", seed=1)
        r["rollout_random_safe"], f2 = rollouts(net, rep, d, held_idx, "random", seed=2)
        all_fails[rep] = (f1 + f2)[:2]
        results[rep] = r
        log(f"{rep} done: {json.dumps(r['one_step']['all'])}")
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    with open(os.path.join(HERE, "failures.md"), "w") as f:
        for rep, fl in all_fails.items():
            for x in fl:
                f.write(f"### {rep} {x['kind']} rollout {x['rollout']} step {x['step']} action {x['action']}\n```\n{x['ascii']}\n```\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
