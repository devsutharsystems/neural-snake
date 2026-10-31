"""Spike A debug iteration. Train/validation only (held-out seeds are never used).

Usage (always after `source scripts/env.sh`):
  .venv/bin/python spikes/spike_a/v2.py overfit <R1old|R1v2> [seconds]
  .venv/bin/python spikes/spike_a/v2.py full [seconds]
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import common  # noqa: E402
from common import N, TO4  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gen import pick, safe_moves  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BUCKETS = [("1-3", 1, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 999)]
HORIZON = 20
ORD_SCALE = 144.0


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    """Receptive field: stem 3x3 + 7 dilated 3x3 convs (dilations 1,2,4,1,2,4,1) = 1 + 2*(1+15) = 33 cells wide (covers 12x12 from any cell).
    Two global paths (after conv 2 and conv 5): mean+max pool over all cells -> Linear -> broadcast add to every cell, so a signal computed
    at the head (head's next cell is food) can reach the tail cell and be used by later layers."""
    DIL = [1, 2, 4, 1, 2, 4, 1]

    def __init__(self, cin, ncls, C=int(os.environ.get("NEURAL_C", "32"))):
        super().__init__()
        self.stem = nn.Conv2d(cin, C, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(C, C, 3, padding=d, dilation=d) for d in self.DIL])
        self.g1 = nn.Linear(2 * C, C)
        self.g2 = nn.Linear(2 * C, C)
        self.cls_head = nn.Conv2d(C, ncls, 1)
        self.food_head = nn.Conv2d(C, 1, 1)
        self.over_head = nn.Linear(2 * C, 1)
        self.ord_head = nn.Conv2d(C, 1, 1)  # used only by R1old
        self.dir_head = nn.Linear(2 * C, 4)  # used only by R1old

    @staticmethod
    def pool(h):
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for i, conv in enumerate(self.convs):
            h = h + F.relu(conv(h))
            if i == 1:
                h = h + F.relu(self.g1(self.pool(h)))[:, :, None, None]
            if i == 4:
                h = h + F.relu(self.g2(self.pool(h)))[:, :, None, None]
        g = self.pool(h)
        return {"cls": self.cls_head(h), "food": self.food_head(h).flatten(1), "over": self.over_head(g).squeeze(1),
                "ord": self.ord_head(h).squeeze(1), "dir": self.dir_head(g)}


def oh(t, k):
    return F.one_hot(t.long(), k).permute(0, 3, 1, 2).float()


def act_planes(a, B):
    return F.one_hot(a.long(), 5).float().view(B, 5, 1, 1).expand(B, 5, N, N)


def build_input(rep, T):
    """T: dict of device tensors for a batch."""
    B = T["act"].shape[0]
    a = act_planes(T["act"], B)
    if rep == "R1v2":
        return torch.cat([oh(T["x14"], 14), a], 1)
    if rep == "R1old":  # the Phase 0 encoding: 4 classes + body order as a scalar/144 plane + direction as constant planes
        d = F.one_hot(T["dir"].long(), 4).float().view(B, 4, 1, 1).expand(B, 4, N, N)
        return torch.cat([oh(T["cls"], 4), (T["order"].float() / ORD_SCALE).unsqueeze(1), d, a], 1)
    return torch.cat([oh(T["cls"], 4), oh(T["prev_cls"], 4), a], 1)  # R2


def cin_of(rep):
    return {"R1v2": 19, "R1old": 14, "R2": 13}[rep]


def ncls_of(rep):
    return 14 if rep == "R1v2" else 4


def target_cls(rep, T):
    return T["y14"] if rep == "R1v2" else T["n_cls"]


def cur_cls(rep, T):
    return T["x14"] if rep == "R1v2" else T["cls"]


def to_device(d, x14=None, y14=None):
    T = {k: torch.from_numpy(np.ascontiguousarray(d[k])).to(DEV) for k in
         ["cls", "order", "dir", "act", "prev_cls", "n_cls", "n_order", "n_dir", "eaten", "food", "over", "length"]}
    if x14 is not None:
        T["x14"], T["y14"] = torch.from_numpy(x14).to(DEV), torch.from_numpy(y14).to(DEV)
    return T


def sub(T, idx):
    return {k: v[idx] for k, v in T.items()}


def loss_fn(rep, net, B, pos_weight):
    out = net(build_input(rep, B))
    tgt, cur = target_cls(rep, B).long(), cur_cls(rep, B).long()
    w = 1.0 + 3.0 * (tgt != cur).float()  # changed cells carry the learning signal
    loss = (F.cross_entropy(out["cls"], tgt, reduction="none") * w).mean()
    loss = loss + F.binary_cross_entropy_with_logits(out["over"], B["over"].float(), pos_weight=pos_weight)
    ok = (B["eaten"] == 1) & (B["food"] < 144)
    if ok.any():
        loss = loss + F.cross_entropy(out["food"][ok], B["food"][ok].long())
    if rep == "R1old":
        loss = loss + 20 * F.l1_loss(out["ord"], B["n_order"].float() / ORD_SCALE) + F.cross_entropy(out["dir"], B["n_dir"].long())
    return loss


def sampling_weights(lengths):
    bucket = np.digitize(lengths, [4, 8, 16])
    cnt = np.bincount(bucket, minlength=4).astype(float)
    w = 1.0 / np.sqrt(cnt[bucket])
    return w / w.sum()


@torch.no_grad()
def predict_batches(rep, net, T, n, bs=1024):
    net.eval()
    outs = {"cls": [], "over": [], "food": [], "ord": [], "dir": []}
    for i in range(0, n, bs):
        B = sub(T, slice(i, min(i + bs, n)))
        o = net(build_input(rep, B))
        outs["cls"].append(o["cls"].argmax(1).cpu())
        outs["over"].append((o["over"] > 0).cpu())
        outs["food"].append(F.softmax(o["food"], 1).cpu())
        outs["ord"].append(torch.clamp(torch.round(o["ord"] * ORD_SCALE), 0, 144).cpu())
        outs["dir"].append(o["dir"].argmax(1).cpu())
    net.train()
    return {k: torch.cat(v) for k, v in outs.items()}


def exact_flags(rep, P, T, n):
    tgt = target_cls(rep, T)[:n].cpu()
    grid = (P["cls"] == tgt).reshape(n, -1).all(1)
    if rep == "R1v2":
        t4 = torch.from_numpy(TO4)[P["cls"].long()]
        type_ok = (t4 == T["n_cls"][:n].cpu()).reshape(n, -1).all(1)
        return grid, type_ok
    if rep == "R1old":
        body = (T["n_cls"][:n].cpu() >= 2)
        full = grid & (((P["ord"] == T["n_order"][:n].cpu().float()) | ~body).reshape(n, -1).all(1)) & (P["dir"] == T["n_dir"][:n].cpu().long())
        return grid, full
    return grid, grid


# ----------------------------------------------------------------------------- overfit
def overfit(rep, seconds):
    tr, _ = common.load_trainval()
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(tr["act"]), 1000, replace=False))
    d = {k: v[idx] for k, v in tr.items()}
    x14, y14 = common.make_14(d)
    T = to_device(d, x14, y14)
    torch.manual_seed(0)
    net = Net(cin_of(rep), ncls_of(rep)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0)
    pos_w = torch.tensor(float((d["over"] == 0).sum()) / max(float((d["over"] == 1).sum()), 1), device=DEV)
    t0, step, log = time.time(), 0, []
    while time.time() - t0 < seconds:
        for g in opt.param_groups:
            g["lr"] = 2e-3 * max(0.03, 1 - (time.time() - t0) / seconds)  # linear decay: exact fitting needs a small final lr
        b = sub(T, torch.from_numpy(np.random.randint(0, 1000, 256)).to(DEV))
        loss = loss_fn(rep, net, b, pos_w)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 100 == 0:
            P = predict_batches(rep, net, T, 1000)
            g, f = exact_flags(rep, P, T, 1000)
            log.append((step, round(time.time() - t0, 1), round(float(loss), 4), float(g.float().mean()), float(f.float().mean())))
            print(log[-1], flush=True)
            if f.float().mean() >= 0.99 and g.float().mean() >= 0.99:
                break
    res = {"rep": rep, "device": str(DEV), "steps": step, "seconds": round(time.time() - t0, 1), "final_grid_exact": log[-1][3],
           "final_second_metric": log[-1][4], "second_metric": {"R1v2": "type-collapsed exact", "R1old": "grid+order+dir exact"}[rep],
           "reached_99": bool(log[-1][3] >= 0.99 and log[-1][4] >= 0.99), "pos_weight": float(pos_w), "log": log}
    json.dump(res, open(os.path.join(HERE, f"overfit_{rep}_lrdecay.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}))


# ----------------------------------------------------------------------------- full run
def train(rep, T, lengths, seconds, pos_w):
    torch.manual_seed(0)
    net = Net(cin_of(rep), ncls_of(rep)).to(DEV)
    nparams = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    w = torch.from_numpy(sampling_weights(lengths)).float()
    t0, step = time.time(), 0
    while time.time() - t0 < seconds:
        frac = (time.time() - t0) / seconds
        for g in opt.param_groups:
            g["lr"] = 2e-3 * (1 - 0.9 * frac)
        idx = torch.multinomial(w, 256, replacement=True).to(DEV)
        loss = loss_fn(rep, net, sub(T, idx), pos_w)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 1000 == 0:
            print(f"{rep} step {step} t={time.time()-t0:.0f}s loss={loss.item():.4f}", flush=True)
    return net, {"params": nparams, "steps": step, "train_wall_s": round(time.time() - t0, 1), "device": str(DEV),
                 "pos_weight_game_over": float(pos_w), "final_loss": float(loss)}


def eval_val(rep, net, d, T, x14, y14):
    n = len(d["act"])
    P = predict_batches(rep, net, T, n)
    grid_ok, second = exact_flags(rep, P, T, n)
    length = d["length"]
    base4 = (d["cls"] == d["n_cls"]).reshape(n, -1).all(1)
    base14 = (x14 == y14).reshape(n, -1).all(1) if x14 is not None else None
    res = {"exact": {}}
    for name, lo, hi in BUCKETS + [("all", 0, 999)]:
        m = (length >= lo) & (length <= hi)
        e = {"n": int(m.sum()), "baseline_copy_prev_4class": float(base4[m].mean()), "model_exact": float(grid_ok.numpy()[m].mean())}
        if rep == "R1v2":
            e["baseline_copy_prev_14class"] = float(base14[m].mean())
            e["model_type_collapsed_exact(4class)"] = float(second.numpy()[m].mean())
        res["exact"][name] = e
    # per-cell error by true class (true next state)
    pred = P["cls"].numpy()
    if rep == "R1v2":
        true = y14
        groups = {"empty": true == 0, "food": true == 1, "head": (true >= 2) & (true <= 5), "body": (true >= 6) & (true <= 9),
                  "tail": true >= 10}
        wrong = pred != true
        wrong4 = TO4[pred] != d["n_cls"]
        res["per_cell_error"] = {k: {"cells": int(m.sum()), "pct_wrong_14class": round(100 * float(wrong[m].mean()), 3),
                                     "pct_wrong_type_only": round(100 * float(wrong4[m].mean()), 3)} for k, m in groups.items()}
    else:
        true = d["n_cls"]
        tail = d["n_order"] == 1
        groups = {"empty": true == 0, "food": true == 1, "head": true == 2, "body": (true == 3) & ~tail, "tail": tail & (true == 3)}
        wrong = pred != true
        res["per_cell_error"] = {k: {"cells": int(m.sum()), "pct_wrong": round(100 * float(wrong[m].mean()), 3)} for k, m in groups.items()}
    over, pov = d["over"].astype(bool), P["over"].numpy()
    tp, fp, fn = (pov & over).sum(), (pov & ~over).sum(), (~pov & over).sum()
    res["game_over"] = {"n_true": int(over.sum()), "predicted": int(pov.sum()), "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1))}
    em = np.where(d["eaten"] == 1)[0]
    if len(em):
        s = torch.multinomial(P["food"][em], 1).squeeze(1).numpy()
        res["food_spawn_on_empty_pct"] = round(100 * float(np.mean([d["n_cls"][i].reshape(-1)[j] == 0 for i, j in zip(em, s)])), 2)
        res["n_eat_steps"] = int(len(em))
    return res


@torch.no_grad()
def one_pred(rep, net, grid_in, prev, order, direc, a):
    """Single-state prediction; returns (argmax grid, over, food_probs)."""
    x = {"act": torch.tensor([a], device=DEV)}
    if rep == "R1v2":
        x["x14"] = torch.from_numpy(grid_in[None]).to(DEV)
    else:
        x["cls"] = torch.from_numpy(grid_in[None]).to(DEV)
        x["prev_cls"] = torch.from_numpy(prev[None]).to(DEV)
    net.eval()
    o = net(build_input(rep, x))
    return o["cls"].argmax(1)[0].cpu().numpy().astype(np.uint8), bool(o["over"][0] > 0), F.softmax(o["food"], 1).cpu()


def rollouts(rep, net, d, x14, kind, n=50, seed=0):
    rng = random.Random(seed)
    L, over = d["length"], d["over"]
    pools = [[i for i in np.where((L >= lo) & (L <= hi) & (over == 0))[0]] for _, lo, hi in BUCKETS]
    starts = [rng.choice(pools[k % 4] or pools[0]) for k in range(n)]
    fm, fm_type, fi, matched, reasons, fails = [], [], [], 0, [], []
    for k, ti in enumerate(starts):
        eng = common.engine_from_state(d["cls"][ti], d["order"][ti], d["dir"][ti])
        m_grid = x14[ti].copy() if rep == "R1v2" else d["cls"][ti].copy()
        m_prev = d["prev_cls"][ti].copy()
        m_len = len(eng.body)
        arng = random.Random(seed * 1000 + k)
        mis = mis_t = inval = None
        ended = False
        for t in range(1, HORIZON + 1):
            a = pick("bfs", eng, arng) if kind == "bfs" else (arng.choice(safe_moves(eng) or [0, 1, 2, 3, 4]))
            eaten = eng.step(a)
            food_e = None if (eaten or eng.food is None) else eng.food
            e14 = common.grid14(eng.body, eng.direction, food_e)
            e4 = common.cls4_grid(eng)
            if eaten and eng.food is not None:
                e4[eng.food] = 0
            det, pov, fp = one_pred(rep, net, m_grid, m_prev, None, None, int(a))
            det4 = common.TO4[det] if rep == "R1v2" else det
            food_cell = int(torch.multinomial(fp, 1)[0]) if eaten else None
            m_len += 1 if eaten else 0
            if not (eng.game_over and not eaten):
                r = common.valid_state(det4, food_cell, m_len)
                if r and inval is None:
                    inval = t
                    reasons.append(r)
            strict_bad = (not np.array_equal(det, e14 if rep == "R1v2" else e4)) or pov != eng.game_over
            type_bad = (not np.array_equal(det4, e4)) or pov != eng.game_over
            if mis is None and strict_bad:
                mis = t
                if len(fails) < 4:
                    fails.append({"rollout": k, "kind": kind, "step": t, "action": int(a),
                                  "ascii": common.ascii_pair14(e14, det) if rep == "R1v2" else ascii_pair4(e4, det)})
            if mis_t is None and type_bad:
                mis_t = t
            if eng.game_over:
                ended = True
                break
            nxt = det.copy()
            if eaten and eng.food is not None:
                nxt[eng.food] = 1
            m_prev, m_grid = (m_grid if rep != "R1v2" else m_prev), nxt
            if rep != "R1v2":
                m_prev = m_prev  # R2: previous model frame
        if mis is None and ended:
            matched += 1
        fm.append(mis or HORIZON + 1); fm_type.append(mis_t or HORIZON + 1); fi.append(inval or HORIZON + 1)
    a = lambda v, q: float(np.percentile(v, q))
    return {"kind": kind, "n": n, "first_mismatch_median": a(fm, 50), "first_mismatch_p10": a(fm, 10),
            "first_type_mismatch_median": a(fm_type, 50), "first_type_mismatch_p10": a(fm_type, 10),
            "no_mismatch_within_20": int(sum(v == HORIZON + 1 for v in fm)), "matched_to_end": matched,
            "first_invalid_median": a(fi, 50), "first_invalid_p10": a(fi, 10),
            "no_invalid_within_20": int(sum(v == HORIZON + 1 for v in fi)),
            "invalid_reasons": {r: reasons.count(r) for r in set(reasons)},
            "note": "21 = censored (no event in 20 steps)"}, fails


def ascii_pair4(eng, mod):
    ch = ".Fho"
    rows = ["".join(ch[v] for v in eng[r]) + "   " + "".join(ch[v] for v in mod[r]) for r in range(N)]
    return "engine (h head, o body, F food)   model\n" + "\n".join(rows)


def full(seconds):
    tr, va = common.load_trainval()
    x14t, y14t = common.make_14(tr)
    x14v, y14v = common.make_14(va)
    print(f"train {len(tr['act'])} val {len(va['act'])}", flush=True)
    Tt, Tv = to_device(tr, x14t, y14t), to_device(va, x14v, y14v)
    pos_w = torch.tensor(float((tr["over"] == 0).sum()) / float((tr["over"] == 1).sum()), device=DEV)
    results = {"seconds_per_model": seconds, "device": str(DEV), "n_train": int(len(tr["act"])), "n_val": int(len(va["act"])),
               "pos_weight_game_over": float(pos_w)}
    results["r2_information_limit_val"] = common.info_limit_tail(va)
    fails_all = {}
    for rep in ["R1v2", "R2"]:
        net, info = train(rep, Tt, tr["length"], seconds, pos_w)
        r = {"train": info, "val": eval_val(rep, net, va, Tv, x14v, y14v)}
        r["rollout_bfs"], f1 = rollouts(rep, net, va, x14v, "bfs", seed=1)
        r["rollout_random_safe"], f2 = rollouts(rep, net, va, x14v, "random", seed=2)
        fails_all[rep] = (f1 + f2)[:2]
        results[rep] = r
        print(rep, json.dumps(r["val"]["exact"]["all"]), flush=True)
    json.dump(results, open(os.path.join(HERE, "results_v2.json"), "w"), indent=1)
    with open(os.path.join(HERE, "failures_v2.md"), "w") as f:
        for rep, fl in fails_all.items():
            for x in fl:
                f.write(f"### {rep} {x['kind']} rollout {x['rollout']} step {x['step']} action {x['action']} (UP0 DOWN1 LEFT2 RIGHT3 NOOP4)\n```\n{x['ascii']}\n```\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "overfit":
        overfit(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 180.0)
    else:
        full(float(sys.argv[2]) if len(sys.argv) > 2 else 300.0)
