"""Spike A debug 2: fast on-device training loop, auxiliary "ate" head, eat/game-over oversampling. R1v2 only.
Train/validation only (held-out seeds are never loaded; common.load_trainval drops them).

Usage (always after `source scripts/env.sh`):
  .venv/bin/python spikes/spike_a/v3.py bench
  .venv/bin/python spikes/spike_a/v3.py overfit <arch> <device> [seconds]
  .venv/bin/python spikes/spike_a/v3.py full <arch> <device> [seconds]
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import common  # noqa: E402
from common import N, TO4  # noqa: E402
import v2  # noqa: E402  (reused: rollouts, BUCKETS, sampling_weights)

BUCKETS = v2.BUCKETS
FRAC_EAT, FRAC_OVER = 0.2, 0.2
CLASS_NAMES = ["empty", "food", "head", "body", "tail"]


def dev_of(name):
    return torch.device(name)


# ----------------------------------------------------------------------------- model
class Net3(nn.Module):
    """R1v2 net. arch: dil = stem + 7 dilated 3x3 (RF 33); plain11 = stem + 11 plain 3x3 (RF 25); k5 = stem + 5 plain 5x5 (RF 23).
    All cover the 12x12 grid from any cell (needs RF >= 23). Two global-pool broadcasts as in v2.
    ate head: global pool after conv index 1 -> MLP -> logit; sigmoid(p) is broadcast as an extra channel, injected through a 1x1 conv
    after conv 1 and concatenated into the cls/food/over heads. It is part of the graph (exportable to ONNX), not logic outside it."""
    SPECS = {"dil": [(3, d) for d in (1, 2, 4, 1, 2, 4, 1)], "plain11": [(3, 1)] * 11, "k5": [(5, 1)] * 5}

    def __init__(self, arch="dil", C=32, ate=True, cin=19, ncls=14, food_detach=False):
        super().__init__()
        self.ate, self.arch, self.food_detach = ate, arch, food_detach  # food_detach: food head reads stop-gradient trunk features
        self.stem = nn.Conv2d(cin, C, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(C, C, k, padding=d * (k // 2), dilation=d) for k, d in self.SPECS[arch]])
        self.g2_at = len(self.convs) - 3
        self.g1 = nn.Linear(2 * C, C)
        self.g2 = nn.Linear(2 * C, C)
        e = 1 if ate else 0
        if ate:
            self.ate_mlp = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU(), nn.Linear(C, 1))
            self.inj = nn.Conv2d(C + 1, C, 1)
        self.cls_head = nn.Conv2d(C + e, ncls, 1)
        self.food_head = nn.Conv2d(C + e, 1, 1)
        self.over_head = nn.Linear(2 * C + e, 1)
        # prior-shift corrections (logit units) applied in eval mode only, set after training from TRAIN-set base rates
        self.register_buffer("over_shift", torch.zeros(()))
        self.register_buffer("ate_shift", torch.zeros(()))

    @staticmethod
    def pool(h):
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        p = a_logit = None
        for i, conv in enumerate(self.convs):
            h = h + F.relu(conv(h))
            if i == 1:
                g = self.pool(h)
                h = h + F.relu(self.g1(g))[:, :, None, None]
                if self.ate:
                    a_logit = self.ate_mlp(g).squeeze(1)
                    p = torch.sigmoid(a_logit)[:, None, None, None].expand(-1, 1, N, N)
                    h = h + F.relu(self.inj(torch.cat([h, p], 1)))
            if i == self.g2_at:
                h = h + F.relu(self.g2(self.pool(h)))[:, :, None, None]
        g = self.pool(h)
        hh = torch.cat([h, p], 1) if self.ate else h
        gg = torch.cat([g, p[:, 0, 0, :1].reshape(-1, 1)], 1) if self.ate else g
        over = self.over_head(gg).squeeze(1)
        fh = hh.detach() if self.food_detach else hh
        out = {"cls": self.cls_head(hh), "food": self.food_head(fh).flatten(1), "over": over}
        if self.ate:
            out["ate"] = a_logit
        if not self.training:
            out["over"] = over - self.over_shift
            if self.ate:
                out["ate"] = a_logit - self.ate_shift
        return out


# ----------------------------------------------------------------------------- data on device
def make_device_data(d, x14, y14, dev):
    """Everything the training loop needs, resident on `dev` as uint8 (one-hot is built on device)."""
    u8 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(torch.uint8).to(dev)
    return {"x14": u8(x14), "y14": u8(y14), "act": u8(d["act"]), "over": u8(d["over"]), "eaten": u8(d["eaten"]),
            "food": u8(d["food"]), "n_cls": u8(d["n_cls"]), "length": u8(np.minimum(d["length"], 255))}


def build_x(Tb):
    B = Tb["act"].shape[0]
    oh = F.one_hot(Tb["x14"].long(), 14).permute(0, 3, 1, 2).float()
    a = F.one_hot(Tb["act"].long(), 5).float().view(B, 5, 1, 1).expand(B, 5, N, N)
    return torch.cat([oh, a], 1).contiguous(memory_format=torch.channels_last)


class Sampler:
    """20% eat transitions, 20% game-over transitions, 60% natural-with-length-bucket-weights. All sampling on device, no sync."""

    def __init__(self, d, dev, B):
        self.B, self.dev = B, dev
        self.nE, self.nO = int(round(FRAC_EAT * B)), int(round(FRAC_OVER * B))
        self.nN = B - self.nE - self.nO
        self.eat = torch.from_numpy(np.where(d["eaten"] == 1)[0]).to(dev)
        self.over = torch.from_numpy(np.where(d["over"] == 1)[0]).to(dev)
        self.w = torch.from_numpy(v2.sampling_weights(d["length"])).float().to(dev)
        self.n_eat, self.n_over = len(self.eat), len(self.over)

    def idx(self):
        d = self.dev
        e = self.eat[torch.randint(0, self.n_eat, (self.nE,), device=d)]
        o = self.over[torch.randint(0, self.n_over, (self.nO,), device=d)]
        n = torch.multinomial(self.w, self.nN, replacement=True)
        return torch.cat([e, o, n])


def loss_fn(net, Tb):
    out = net(build_x(Tb))
    tgt, cur = Tb["y14"].long(), Tb["x14"].long()
    w = 1.0 + 3.0 * (tgt != cur).float()
    loss = (F.cross_entropy(out["cls"], tgt, reduction="none") * w).mean()
    loss = loss + F.binary_cross_entropy_with_logits(out["over"], Tb["over"].float())
    if "ate" in out:
        loss = loss + F.binary_cross_entropy_with_logits(out["ate"], Tb["eaten"].float())
    # food-spawn CE on eat rows, masked instead of boolean-indexed (no host sync); food==255 means no spawn
    valid = ((Tb["eaten"] == 1) & (Tb["food"] < 144)).float()
    ce = F.cross_entropy(out["food"], Tb["food"].long().clamp(max=143), reduction="none")
    loss = loss + (ce * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


def prior_shifts(net, S, T, d, dev, n_batches=100):
    """Eval-time logit shift = logit(batch base rate under the oversampling) - logit(natural TRAIN base rate)."""
    idx = torch.cat([S.idx() for _ in range(n_batches)])
    lg = lambda p: math.log(p / (1 - p))
    shifts = {}
    for k, key in (("over_shift", "over"), ("ate_shift", "eaten")):
        pb = float(T[key][idx].float().mean())
        pn = float(np.mean(d[key]))
        shifts[k] = lg(pb) - lg(pn)
        shifts[k + "_info"] = {"batch_rate": round(pb, 4), "natural_train_rate": round(pn, 4)}
        if k == "ate_shift" and not net.ate:
            continue
        getattr(net, k).fill_(shifts[k])
    return shifts


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def make_opt(net, wd):
    try:
        return torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd, fused=True)
    except Exception:
        return torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(net, T, n, bs=1024):
    net.eval()
    o = {"cls": [], "over": [], "food": [], "ate": []}
    for i in range(0, n, bs):
        Tb = {k: v[i:i + bs] for k, v in T.items()}
        out = net(build_x(Tb))
        o["cls"].append(out["cls"].argmax(1).cpu())
        o["over"].append((out["over"] > 0).cpu())
        o["food"].append(F.softmax(out["food"], 1).cpu())
        o["ate"].append((out["ate"] > 0).cpu() if "ate" in out else torch.zeros(len(Tb["act"]), dtype=torch.bool))
    net.train()
    return {k: torch.cat(v) for k, v in o.items()}


def eval_val(net, d, T, x14, y14):
    n = len(d["act"])
    P = predict(net, T, n)
    pred = P["cls"].numpy()
    ok = (pred == y14).reshape(n, -1).all(1)
    pov, over = P["over"].numpy(), d["over"].astype(bool)
    ok_over = ok & (pov == over)
    eat = d["eaten"].astype(bool)
    length = d["length"]
    base = (x14 == y14).reshape(n, -1).all(1)
    base4 = (d["cls"] == d["n_cls"]).reshape(n, -1).all(1)
    res = {"exact": {}}
    for name, lo, hi in BUCKETS + [("all", 0, 999)]:
        m0 = (length >= lo) & (length <= hi)
        row = {}
        for sname, m in (("all_steps", m0), ("eat_steps", m0 & eat), ("non_eat_steps", m0 & ~eat)):
            row[sname] = {"n": int(m.sum()), "model_exact": round(float(ok[m].mean()), 4) if m.any() else None,
                          "model_exact_and_over_flag": round(float(ok_over[m].mean()), 4) if m.any() else None,
                          "copy14": round(float(base[m].mean()), 4) if m.any() else None,
                          "copy4": round(float(base4[m].mean()), 4) if m.any() else None,
                          "model_type_collapsed_exact": round(float((TO4[pred] == d["n_cls"]).reshape(n, -1).all(1)[m].mean()), 4) if m.any() else None}
        res["exact"][name] = row
    true = y14
    groups = {"empty": true == 0, "food": true == 1, "head": (true >= 2) & (true <= 5), "body": (true >= 6) & (true <= 9), "tail": true >= 10}
    wrong = pred != true
    wrong4 = TO4[pred] != d["n_cls"]
    res["per_cell_error"] = {}
    for gname, g in groups.items():
        e = {"cells": int(g.sum()), "pct_wrong_14class": round(100 * float(wrong[g].mean()), 3),
             "pct_wrong_type_only": round(100 * float(wrong4[g].mean()), 3)}
        for sname, rows in (("eat", eat), ("non_eat", ~eat)):
            gm = g & rows[:, None, None]
            e[f"pct_wrong_14class_{sname}_steps"] = round(100 * float(wrong[gm].mean()), 3) if gm.any() else None
        res["per_cell_error"][gname] = e
    tp, fp, fn = (pov & over).sum(), (pov & ~over).sum(), (~pov & over).sum()
    res["game_over"] = {"n_true": int(over.sum()), "predicted": int(pov.sum()), "precision": round(float(tp / max(tp + fp, 1)), 4),
                        "recall": round(float(tp / max(tp + fn, 1)), 4)}
    pat = P["ate"].numpy()
    if net.ate:
        tpa, fpa, fna = (pat & eat).sum(), (pat & ~eat).sum(), (~pat & eat).sum()
        res["ate_head"] = {"accuracy": round(float((pat == eat).mean()), 4), "n_eat": int(eat.sum()),
                           "precision": round(float(tpa / max(tpa + fpa, 1)), 4), "recall": round(float(tpa / max(tpa + fna, 1)), 4),
                           "accuracy_always_no": round(float((~eat).mean()), 4)}
    em = np.where(d["eaten"] == 1)[0]
    s = torch.multinomial(P["food"][em], 1).squeeze(1).numpy()
    res["food_spawn_on_empty_pct"] = round(100 * float(np.mean([d["n_cls"][i].reshape(-1)[j] == 0 for i, j in zip(em, s)])), 2)
    res["n_eat_steps"] = int(len(em))
    return res


# ----------------------------------------------------------------------------- training
def run_training(net, T, S, seconds, dev, wd, log_every_s=None, eval_cb=None, lr0=2e-3, cosine=False):
    opt = make_opt(net, wd)
    t0, step, curve, last = time.time(), 0, [], time.time()
    while True:
        el = time.time() - t0
        if el >= seconds:
            break
        set_lr(opt, lr0 * 0.5 * (1 + math.cos(math.pi * el / seconds)) if cosine else lr0 * max(0.03, 1 - el / seconds))
        idx = S.idx()
        Tb = {k: v[idx] for k, v in T.items()}
        loss = loss_fn(net, Tb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if log_every_s and time.time() - last >= log_every_s:
            sync(dev)
            last = time.time()
            if eval_cb(step, time.time() - t0, loss.item()):
                break
    sync(dev)
    return step, time.time() - t0, float(loss)


def load_data(dev):
    tr, va = common.load_trainval()
    x14t, y14t = common.make_14(tr)
    x14v, y14v = common.make_14(va)
    return tr, va, x14t, y14t, x14v, y14v


# ----------------------------------------------------------------------------- bench
def bench():
    tr, va, x14t, y14t, *_ = load_data(None)
    out = {"command": "v3.py bench", "rows": []}
    # --- old loop (v2) reference, measured the same way
    for devname in ("mps",):
        dev = dev_of(devname)
        v2.DEV = dev
        T2 = v2.to_device(tr, x14t, y14t)
        w = torch.from_numpy(v2.sampling_weights(tr["length"])).float()
        pos_w = torch.tensor(73.3, device=dev)
        net = v2.Net(19, 14).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        parts = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        for it in range(60):
            sync(dev); t = time.time()
            idx = torch.multinomial(w, 256, replacement=True).to(dev); b = v2.sub(T2, idx)
            sync(dev); t1 = time.time()
            loss = v2.loss_fn("R1v2", net, b, pos_w)  # includes build_input (one-hot) + boolean-index on `ok.any()` (host sync)
            sync(dev); t2 = time.time()
            opt.zero_grad(); loss.backward()
            sync(dev); t3 = time.time()
            opt.step()
            sync(dev); t4 = time.time()
            if it >= 10:
                for k, a, b_ in (("data", t, t1), ("forward", t1, t2), ("backward", t2, t3), ("optimizer", t3, t4)):
                    parts[k] += b_ - a
        n = 50
        sync(dev); t0 = time.time()
        for it in range(100):
            idx = torch.multinomial(w, 256, replacement=True).to(dev)
            loss = v2.loss_fn("R1v2", net, v2.sub(T2, idx), pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        sync(dev); e2e = 100 * 256 / (time.time() - t0)
        out["rows"].append({"name": "OLD v2 loop (dil, no ate, CPU sampling, host sync in loss)", "device": devname, "batch": 256,
                            "ms": {k: round(1000 * v / n, 2) for k, v in parts.items()}, "samples_per_s_e2e": round(e2e)})
        print(out["rows"][-1], flush=True)
        del T2, net, opt
    # --- new loop grid
    results_warn = {}
    for devname in ("mps", "cpu"):
        dev = dev_of(devname)
        T = make_device_data(tr, x14t, y14t, dev)
        for arch in ("dil", "plain11", "k5"):
            for B in (256, 1024):
                torch.manual_seed(0)
                S = Sampler(tr, dev, B)
                net = Net3(arch, ate=True).to(dev)
                opt = make_opt(net, 1e-4)
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always")
                    nwarm, ntime, ne2e = (5, 12, 15) if devname == "cpu" else (10, 30, 60)
                    parts = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
                    for it in range(nwarm + ntime):
                        sync(dev); t = time.time()
                        idx = S.idx(); Tb = {k: v[idx] for k, v in T.items()}; x = build_x(Tb)
                        sync(dev); t1 = time.time()
                        o = net(x)
                        loss = F.cross_entropy(o["cls"], Tb["y14"].long()) + F.binary_cross_entropy_with_logits(o["over"], Tb["over"].float()) \
                            + F.binary_cross_entropy_with_logits(o["ate"], Tb["eaten"].float())
                        sync(dev); t2 = time.time()
                        opt.zero_grad(set_to_none=True); loss.backward()
                        sync(dev); t3 = time.time()
                        opt.step()
                        sync(dev); t4 = time.time()
                        if it >= nwarm:
                            for k, a, b_ in (("data", t, t1), ("forward", t1, t2), ("backward", t2, t3), ("optimizer", t3, t4)):
                                parts[k] += b_ - a
                    sync(dev); t0 = time.time()
                    for it in range(ne2e):
                        idx = S.idx()
                        loss = loss_fn(net, {k: v[idx] for k, v in T.items()})
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    sync(dev)
                    e2e = ne2e * B / (time.time() - t0)
                ws = sorted({str(w_.message)[:160] for w_ in wl})
                row = {"name": f"NEW {arch}", "device": devname, "batch": B, "params": sum(p.numel() for p in net.parameters()),
                       "ms": {k: round(1000 * v / ntime, 2) for k, v in parts.items()}, "samples_per_s_e2e": round(e2e), "warnings": ws}
                out["rows"].append(row)
                print(row, flush=True)
                del net, opt
        del T
    json.dump(out, open(os.path.join(HERE, "speed_v3.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- overfit
def overfit(arch, devname, seconds):
    dev = dev_of(devname)
    v2.DEV = dev
    tr, *_ = load_data(dev)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(tr["act"]), 1000, replace=False))
    d = {k: v[idx] for k, v in tr.items()}
    x14, y14 = common.make_14(d)
    T = make_device_data(d, x14, y14, dev)
    torch.manual_seed(0)
    S = Sampler(d, dev, 256)
    net = Net3(arch, ate=True).to(dev)
    curve = []

    def cb(step, t, loss):
        P = predict(net, T, 1000)
        ok = float((P["cls"].numpy() == y14).reshape(1000, -1).all(1).mean())
        ate = float((P["ate"].numpy() == d["eaten"].astype(bool)).mean())
        curve.append({"t_s": round(t, 1), "step": step, "samples_per_s": round(step * 256 / t), "loss": round(loss, 4),
                      "exact": round(ok, 4), "ate_acc": round(ate, 4)})
        print(curve[-1], flush=True)
        return ok >= 0.99
    step, el, loss = run_training(net, T, S, seconds, dev, 0.0, log_every_s=30.0, eval_cb=cb)
    P = predict(net, T, 1000)
    pred = P["cls"].numpy()
    ok = (pred == y14).reshape(1000, -1).all(1)
    res = {"command": f"v3.py overfit {arch} {devname} {seconds}", "arch": arch, "device": devname, "rows": 1000, "steps": step,
           "seconds": round(el, 1), "samples_per_s": round(step * 256 / el), "final_exact": round(float(ok.mean()), 4),
           "reached_99": bool(ok.mean() >= 0.99), "curve": curve, "n_eat_rows": int(d["eaten"].sum()), "n_over_rows": int(d["over"].sum())}
    if not res["reached_99"]:
        eat = d["eaten"].astype(bool)
        wrong = pred != y14
        res["exact_eat_steps"] = round(float(ok[eat].mean()), 4)
        res["exact_non_eat_steps"] = round(float(ok[~eat].mean()), 4)
        groups = {"empty": y14 == 0, "food": y14 == 1, "head": (y14 >= 2) & (y14 <= 5), "body": (y14 >= 6) & (y14 <= 9), "tail": y14 >= 10}
        res["per_class_error_pct"] = {g: {"eat_steps": round(100 * float(wrong[m & eat[:, None, None]].mean()), 3) if (m & eat[:, None, None]).any() else None,
                                           "non_eat_steps": round(100 * float(wrong[m & ~eat[:, None, None]].mean()), 3)} for g, m in groups.items()}
        res["per_14class_error_pct"] = {int(c): round(100 * float(wrong[y14 == c].mean()), 2) for c in range(14) if (y14 == c).any()}
    json.dump(res, open(os.path.join(HERE, f"overfit_v3_{arch}_{devname}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


# ----------------------------------------------------------------------------- full
def full(arch, devname, seconds):
    dev = dev_of(devname)
    v2.DEV = dev
    tr, va, x14t, y14t, x14v, y14v = load_data(dev)
    Tt, Tv = make_device_data(tr, x14t, y14t, dev), make_device_data(va, x14v, y14v, dev)
    torch.manual_seed(0)
    S = Sampler(tr, dev, 256)
    net = Net3(arch, ate=True).to(dev).to(memory_format=torch.channels_last)
    shifts = prior_shifts(net, S, Tt, tr, dev)
    logs = []

    def cb(step, t, loss):
        print(f"step {step} t={t:.0f}s loss={loss:.4f} samples/s={step*256/t:.0f}", flush=True)
        logs.append((step, round(t, 1), round(loss, 4)))
        return False
    step, el, loss = run_training(net, Tt, S, seconds, dev, 1e-4, log_every_s=60.0, eval_cb=cb)
    res = {"command": f"v3.py full {arch} {devname} {seconds}", "arch": arch, "device": devname, "params": sum(p.numel() for p in net.parameters()),
           "steps": step, "seconds": round(el, 1), "samples_per_s": round(step * 256 / el), "final_loss": round(loss, 4),
           "prior_shifts": shifts, "n_train": int(len(tr["act"])), "n_val": int(len(va["act"])),
           "train_eat_rate": round(float(tr["eaten"].mean()), 4), "train_over_rate": round(float(tr["over"].mean()), 4),
           "val_eat_rate": round(float(va["eaten"].mean()), 4), "val_over_rate": round(float(va["over"].mean()), 4)}
    res["val"] = eval_val(net, va, Tv, x14v, y14v)
    res["rollout_bfs"], f1 = v2.rollouts("R1v2", net, va, x14v, "bfs", seed=1)
    res["rollout_random_safe"], f2 = v2.rollouts("R1v2", net, va, x14v, "random", seed=2)
    ex = res["val"]["exact"]
    res["implied_median_from_p"] = {}
    for name, p in (("p=overall_exact", ex["all"]["all_steps"]["model_exact"]),
                    ("p=non_eat_exact", ex["all"]["non_eat_steps"]["model_exact"]),
                    ("p=bucket_balanced_exact(rollout start mix)", float(np.mean([ex[b]["all_steps"]["model_exact"] for b, *_ in BUCKETS])))):
        res["implied_median_from_p"][name] = {"p": p, "median_steps": round(math.log(0.5) / math.log(p), 3) if 0 < p < 1 else None}
    json.dump(res, open(os.path.join(HERE, "results_v3.json"), "w"), indent=1)
    with open(os.path.join(HERE, "failures_v3.md"), "w") as f:
        for x in (f1[:1] + f2[:1]):
            f.write(f"### R1v2 {x['kind']} rollout {x['rollout']} step {x['step']} action {x['action']} (UP0 DOWN1 LEFT2 RIGHT3 NOOP4)\n```\n{x['ascii']}\n```\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bench":
        bench()
    elif cmd == "overfit":
        overfit(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 180.0)
    else:
        full(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 600.0)
