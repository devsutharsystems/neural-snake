"""Final evaluation of the SHIPPED model (web/model/snake.onnx, onnxruntime-python, CPU) following PLAN.md section 13.

  .venv/bin/python scripts/eval_heldout.py dryrun                      # VALIDATION rows only (seed % 5 == 1) of data/run1 and data/spike_a
  .venv/bin/python scripts/eval_heldout.py heldout --confirm-heldout   # held-out rows (seed % 5 == 0); refuses to run twice (eval/heldout.lock)

Outputs: eval/dryrun_<dataset>_validation.json | eval/results_heldout.json + eval/results_heldout.md
The held-out code path is unreachable without the sub-command `heldout`, the flag --confirm-heldout and the absence of eval/heldout.lock;
the lock is created (O_EXCL) BEFORE any held-out row is read. Headline = data/run1 held-out; data/spike_a held-out is reported separately
and labelled "seen once in Phase 0 by throwaway models".
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spikes", "spike_a"))
import common  # noqa: E402
import gen  # noqa: E402
from common import TO4  # noqa: E402

ONNX = os.path.join(ROOT, "web", "model", "snake.onnx")
CKPT = os.path.join(ROOT, "checkpoints", "run2_best.pt")
OUT = os.path.join(ROOT, "eval")
LOCK = os.path.join(OUT, "heldout.lock")
BUCKETS = [("1-3", 1, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 999)]
EXTRA_BUCKETS = [("16-31", 16, 31), ("32-63", 32, 63), ("64+", 64, 999)]
ROLLOUT_CAP, ROLLOUTS_PER_KIND = 1000, 100
SPLIT_MOD = {"validation": 1, "heldout": 0}
DYNAMICS_REASONS = ("heads!=1", "wrong_length", "disconnected")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    return {"sha256": {"web/model/snake.onnx": sha256(ONNX), "checkpoints/run2_best.pt": sha256(CKPT),
                       "scripts/eval_heldout.py": sha256(os.path.abspath(__file__))},
            "onnxruntime_py": ort.__version__, "git_head": head, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def load_split(dataset, split):
    """All arrays for rows with episode seed % 5 == SPLIT_MOD[split]. The caller is responsible for the held-out gate."""
    d = os.path.join(ROOT, "data", dataset)
    ep = np.load(os.path.join(d, "ep.npy"))
    mask = (ep % 5) == SPLIT_MOD[split]
    out = {k: np.ascontiguousarray(np.load(os.path.join(d, k + ".npy"), mmap_mode="r")[mask]) for k in common.KEYS}
    assert set((out["ep"] % 5).tolist()) == {SPLIT_MOD[split]}
    return out


def make_session():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(ONNX, sess_options=so, providers=["CPUExecutionProvider"])


def run_model(sess, grid, action):
    """grid: int array [12,12] (14-class); returns (argmax grid int [12,12], food probs [144], p_ate, p_over)."""
    g, f, a, o = sess.run(None, {"grid": grid.astype(np.int32)[None], "action": np.array([action], np.int32)})
    return g[0], f[0], float(a[0]), float(o[0])


# ----------------------------------------------------------------------------- one-step metrics
def one_step(sess, d, x14, y14, seed=0):
    n = len(d["act"])
    pred = np.zeros((n, 12, 12), np.uint8)
    p_ate = np.zeros(n)
    p_over = np.zeros(n)
    food_p = np.zeros((n, 144), np.float32)
    keep_food = d["eaten"] == 1
    for i in range(n):
        g, f, a, o = run_model(sess, x14[i], int(d["act"][i]))
        pred[i], p_ate[i], p_over[i] = g, a, o
        if keep_food[i]:
            food_p[i] = f
    ok = (pred == y14).reshape(n, -1).all(1)
    ok_over = ok & ((p_over > 0.5) == d["over"].astype(bool))
    eat = d["eaten"].astype(bool)
    over = d["over"].astype(bool)
    length = d["length"]
    copy14 = (x14 == y14).reshape(n, -1).all(1)
    copy4 = (d["cls"] == d["n_cls"]).reshape(n, -1).all(1)
    ex = {}
    for name, lo, hi in BUCKETS + EXTRA_BUCKETS + [("all", 0, 999)]:
        m0 = (length >= lo) & (length <= hi)
        row = {}
        for sname, m in (("all_steps", m0), ("eat_steps", m0 & eat), ("non_eat_steps", m0 & ~eat)):
            row[sname] = {"n": int(m.sum()), "model_exact": round(float(ok[m].mean()), 5) if m.any() else None,
                          "model_exact_and_over_flag": round(float(ok_over[m].mean()), 5) if m.any() else None,
                          "copy14": round(float(copy14[m].mean()), 5) if m.any() else None,
                          "copy4": round(float(copy4[m].mean()), 5) if m.any() else None}
        ex[name] = row
    wrong = pred != y14
    wrong4 = TO4[pred] != d["n_cls"]
    groups = {"empty": y14 == 0, "food": y14 == 1, "head": (y14 >= 2) & (y14 <= 5), "body": (y14 >= 6) & (y14 <= 9), "tail": y14 >= 10}
    per_class = {}
    for gname, gm in groups.items():
        e = {"cells": int(gm.sum()), "pct_wrong_14class": round(100 * float(wrong[gm].mean()), 4), "pct_wrong_type_only": round(100 * float(wrong4[gm].mean()), 4)}
        for sname, rows in (("eat", eat), ("non_eat", ~eat)):
            gmr = gm & rows[:, None, None]
            e[f"pct_wrong_14class_{sname}_steps"] = round(100 * float(wrong[gmr].mean()), 4) if gmr.any() else None
        per_class[gname] = e
    pov = p_over > 0.5
    tp, fp, fn = int((pov & over).sum()), int((pov & ~over).sum()), int((~pov & over).sum())
    game_over = {"n_true": int(over.sum()), "predicted": int(pov.sum()), "precision": round(tp / max(tp + fp, 1), 5), "recall": round(tp / max(tp + fn, 1), 5)}
    pat = p_ate > 0.5
    tpa, fpa, fna = int((pat & eat).sum()), int((pat & ~eat).sum()), int((~pat & eat).sum())
    ate = {"accuracy": round(float((pat == eat).mean()), 5), "n_eat": int(eat.sum()), "precision": round(tpa / max(tpa + fpa, 1), 5),
           "recall": round(tpa / max(tpa + fna, 1), 5), "accuracy_if_always_no": round(float((~eat).mean()), 5)}
    # food head: cross-entropy on the true spawn cell vs the uniform-over-empty-cells floor (recomputed on THESE rows)
    em = np.where(eat & (d["food"] < 144))[0]
    p_true = food_p[em, d["food"][em].astype(int)].astype(np.float64)
    ce = float(-np.log(np.clip(p_true, 1e-30, None)).mean())
    empty = (y14[em] == 0).reshape(len(em), -1).sum(1)
    floor = float(np.log(empty).mean())
    snake_mass = float((food_p[em] * ((y14[em] >= 2).reshape(len(em), -1))).sum(1).mean())
    rng = np.random.default_rng(seed)
    cells = np.array([rng.choice(144, p=(p / p.sum())) for p in food_p[em].astype(np.float64)])
    valid = d["n_cls"][em].reshape(len(em), -1)[np.arange(len(em)), cells] == 0
    food = {"eat_rows_with_spawn": int(len(em)), "head_cross_entropy": round(ce, 4), "uniform_floor_mean_ln_empty": round(floor, 4),
            "ce_minus_floor": round(ce - floor, 4), "mean_prob_mass_on_snake_cells": round(snake_mass, 5),
            "spawn_valid_pct_sampled_on_empty_cell": round(100 * float(valid.mean()), 3), "spawn_on_snake_pct": round(100 * float(1 - valid.mean()), 3)}
    overall = ex["all"]["all_steps"]["model_exact"]
    bal = float(np.mean([ex[b]["all_steps"]["model_exact"] for b, *_ in BUCKETS if ex[b]["all_steps"]["model_exact"] is not None]))
    implied = {k: {"p": round(p, 5), "median_steps": round(math.log(0.5) / math.log(p), 2) if 0 < p < 1 else None}
               for k, p in (("overall_exact", overall), ("non_eat_exact", ex["all"]["non_eat_steps"]["model_exact"]), ("bucket_balanced_exact", bal))}
    return {"rows": int(n), "exact": ex, "per_class_error": per_class, "game_over": game_over, "ate_head": ate, "food_head": food,
            "implied_median_from_one_step_p": implied}


# ----------------------------------------------------------------------------- rollouts
def ascii_grids(e14, det):
    return common.ascii_pair14(e14.astype(np.uint8), det.astype(np.uint8))


def rollouts(sess, d, x14, kind, n, cap, seed):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    L, over = d["length"], d["over"]
    pools = [np.where((L >= lo) & (L <= hi) & (over == 0))[0] for _, lo, hi in BUCKETS]
    starts = [int(rng.choice(pools[k % 4] if len(pools[k % 4]) else pools[0])) for k in range(n)]
    per = []
    for k, ti in enumerate(starts):
        eng = common.engine_from_state(d["cls"][ti], d["order"][ti], d["dir"][ti])
        eng.rng = random.Random(seed * 100003 + k)
        arng = random.Random(seed * 1000 + k)
        grid = x14[ti].astype(np.int32).copy()
        mlen = len(eng.body)
        r = {"kind": kind, "start_length": mlen, "steps": 0, "ended": False, "first_mismatch": None, "len_at_first_mismatch": None,
             "first_invalid": None, "invalid_reason": None, "eat_events": 0, "food_on_snake_events": 0, "first_food_on_snake": None,
             "max_len_engine": mlen, "max_len_model": int((grid >= 2).sum()), "ascii": None, "rollout": k}
        for t in range(1, cap + 1):
            a = gen.pick("bfs", eng, arng) if kind == "bfs" else arng.choice(gen.safe_moves(eng) or [0, 1, 2, 3, 4])
            eaten = eng.step(a)
            det, fprob, p_ate, p_over = run_model(sess, grid, int(a))
            det = det.astype(np.uint8)
            food_e = None if (eaten or eng.food is None) else eng.food
            e14 = common.grid14(eng.body, eng.direction, food_e)
            det4 = TO4[det]
            mlen += 1 if eaten else 0
            r["steps"] = t
            r["max_len_engine"] = max(r["max_len_engine"], len(eng.body))
            r["max_len_model"] = max(r["max_len_model"], int((det >= 2).sum()))
            if not (eng.game_over and not eaten):
                reason = common.valid_state(det4, None, mlen)  # food_cell=None: food_on_snake is counted separately below
                if reason and r["first_invalid"] is None:
                    r["first_invalid"], r["invalid_reason"] = t, reason
            if eaten and eng.food is not None:
                r["eat_events"] += 1
                p = fprob.astype(np.float64)
                cell = int(nprng.choice(144, p=p / p.sum()))  # what the page would do: sample from the model's food head, write it, never check
                if det4.reshape(-1)[cell] >= 2:
                    r["food_on_snake_events"] += 1
                    if r["first_food_on_snake"] is None:
                        r["first_food_on_snake"] = t
            if r["first_mismatch"] is None and ((not np.array_equal(det, e14)) or ((p_over > 0.5) != eng.game_over)):
                r["first_mismatch"], r["len_at_first_mismatch"] = t, len(eng.body)
                r["ascii"] = {"step": t, "action": int(a), "engine_length": len(eng.body), "text": ascii_grids(e14, det)}
            if eng.game_over:
                r["ended"] = True
                break
            grid = det.astype(np.int32).copy()
            if eaten and eng.food is not None:
                grid[eng.food[0], eng.food[1]] = 1  # food sync: the engine's food replaces the model's guess
        per.append(r)
    return per


KM_HORIZONS = (50, 100, 200, 500, 1000)


def kaplan_meier(times, events, horizons=KM_HORIZONS):
    """Kaplan-Meier estimate of P(no mismatch yet), implemented directly (no libraries).
    times[i]: step of the first mismatch (event) or the step at which rollout i was censored (engine game over without a mismatch, or the cap).
    S(t) = product over event times u <= t of (1 - deaths(u) / at_risk(u)), at_risk(u) = number of rollouts with time >= u.
    Median = first event time where S < 0.5, or None ("not reached") if the curve never drops below 0.5."""
    event_times = sorted({t for t, e in zip(times, events) if e})
    curve, s = [], 1.0
    for u in event_times:
        at_risk = sum(1 for t in times if t >= u)
        deaths = sum(1 for t, e in zip(times, events) if e and t == u)
        s *= 1 - deaths / at_risk
        curve.append((u, s))

    def surv(h):
        last = 1.0
        for u, v in curve:
            if u > h:
                break
            last = v
        return last
    return {"S": {str(h): round(surv(h), 4) for h in horizons}, "rollouts_still_observed_at_step": {str(h): sum(1 for t in times if t >= h) for h in horizons},
            "median_step": next((u for u, v in curve if v < 0.5), None)}


def summarise(per, cap):
    pct = lambda v, q: float(np.percentile(v, q))
    mism = [r for r in per if r["first_mismatch"] is not None]
    matched_end = [r for r in per if r["first_mismatch"] is None and r["ended"]]
    cens_cap = [r for r in per if r["first_mismatch"] is None and not r["ended"]]
    v_end = [r["first_mismatch"] or r["steps"] for r in per]  # PLAN convention: matched-to-end counts as censored at its end step
    v_cap1 = [r["first_mismatch"] or cap + 1 for r in per]  # Run 1/2 convention: no mismatch -> cap + 1
    inv = [r for r in per if r["first_invalid"] is not None]
    v_inv = [r["first_invalid"] or r["steps"] for r in per]
    steps_total = sum((r["first_mismatch"] or r["steps"]) for r in per)
    hazard = len(mism) / max(steps_total, 1)
    p_roll = 1 - hazard
    lens = [r["len_at_first_mismatch"] for r in mism]
    eat_ev = sum(r["eat_events"] for r in per)
    fos = sum(r["food_on_snake_events"] for r in per)
    km = kaplan_meier([r["first_mismatch"] or r["steps"] for r in per], [r["first_mismatch"] is not None for r in per])
    km["median"] = km.pop("median_step")
    km["median"] = "not reached" if km["median"] is None else km["median"]
    return {"n": len(per), "cap": cap,
            "headline": {"mismatched": len(mism), "matched_to_end": len(matched_end), "alive_at_cap_no_mismatch": len(cens_cap),
                         "kaplan_meier_P_no_mismatch_yet": km,
                         "km_note": "a rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step"},
            "conventions_not_headline": {
                "note": "median conventions kept for reference only; neither is a headline",
                "first_mismatch_median_censored_at_end_step": pct(v_end, 50), "first_mismatch_p10_censored_at_end_step": pct(v_end, 10),
                "first_mismatch_median_nomismatch_as_cap_plus_1": pct(v_cap1, 50), "first_mismatch_p10_nomismatch_as_cap_plus_1": pct(v_cap1, 10),
                "first_invalid_median_censored_at_end_step": pct(v_inv, 50), "first_invalid_p10_censored_at_end_step": pct(v_inv, 10),
                "implied_median_from_rollout_hazard": round(math.log(0.5) / math.log(p_roll), 2) if 0 < p_roll < 1 else None},
            "dynamics_invalid": {"rollouts_with_invalid_state": len(inv),
                                 "reasons": {x: sum(1 for r in inv if r["invalid_reason"] == x) for x in DYNAMICS_REASONS},
                                 "note": "heads != 1, wrong length, disconnected body; food_on_snake is NOT included here"},
            "food_on_snake": {"eat_events": eat_ev, "events": fos, "pct_of_eat_events": round(100 * fos / max(eat_ev, 1), 3),
                              "rollouts_with_event": sum(1 for r in per if r["food_on_snake_events"]),
                              "note": "sampled from the model's food head and checked against the model's own predicted grid, as the page would write it"},
            "longest_snake_engine": max(r["max_len_engine"] for r in per), "longest_snake_model_predicted": max(r["max_len_model"] for r in per),
            "rollouts_engine_length_ge_64": sum(1 for r in per if r["max_len_engine"] >= 64),
            "length_at_first_mismatch": {"median": pct(lens, 50) if lens else None, "max": max(lens) if lens else None},
            "per_step_accuracy_from_rollouts": round(p_roll, 5)}


def pick_failures(per_all, k=3):
    recs = [r for r in per_all if r["ascii"]]
    chosen, seen = [], set()
    cand = [next((r for r in recs if r["kind"] == "bfs"), None), next((r for r in recs if r["kind"] == "random"), None),
            max(recs, key=lambda r: r["len_at_first_mismatch"]) if recs else None]
    for r in cand + recs:
        if r is not None and id(r) not in seen and len(chosen) < k:
            seen.add(id(r))
            chosen.append({"kind": r["kind"], "rollout": r["rollout"], **r["ascii"]})
    return chosen


# ----------------------------------------------------------------------------- driver
def evaluate_dataset(dataset, split, sess):
    t0 = time.time()
    d = load_split(dataset, split)
    x14, y14 = common.make_14(d)
    res = {"dataset": dataset, "split": split, "rows": int(len(d["act"])), "episodes": int(len(set(d["ep"].tolist())))}
    res["one_step"] = one_step(sess, d, x14, y14)
    res["rollout_cap"] = ROLLOUT_CAP
    pb = rollouts(sess, d, x14, "bfs", ROLLOUTS_PER_KIND, ROLLOUT_CAP, seed=1)
    pr = rollouts(sess, d, x14, "random", ROLLOUTS_PER_KIND, ROLLOUT_CAP, seed=2)
    res["rollout_bfs"], res["rollout_random_safe"] = summarise(pb, ROLLOUT_CAP), summarise(pr, ROLLOUT_CAP)
    res["rollout_all_200"] = summarise(pb + pr, ROLLOUT_CAP)
    res["failure_grids"] = pick_failures(pb + pr)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def md_dataset(title, r):
    o = r["one_step"]
    L = [f"## {title}", "", f"dataset `data/{r['dataset']}`, split `{r['split']}`, {r['rows']} rows from {r['episodes']} episodes ({r['seconds']} s).", "",
         "### One-step whole-board exact match (model / copy-previous-frame baseline; n)", "", "| length | all steps | eat steps | non-eat steps |", "|---|---|---|---|"]
    for b, row in o["exact"].items():
        L.append(f"| {b} | " + " | ".join(f"{row[s]['model_exact']} / {row[s]['copy14']} (n={row[s]['n']})" for s in ("all_steps", "eat_steps", "non_eat_steps")) + " |")
    L += ["", "### Per-class cell error (%)", "", "| class | cells | wrong (14-class) | type only | eat steps | non-eat steps |", "|---|---|---|---|---|---|"]
    for k, e in o["per_class_error"].items():
        L.append(f"| {k} | {e['cells']} | {e['pct_wrong_14class']} | {e['pct_wrong_type_only']} | {e['pct_wrong_14class_eat_steps']} | {e['pct_wrong_14class_non_eat_steps']} |")
    L += ["", f"Game over: {o['game_over']}", "", f"Ate head: {o['ate_head']}", "", f"Food head vs floor: {o['food_head']}", "",
          f"Implied median ln(0.5)/ln(p) from one-step p: {o['implied_median_from_one_step_p']}", "", f"### Rollouts (cap {r['rollout_cap']}, food synced)", ""]
    names = (("BFS", "rollout_bfs"), ("random non-suicidal", "rollout_random_safe"), ("combined", "rollout_all_200"))
    hs = [str(h) for h in KM_HORIZONS]
    L += ["Headline: counts, and the Kaplan-Meier estimate of P(no mismatch yet). A rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. "
          "KM median is \"not reached\" if the curve never drops below 0.5.", "",
          "| policy | rollouts | mismatched | matched to end | alive at cap | " + " | ".join(f"S({h})" for h in hs) + " | KM median |",
          "|---|---|---|---|---|" + "---|" * len(hs) + "---|"]
    for label, key in names:
        s = r[key]
        h, km = s["headline"], s["headline"]["kaplan_meier_P_no_mismatch_yet"]
        obs = km["rollouts_still_observed_at_step"]
        cells = [str(km["S"][x]) if obs[x] > 0 else "n/a (none observed)" for x in hs]  # no rollout reaches step h: S(h) would only repeat the last estimate
        L.append(f"| {label} | {s['n']} | {h['mismatched']} | {h['matched_to_end']} | {h['alive_at_cap_no_mismatch']} | " + " | ".join(cells) + f" | {km['median']} |")
    L += ["", "Rollouts still under observation at step h (not yet mismatched, ended or capped): " + "; ".join(
        f"{label}: " + ", ".join(f"{x}: {r[key]['headline']['kaplan_meier_P_no_mismatch_yet']['rollouts_still_observed_at_step'][x]}" for x in hs) for label, key in names), "",
          "| policy | rollouts with a dynamics-invalid state | reasons (heads != 1 / wrong length / disconnected) | food-on-snake events / eat events | rollouts with a food-on-snake event | "
          "longest snake (engine / model) | rollouts with engine length >= 64 | snake length at first mismatch (median / max) |", "|---|---|---|---|---|---|---|---|"]
    for label, key in names:
        s = r[key]
        di, fo, lm = s["dynamics_invalid"], s["food_on_snake"], s["length_at_first_mismatch"]
        L.append(f"| {label} | {di['rollouts_with_invalid_state']} | " + " / ".join(str(di["reasons"][x]) for x in DYNAMICS_REASONS) +
                 f" | {fo['events']} / {fo['eat_events']} ({fo['pct_of_eat_events']}%) | {fo['rollouts_with_event']} | {s['longest_snake_engine']} / {s['longest_snake_model_predicted']} | "
                 f"{s['rollouts_engine_length_ge_64']} | {lm['median']} / {lm['max']} |")
    L += ["", "Dynamics-invalid (heads, length, connectivity) and food-on-snake are reported separately. Food-on-snake = the cell sampled from the model's food head lies on a snake cell of the model's own predicted grid.",
          "The censored-at-end-step median, the cap+1 median and the rollout-hazard implied median are in the JSON under `conventions_not_headline` only."]
    L += ["", "### Failure grids (first mismatch; engine left, model right)", ""]
    for g in r["failure_grids"]:
        L += [f"{g['kind']} rollout {g['rollout']}, step {g['step']}, action {g['action']}, engine snake length {g['engine_length']}", "```", g["text"], "```", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dryrun", "heldout"])
    ap.add_argument("--confirm-heldout", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prov = provenance()
    sess = make_session()
    if args.mode == "dryrun":
        for ds in ("run1", "spike_a"):
            res = {"mode": "dryrun (VALIDATION rows only)", "provenance": prov, **evaluate_dataset(ds, "validation", sess)}
            json.dump(res, open(os.path.join(OUT, f"dryrun_{ds}_validation.json"), "w"), indent=1)
            open(os.path.join(OUT, f"dryrun_{ds}_validation.md"), "w").write("\n".join(md_dataset(f"DRY RUN on VALIDATION rows: data/{ds}", res)))
            print(f"[dryrun {ds}] rows={res['rows']} exact(all)={res['one_step']['exact']['all']['all_steps']['model_exact']} seconds={res['seconds']}", flush=True)
        return 0
    if not args.confirm_heldout:
        print("refusing: held-out evaluation needs --confirm-heldout")
        return 2
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # created BEFORE any held-out row is read
    except FileExistsError:
        print(f"refusing: {LOCK} exists, the held-out evaluation was already started. No second run.")
        return 3
    with os.fdopen(fd, "w") as f:
        json.dump({"started": prov["utc"], "provenance": prov}, f, indent=1)
    out = {"provenance": prov, "rollout_cap": ROLLOUT_CAP, "note": "held-out evaluated once; no changes to model, script or metrics after this run"}
    out["headline_run1_heldout"] = evaluate_dataset("run1", "heldout", sess)
    out["spike_a_heldout_seen_once_in_phase0_by_throwaway_models"] = evaluate_dataset("spike_a", "heldout", sess)
    json.dump(out, open(os.path.join(OUT, "results_heldout.json"), "w"), indent=1)
    md = ["# Held-out evaluation (run once)", "", f"Shipped model `web/model/snake.onnx` (sha256 {prov['sha256']['web/model/snake.onnx']}), onnxruntime {prov['onnxruntime_py']}, "
          f"script sha256 {prov['sha256']['scripts/eval_heldout.py']}, git HEAD {prov['git_head']}, {prov['utc']}.", ""]
    md += md_dataset("HEADLINE: data/run1 held-out (seed % 5 == 0, never seen by any model or script before this run)", out["headline_run1_heldout"])
    md += md_dataset("data/spike_a held-out (seen once in Phase 0 by throwaway models)", out["spike_a_heldout_seen_once_in_phase0_by_throwaway_models"])
    open(os.path.join(OUT, "results_heldout.md"), "w").write("\n".join(md))
    print("held-out evaluation done; wrote eval/results_heldout.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
