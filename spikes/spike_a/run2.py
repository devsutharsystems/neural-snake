"""Run 2: R1v2 + ate head, food head on stop-gradient trunk features, cosine LR to zero over the max run, wd 1e-4.
Model selection every 2.5 min on a fixed 5,000-row subset of the NEW-seed validation rows (data/run1, seed % 5 == 1; unused in Run 1);
best checkpoint -> checkpoints/run2_best.pt; early stop after 10 min without improvement; max 30 min.
Final evaluation on run2_best.pt only, on the 10,374 Spike A validation rows (as Run 1). Held-out rows are never loaded.

  .venv/bin/python spikes/spike_a/run2.py train [max_seconds]    (default 1800)
  .venv/bin/python spikes/spike_a/run2.py eval                    (final evaluation of checkpoints/run2_best.pt only)
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import common  # noqa: E402
import run1  # noqa: E402
import v3  # noqa: E402
from diag_run1 import head_losses  # noqa: E402

DEV = run1.DEV
SEL_EVERY_S, PATIENCE_S, SEL_ROWS = 150.0, 600.0, 5000
BEST = os.path.join(run1.CKPT_DIR, "run2_best.pt")


def new_net():
    return v3.Net3("dil", ate=True, food_detach=True).to(DEV).to(memory_format=torch.channels_last)


def load_all2():
    tr_old, va = common.load_trainval()  # Spike A validation (seed % 5 == 1): final evaluation only
    common.DATA = os.path.join(ROOT, "data", "run1")
    tr_new, va_new = common.load_trainval()  # va_new: new-seed validation rows, used for model selection only
    common.DATA = os.path.join(ROOT, "data", "spike_a")
    tr = {k: np.concatenate([tr_old[k], tr_new[k]]) for k in tr_old}
    return tr, va, va_new


def main_train(max_seconds):
    tr, va, va_new = load_all2()
    sel = np.sort(np.random.default_rng(0).choice(len(va_new["act"]), SEL_ROWS, replace=False))
    vs = {k: v[sel] for k, v in va_new.items()}
    del va_new
    x14t, y14t = common.make_14(tr)
    x14s, y14s = common.make_14(vs)
    x14v, y14v = common.make_14(va)
    print(f"train={len(tr['act'])} selection_rows(new-seed val)={SEL_ROWS} final_eval_rows={len(va['act'])}", flush=True)
    Tt, Ts = v3.make_device_data(tr, x14t, y14t, DEV), v3.make_device_data(vs, x14s, y14s, DEV)
    eat_s = vs["eaten"].astype(bool)
    torch.manual_seed(0)
    S = v3.Sampler(tr, DEV, 256)
    net = new_net()
    shifts = v3.prior_shifts(net, S, Tt, tr, DEV)
    st = {"best": -1.0, "best_t": 0.0, "curve": []}

    def cb(step, t, loss):
        a = run1.subset_acc(net, Ts, y14s, eat_s)
        hl = head_losses(net, Ts, SEL_ROWS)
        st["curve"].append({"t_s": round(t), "step": step, "train_loss": round(loss, 4), "sel5000": a, "sel_head_losses": hl})
        improved = a["overall"] > st["best"]
        if improved:
            st["best"], st["best_t"] = a["overall"], t
            run1.save_ckpt(net, "run2_best_tmp", {"step": step, "t_s": t, "sel_acc": a})
            os.replace(os.path.join(run1.CKPT_DIR, "run1_run2_best_tmp.pt"), BEST)
        print(f"t={t:.0f}s step={step} loss={loss:.4f} sel5000={a} heads={hl} {'BEST' if improved else ''}", flush=True)
        if t - st["best_t"] >= PATIENCE_S:
            print(f"early stop: no improvement for {PATIENCE_S:.0f}s (best {st['best']} at t={st['best_t']:.0f}s)", flush=True)
            return True
        return False
    step, el, loss = v3.run_training(net, Tt, S, max_seconds, DEV, 1e-4, log_every_s=SEL_EVERY_S, eval_cb=cb, cosine=True)
    if not st["curve"] or st["curve"][-1]["step"] != step:  # last partial interval: make sure the final weights were considered
        a = run1.subset_acc(net, Ts, y14s, eat_s)
        if a["overall"] > st["best"]:
            st["best"], st["best_t"] = a["overall"], el
            run1.save_ckpt(net, "run2_best_tmp", {"step": step, "t_s": el, "sel_acc": a})
            os.replace(os.path.join(run1.CKPT_DIR, "run1_run2_best_tmp.pt"), BEST)
        st["curve"].append({"t_s": round(el), "step": step, "train_loss": round(loss, 4), "sel5000": a, "note": "end of run"})
    print(f"training stopped t={el:.0f}s steps={step}; best sel5000 overall={st['best']} at t={st['best_t']:.0f}s", flush=True)
    info = {"command": f"run2.py train {max_seconds}", "n_train": int(len(tr["act"])), "n_val": int(len(va["act"])), "steps": step,
            "seconds": round(el, 1), "samples_per_s": round(step * 256 / el), "final_loss": round(loss, 4), "prior_shifts": shifts,
            "selection": {"rows": SEL_ROWS, "source": "new-seed validation (data/run1, seed % 5 == 1)", "best_sel_acc": st["best"],
                          "best_at_s": round(st["best_t"], 1), "curve": st["curve"]}}
    json.dump(info, open(os.path.join(HERE, "run2_train_info.json"), "w"), indent=1)
    final_eval(va, x14v, y14v, info)


def final_eval(va, x14v, y14v, info):
    Tv = v3.make_device_data(va, x14v, y14v, DEV)
    net = new_net()
    ck = torch.load(BEST)
    net.load_state_dict(ck["state_dict"])
    info = dict(info, evaluated_checkpoint="checkpoints/run2_best.pt", checkpoint_step=ck.get("step"), checkpoint_t_s=round(ck.get("t_s", 0), 1))
    run1.evaluate(net, va, Tv, x14v, y14v, info, tag="run2")


def main_eval():
    _, va = common.load_trainval()
    x14v, y14v = common.make_14(va)
    final_eval(va, x14v, y14v, {"command": "run2.py eval"})


if __name__ == "__main__":
    if sys.argv[1] == "train":
        main_train(float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0)
    else:
        main_eval()
