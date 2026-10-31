"""Compute-side speed variants on MPS (batch 256): channels_last, autocast fp16/bf16, width. Output: speed_v3_variants.json"""
import json, os, sys, time
import torch
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import v3
dev = torch.device("mps")
tr, va, x14t, y14t, *_ = v3.load_data(dev)
T = v3.make_device_data(tr, x14t, y14t, dev)
rows = []
def run(name, arch, C, cl, amp, B=256):
    torch.manual_seed(0)
    S = v3.Sampler(tr, dev, B); net = v3.Net3(arch, C=C).to(dev)
    if cl: net = net.to(memory_format=torch.channels_last)
    opt = v3.make_opt(net, 1e-4)
    def step():
        idx = S.idx(); Tb = {k: v[idx] for k, v in T.items()}
        if amp:
            with torch.autocast("mps", dtype=amp):
                loss = v3.loss_fn(net, Tb)
        else:
            loss = v3.loss_fn(net, Tb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); return loss
    try:
        for _ in range(10): step()
        v3.sync(dev); t = time.time()
        for _ in range(60): l = step()
        v3.sync(dev); sps = 60 * B / (time.time() - t)
        r = {"name": name, "arch": arch, "C": C, "channels_last": cl, "amp": str(amp), "batch": B, "samples_per_s": round(sps), "finite_loss": bool(torch.isfinite(l))}
    except Exception as e:
        r = {"name": name, "error": str(e)[:200]}
    rows.append(r); print(r, flush=True)
run("base dil C32", "dil", 32, False, None)
run("dil C32 channels_last", "dil", 32, True, None)
run("dil C32 fp16", "dil", 32, False, torch.float16)
run("dil C32 bf16", "dil", 32, False, torch.bfloat16)
run("dil C32 cl+fp16", "dil", 32, True, torch.float16)
run("dil C16", "dil", 16, False, None)
run("dil C64", "dil", 64, False, None)
run("plain11 C32 fp16", "plain11", 32, False, torch.float16)
run("k5 C32 fp16", "k5", 32, False, torch.float16)
json.dump({"command": "v3 bench_variants.py", "rows": rows}, open(os.path.join(HERE, "speed_v3_variants.json"), "w"), indent=1)
