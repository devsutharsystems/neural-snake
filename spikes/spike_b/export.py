"""Spike B: export a random-weights Net (same shapes/size as Spike A R1) to ONNX and check vs onnxruntime."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "spike_a"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from train_eval import Net  # noqa: E402
import onnx, onnxruntime as ort  # noqa: E402

torch.manual_seed(0)
net = Net(14).eval()
out_path = os.path.join(HERE, "random_net.onnx")
x = torch.randn(1, 14, 12, 12)
prog = torch.onnx.export(net, (x,), out_path, input_names=["x"],
                         output_names=["cls", "order", "food", "dir", "over"], dynamo=True)
m = onnx.load(out_path)  # loads external data too
onnx.save(m, out_path, save_as_external_data=False)  # embed weights: single-file model for the browser
if os.path.exists(out_path + ".data"): os.remove(out_path + ".data")
opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
diffs = []
for i in range(10):
    xi = torch.randn(1, 14, 12, 12, generator=torch.Generator().manual_seed(100 + i))
    with torch.no_grad():
        ref = net(xi)
    got = sess.run(None, {"x": xi.numpy()})
    diffs.append(max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got)))
res = {"exporter": "torch.onnx.export(dynamo=True)", "torch": torch.__version__, "onnx_opset": opset,
       "file_bytes": os.path.getsize(out_path), "params": sum(p.numel() for p in net.parameters()),
       "max_abs_diff_over_10_inputs": max(diffs), "per_input": diffs, "onnxruntime_py": ort.__version__}
json.dump(res, open(os.path.join(HERE, "export_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
