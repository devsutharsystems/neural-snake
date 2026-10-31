"""Export checkpoints/run2_best.pt to web/model/snake.onnx (single file, weights embedded).

Graph inputs : grid   int32 [1,12,12]  14-class grid (0 empty, 1 food, 2-5 head, 6-9 body, 10-13 tail; spikes/spike_a/common.py)
               action int32 [1]        0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 NOOP
Graph outputs: next_grid int64 [1,12,12] argmax class per cell (BigInt64Array in JS; the pages convert with Int32Array.from(data, Number))
               food      float32 [1,144] softmax over cells (used only when ate > 0.5)
               ate       float32 [1]     sigmoid(ate logit - ate_shift)   (prior-shift buffer included)
               over      float32 [1]     sigmoid(over logit - over_shift) (prior-shift buffer included)
The one-hot of grid and action is built inside the graph. Run: .venv/bin/python scripts/export_onnx.py -> web/model/snake.onnx, web/model/export_info.json
"""
import collections
import json
import os
import sys

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "spikes", "spike_a"))
import v3  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "run2_best.pt")
OUT = os.path.join(ROOT, "web", "model", "snake.onnx")
N = 12


def load_net():
    net = v3.Net3("dil", ate=True, food_detach=True)
    net.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    return net.eval()


class Wrapped(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("cls_ids", torch.arange(14, dtype=torch.int32).view(1, 14, 1, 1))
        self.register_buffer("act_ids", torch.arange(5, dtype=torch.int32).view(1, 5))

    def forward(self, grid, action):
        x_grid = (grid.unsqueeze(1) == self.cls_ids).float()  # [1,14,12,12]
        x_act = (action.unsqueeze(1) == self.act_ids).float().view(-1, 5, 1, 1).expand(-1, 5, N, N)
        out = self.net(torch.cat([x_grid, x_act], 1))
        # next_grid stays ArgMax's int64 output: a following Cast(int64->int32) ran on the CPU EP under WebGPU (Chrome placement log), so JS converts instead
        return (out["cls"].argmax(1), torch.softmax(out["food"], 1), torch.sigmoid(out["ate"]), torch.sigmoid(out["over"]))


def main():
    net = load_net()
    w = Wrapped(net).eval()
    grid = torch.zeros(1, N, N, dtype=torch.int32)
    action = torch.zeros(1, dtype=torch.int32)
    opset = int(os.environ.get("OPSET", "18"))
    torch.onnx.export(w, (grid, action), OUT, input_names=["grid", "action"], output_names=["next_grid", "food", "ate", "over"],
                      dynamo=True, opset_version=opset)
    m = onnx.load(OUT)
    onnx.save(m, OUT, save_as_external_data=False)  # embed weights: single-file model for the browser
    if os.path.exists(OUT + ".data"):
        os.remove(OUT + ".data")
    m = onnx.load(OUT)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    info = {"command": "scripts/export_onnx.py", "checkpoint": "checkpoints/run2_best.pt (git-ignored)", "torch": torch.__version__,
            "onnx": onnx.__version__, "onnxruntime_py": ort.__version__, "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
            "file_bytes": os.path.getsize(OUT), "params": sum(p.numel() for p in net.parameters()),
            "over_shift": float(net.over_shift), "ate_shift": float(net.ate_shift), "op_counts": dict(sorted(ops.items())),
            "inputs": [(i.name, i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
            "outputs": [(o.name, o.type.tensor_type.elem_type, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}
    json.dump(info, open(os.path.join(ROOT, "web", "model", "export_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
