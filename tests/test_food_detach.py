"""The food head must not shape the shared trunk when food_detach=True (Run 2 fix). Pure CPU, random weights, random input."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spikes", "spike_a"))
import v3  # noqa: E402

TRUNK = ("stem", "convs", "g1", "g2", "ate_mlp", "inj", "cls_head", "over_head")


def food_grads(food_detach):
    torch.manual_seed(0)
    net = v3.Net3("dil", ate=True, food_detach=food_detach)
    out = net(torch.rand(4, 19, 12, 12))
    out["food"].square().sum().backward()  # a loss made ONLY of the food output
    trunk = [p.grad for n, p in net.named_parameters() if n.split(".")[0] in TRUNK]
    head = net.food_head.weight.grad
    return trunk, head


def test_food_gradient_does_not_reach_trunk():
    trunk, head = food_grads(True)
    assert head is not None and head.abs().sum() > 0  # the food head itself still learns
    assert all(g is None or g.abs().sum() == 0 for g in trunk)


def test_control_without_detach_gradient_reaches_trunk():
    trunk, _ = food_grads(False)
    assert any(g is not None and g.abs().sum() > 0 for g in trunk)
