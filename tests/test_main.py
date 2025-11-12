import math
import pathlib
import sys

import torch
import torch.optim as optim

# Ensure project root is on sys.path so tests can import the top-level module `main`.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import MainModel, MetaOptimizer, nested_training_step


def test_meta_outputs_clamped():
    meta = MetaOptimizer()
    loss_tensor = torch.tensor(1.0)
    lr, momentum = meta(loss_tensor)
    assert lr.ndim == 0
    assert momentum.ndim == 0
    assert lr.item() >= 1e-8 and lr.item() <= 1e-2
    assert momentum.item() >= 0.0 and momentum.item() <= 0.99


def test_nested_unroll_shapes():
    model = MainModel(4, 1)
    meta = MetaOptimizer()
    optim_dummy = optim.SGD(model.parameters(), lr=1.0)

    data = torch.randn(1, 4)
    labels = data.sum(dim=1, keepdim=True)

    meta_loss, lr, momentum, grads = nested_training_step(
        model, meta, data, labels, optim_dummy
    )

    assert isinstance(meta_loss, torch.Tensor)
    assert isinstance(lr, torch.Tensor)
    assert isinstance(momentum, torch.Tensor)
    assert isinstance(grads, tuple)

    # grads should match number of parameters
    params = tuple(model.parameters())
    assert len(grads) == len(params)
    for g, p in zip(grads, params):
        assert g.shape == p.shape


def test_one_step_end_to_end_no_crash():
    model = MainModel(4, 1)
    meta = MetaOptimizer()
    meta_optim = optim.Adam(meta.parameters(), lr=1e-5)
    optim_dummy = optim.SGD(model.parameters(), lr=1.0)

    data = torch.randn(1, 4)
    labels = data.sum(dim=1, keepdim=True)

    meta_loss, lr, momentum, grads = nested_training_step(
        model, meta, data, labels, optim_dummy
    )

    # meta backward
    meta_optim.zero_grad()
    meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
    meta_optim.step()

    # apply real updates (should not crash)
    with torch.no_grad():
        for (name, param), g in zip(model.named_parameters(), grads):
            if g is not None:
                state = optim_dummy.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.data)
                buf = state["momentum_buffer"]
                buf.mul_(momentum.detach()).add_(g.detach())
                param.add_(buf, alpha=-lr.detach())

    # After one step, parameters should be finite numbers
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_multi_step_stability():
    """
    Run the stabilized training loop for a small number of steps and assert
    that losses remain finite and don't explode to NaN.
    """
    model = MainModel(4, 1)
    meta = MetaOptimizer()
    meta_optim = optim.Adam(meta.parameters(), lr=1e-5)
    optim_dummy = optim.SGD(model.parameters(), lr=1.0)

    steps = 10
    losses = []
    for _ in range(steps):
        data = torch.randn(1, 4)
        labels = data.sum(dim=1, keepdim=True) + torch.randn(1, 1) * 0.01

        meta_loss, lr, momentum, grads = nested_training_step(
            model, meta, data, labels, optim_dummy
        )

        # If meta_loss is not finite, fail early
        assert torch.isfinite(meta_loss).all()

        meta_optim.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=1.0)
        meta_optim.step()

        with torch.no_grad():
            for (name, param), g in zip(model.named_parameters(), grads):
                if g is not None:
                    state = optim_dummy.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param.data)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum.detach()).add_(g.detach())
                    param.add_(buf, alpha=-lr.detach())

        losses.append(float(meta_loss.item()))

    # Check that no NaNs were produced and losses stayed finite
    assert all(math.isfinite(x) for x in losses)
