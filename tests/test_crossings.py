"""Crossing interpolation and tail regularization of completed non-event records."""

import torch

from dpi_evt.models import DPIEvtNet
from dpi_evt.training.losses import crossing_margin_loss, interpolated_crossing


def test_interpolated_crossing_is_exact_in_forward_and_differentiable():
    r = torch.tensor([[0.94, 0.96]], requires_grad=True)
    cycles = torch.tensor([[100.0, 200.0]])

    nliq = interpolated_crossing(r, cycles, threshold=0.95)
    assert torch.allclose(nliq, torch.tensor([150.0]), atol=1e-4)

    nliq.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()
    assert r.grad.abs().sum() > 0


def test_interpolated_crossing_has_gradient_before_crossing():
    r = torch.tensor([[0.10, 0.20, 0.40]], requires_grad=True)
    cycles = torch.tensor([[1.0, 10.0, 100.0]])

    nliq = interpolated_crossing(r, cycles, threshold=0.95)
    assert torch.allclose(nliq, torch.tensor([100.0]))

    nliq.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()
    assert r.grad.abs().sum() > 0


def test_interpolated_crossing_handles_first_node_and_nonmonotone_input():
    cycles = torch.tensor([[5.0, 10.0, 20.0], [5.0, 10.0, 20.0]])
    r = torch.tensor([[0.97, 0.98, 0.99], [0.90, 0.96, 0.94]])
    nliq = interpolated_crossing(r, cycles, threshold=0.95)
    assert torch.allclose(nliq[0], torch.tensor(5.0))
    assert torch.allclose(nliq[1], torch.tensor(5.0 + 5.0 * (0.05 / 0.06)), atol=1e-4)


def test_crossing_margin_penalizes_only_stable_nonliq():
    traj = torch.tensor(
        [[0.10, 0.99], [0.10, 0.99], [0.10, 0.99]],
        requires_grad=True,
    )
    label = torch.tensor([0.0, 0.0, 1.0])
    stable = torch.tensor([1.0, 0.0, 0.0])
    observed = torch.tensor([1.0, 0.0, 1.0])

    loss = crossing_margin_loss(
        traj, label, threshold=0.95, observed=observed, stable=stable,
    )
    assert torch.allclose(loss, torch.tensor(0.04), atol=1e-6)

    loss.backward()
    assert traj.grad[0].abs().sum() > 0
    assert traj.grad[1].abs().sum() == 0 # unfinished non-liq
    assert traj.grad[2].abs().sum() == 0 # liquefied records are not part of the completed-non-event penalty


def test_dpi_evt_reports_curve_crossing_and_keeps_auxiliary_head():
    model = DPIEvtNet(
        static_dim=2,
        prefix_dim=1,
        seq_dim=5,
        seq_len=4,
        prefix_len=2,
        max_cycle_reference=100.0,
        hidden_dim=16,
        probabilistic=False,
        use_flow=False,
        report_nliq_from_curve=True,
    ).eval()
    batch = {
        "static": torch.zeros(3, 2),
        "prefix_summary": torch.zeros(3, 1),
        "prefix_obs": torch.zeros(3, 4),
        "prefix_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(3, 1),
        "seq_in": torch.zeros(3, 4, 5),
        "cycles": torch.tensor([[1.0, 10.0, 40.0, 100.0]]).repeat(3, 1),
        "delta_cycles": torch.tensor([[0.0, 9.0, 30.0, 60.0]]).repeat(3, 1),
        "csr": torch.full((3, 4), 0.15),
    }

    with torch.no_grad():
        out = model.forward_batch(batch)

    assert "nliq_norm_head" in out
    assert torch.allclose(out["nliq_norm"], out["nliq_norm_curve"])
    expected = torch.expm1(out["nliq_norm_curve"] * torch.log1p(torch.tensor(100.0)))
    assert torch.allclose(out["nliq"], expected, atol=1e-5)


