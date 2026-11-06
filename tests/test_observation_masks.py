"""Observation masks separate binary outcomes from event-time censoring."""
import pytest
import torch

from dpi_evt.training.losses import (
    forecast_gaussian_nll,
    gaussian_nll,
    masked_bce_with_logits,
    nliq_censor_mask,
    risk_observation_mask,
    trajectory_supervision_masks,
)


def test_unresolved_censoring_is_not_a_binary_negative():
    batch = {
        "risk_label_observed": torch.tensor([1., 1., 0.]),
        "nliq_censor_valid": torch.tensor([1., 1., 1.]),
    }
    logits = torch.tensor([2., -2., 12.], requires_grad=True)
    loss = masked_bce_with_logits(
        logits, torch.tensor([1., 0., 0.]), risk_observation_mask(batch)
    )
    loss.backward()
    assert logits.grad[2] == 0
    assert nliq_censor_mask(batch)[2] == 1


def test_forecast_loss_uses_explicit_continuation_and_prefix_masks():
    batch = {
        "continuation_mask": torch.tensor([[0., 1., 1., 0.]]),
        "prefix_mask": torch.tensor([[1., 0., 0., 0.]]),
    }
    mean = torch.tensor([[2., 1., 3., 999.]])
    logvar = torch.zeros_like(mean)
    target = torch.zeros_like(mean)
    cont, prefix = trajectory_supervision_masks(batch)
    expected = gaussian_nll(mean, logvar, target, cont)
    expected += .15 * gaussian_nll(mean, logvar, target, prefix)
    torch.testing.assert_close(
        forecast_gaussian_nll(mean, logvar, target, batch), expected
    )


@pytest.mark.parametrize("helper,missing", [
    (risk_observation_mask, "risk_label_observed"),
    (nliq_censor_mask, "nliq_censor_valid"),
    (trajectory_supervision_masks, "continuation_mask"),
])
def test_batches_without_masks_are_rejected(helper, missing):
    with pytest.raises(KeyError, match=missing):
        helper({"n_liq_observed": torch.ones(2), "mask": torch.ones(2, 4)})


def test_empty_observed_outcome_batch_has_zero_loss_and_gradient():
    logits = torch.tensor([1., -1.], requires_grad=True)
    loss = masked_bce_with_logits(logits, torch.zeros(2), torch.zeros(2))
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros(2))
