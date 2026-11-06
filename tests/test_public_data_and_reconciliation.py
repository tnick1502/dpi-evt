from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from extval import kfs, leap, ngl
from analyze_controls import reconcile_curve, threshold_projected_hybrid, external_comparisons
from dpi_evt.training.losses import interpolated_crossing


def test_public_extract_readers_verify_embedded_checksums():
    assert len(kfs.read_cycles()) == 40
    assert len(leap.read_cycles()) == 23
    payload = ngl.read_payload()
    assert {key: len(value) for key, value in payload.items()} == {'ADA': 103, 'CHC': 42}


def test_reconciliations_do_not_read_future_labels():
    cycles = torch.tensor([[1., 10., 20., 100., 1000., 3000.]]).repeat(2, 1)
    split = dict(cycles=cycles, prefix_mask=torch.tensor([[1., 1., 0., 0., 0., 0.]]).repeat(2, 1),
                 prefix_obs=torch.tensor([[.02, .1, 0., 0., 0., 0.]]).repeat(2, 1))
    transformer = {'traj_mean': np.repeat(np.array([[.02, .1, .2, .5, .9, 1.]]), 2, axis=0)}
    gbm = dict(nliq=np.array([200., 3000.]), event_tail_mass=np.array([.1, .8]),
               risk_prob=np.array([.9, .2]))
    with_future = dict(split, label=torch.tensor([1., 0.]),
                       r_obs=torch.ones_like(cycles), n_liq_true=torch.tensor([15., 20.]))
    for function in (reconcile_curve, threshold_projected_hybrid):
        result = function(transformer, gbm, split)
        perturbed = function(transformer, gbm, with_future)
        np.testing.assert_array_equal(result['traj_mean'], perturbed['traj_mean'])
        r = result['traj_mean']
        assert ((r >= 0) & (r <= 1)).all()
        assert (np.diff(r, axis=1) >= 0).all()
        np.testing.assert_allclose(result['nliq'], interpolated_crossing(torch.tensor(r), cycles, .95).numpy())
        np.testing.assert_allclose(r[:, 1], .1)


def test_external_curve_bootstrap_ignores_only_zero_point_rows(monkeypatch):
    import analyze_controls as analysis
    monkeypatch.setattr(analysis, 'B', 1000)
    rows = []
    for fold in range(2):
        for model, multiplier in [('A', 1.), ('B', 4.)]:
            for i, (sse, points) in enumerate([(1., 2), (3., 2), (np.nan, 0)]):
                rows.append(dict(source='external', model=model, fold=fold, specimen_id=str(i),
                    liq_label=1, N_liq_true=20.+i, nliq_pred=21.+i, curve_crosses_threshold=1,
                    traj_sse_continuation=sse*multiplier, continuation_points=points))
    result = external_comparisons(pd.DataFrame(rows), [('A', 'B')])
    curve = result[result.metric.eq('curve_rmse')].iloc[0]
    np.testing.assert_allclose([curve.ratio, curve.ratio_lo, curve.ratio_hi], .5)
    assert curve.tests == 3 and curve.curve_tests_reference == 2
    with pytest.raises(ValueError, match='positive point count'):
        analysis.bootstrap_curve_rmse([np.nan], [1], np.ones((1, 1)))
