"""Quantile ranks, censoring, support limits and split separation for post-hoc intervals."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_intervals import (OnsetCalibrator, corrected_quantile, assert_disjoint,
                                        interval_rows, normalized_outputs, MODELS)


def test_corrected_rank_not_interpolated_quantile():
    assert corrected_quantile(np.arange(20), .95) == 19
    assert np.isinf(corrected_quantile(np.arange(18), .95))
    assert np.isinf(corrected_quantile([], .95))


def test_known_asymmetric_errors_and_censored_exclusion():
    point = np.full(100, 100.)
    residual = np.linspace(-.2, .6, 100)
    truth = np.expm1(np.log1p(point) + residual)
    fitted = OnsetCalibrator.fit(np.r_[truth, np.inf], np.r_[point, np.nan], np.r_[np.ones(100), 0])
    assert fitted.n_events == 100
    assert fitted.lower_log_offset == pytest.approx(residual[4])
    assert fitted.upper_log_offset == pytest.approx(residual[95])
    lo, hi = fitted.predict([100.])
    assert lo[0] == pytest.approx(truth[4])
    assert hi[0] == pytest.approx(truth[95])


def test_empty_or_small_calibration_is_explicitly_vacuous():
    for n in (0, 18):
        c = OnsetCalibrator.fit(np.full(n, 100), np.full(n, 100), np.ones(n))
        lo, hi = c.predict([0, 100, np.inf])
        np.testing.assert_allclose(lo, [10, 10, 10])
        assert np.isposinf(hi).all()
        assert c.parameters()["vacuous"]
        json.dumps(c.parameters(), allow_nan=False)


def test_horizon_tail_is_not_finite_clipping():
    c = OnsetCalibrator(.9, -.5, .5, 100)
    lo, hi = c.predict([3000, np.inf, 10])
    assert np.isposinf(hi[:2]).all()
    assert (lo[:2] < 3000).all()
    assert lo[2] == pytest.approx(10)
    exact = OnsetCalibrator(.9, 0, 0, 100)
    assert np.isfinite(exact.predict([3000])[1][0])


@pytest.mark.parametrize("truth,point,event", [([0], [100], [1]), ([100], [np.nan], [1]),
    ([100], [100], [2]), ([100, 200], [100], [1])])
def test_invalid_calibration_rejected(truth, point, event):
    with pytest.raises(ValueError):
        OnsetCalibrator.fit(truth, point, event)


def test_site_overlap_rejected_even_with_different_specimens():
    a = pd.DataFrame(dict(specimen_id=["a"], site_id=["same"]))
    b = pd.DataFrame(dict(specimen_id=["b"], site_id=["same"]))
    with pytest.raises(ValueError, match="site_id"):
        assert_disjoint(a, b)


def test_missing_native_event_intervals_are_not_fabricated():
    a = dict(nliq=np.array([100.]), truth=np.array([110.]), event=np.array([1]),
             specimen_id=["test"], site_id=["site"], programme=["storm"])
    d = interval_rows(a, [OnsetCalibrator(.9, -.2, .2, 100),
                          OnsetCalibrator(.95, -.3, .3, 100)], "transformer", 0, "internal")
    assert len(d) == 2 and set(d.method) == {"validation_log_asymmetric"}
    assert not d.native_available.any()
    assert set(d.target) == {.90, .95}


def test_normalized_head_is_resolved_without_changing_native_bounds():
    out = {"nliq_pred": np.array([.5], np.float32),
           "nliq_q05": np.array([20.]), "nliq_q95": np.array([np.inf])}
    result = normalized_outputs(out, "cox_ph", {})
    assert result["nliq"][0] == pytest.approx(np.sqrt(3001)-1, rel=1e-6)
    np.testing.assert_array_equal(result["nliq_q95"], out["nliq_q95"])
    assert "nliq" not in out


def test_cummax_uses_curve_crossing_instead_of_raw_head():
    import torch
    cycles = torch.tensor([[1., 10., 100., 1000.]])
    out = {"traj_mean": np.array([[.1, .8, .7, 1.]], np.float32),
           "nliq_pred": np.array([.1], np.float32)}
    result = normalized_outputs(out, "transformer_cummax", {"cycles": cycles})
    assert result["nliq"][0] == pytest.approx(775., abs=.001)
    assert "nliq_pred" not in result


