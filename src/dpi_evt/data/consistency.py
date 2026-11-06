"""Consistency checks for inputs and prepared data/dataset artifacts."""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from dpi_evt.data.grainsize import FRACTION_KEYS, plaxis_classification
from dpi_evt.physics.g0 import vs_from_g0

FORBIDDEN_MODEL_FEATURES = frozenset({"CSR_max", "nonstationarity", "N_max"})


def validate_forecasting_input_contract(population: dict) -> None:
    """Raise an error if a dataset uses inputs that are unavailable at forecast time.

    Full prescribed ``CSR(N)`` is allowed as an exogenous sequential control. Scalar summaries
    computed from the completed loading/response record are not allowed in the static vector.
    """
    names = list(population.get("static_feature_names", []))
    leaked = FORBIDDEN_MODEL_FEATURES.intersection(names)
    if leaked:
        raise RuntimeError(
            f"stale/leaky dataset: forbidden static features {sorted(leaked)}; "
            "run `scripts/run_experiments.py prepare` before training"
        )
    csr = np.asarray(population.get("csr"))
    seq = np.asarray(population.get("seq_inputs"))
    seq_names = list(population.get("seq_feature_names", []))
    if csr.ndim != 2 or seq.ndim != 3 or seq.shape[:2] != csr.shape:
        raise RuntimeError("dataset has an invalid CSR(N)/seq_inputs shape; rerun data preparation")
    if not seq_names or seq_names[0] != "CSR" or not np.allclose(seq[..., 0], csr, equal_nan=False):
        raise RuntimeError("the prescribed CSR(N) program is not preserved as seq_inputs[..., 0]")
    if not np.isfinite(csr).all():
        raise RuntimeError("the prescribed CSR(N) program contains NaN/inf")
    meta = population.get("meta")
    if not isinstance(meta, pd.DataFrame) or "csr_program_known_a_priori" not in meta.columns:
        raise RuntimeError(
            "dataset lacks csr_program_known_a_priori provenance; run `scripts/run_experiments.py prepare`"
        )
    if not bool(meta["csr_program_known_a_priori"].fillna(False).all()):
        raise RuntimeError("CSR(N) must be known before the response for every forecasting sample")


def check_artifact_consistency(src_dir: str = "data/dataset") -> Tuple[bool, List[str]]:
    """Check a prepared dataset and return (success, report_lines).

    The checks cover excluded latent or leaking inputs (risk_score_true, OCR), the DigitRock relation
    Vs = sqrt(G0*1000/rho), velocity bounds, grain-size-derived PLAXIS class,
    finite static features and the presence of measured CRR targets.
    """
    m = pd.read_parquet(os.path.join(src_dir, "meta.parquet"))
    z = np.load(os.path.join(src_dir, "arrays.npz"))
    fn = json.load(open(os.path.join(src_dir, "feature_names.json")))
    names = fn["static_feature_names"] if isinstance(fn, dict) else fn
    N = len(m)
    report: List[str] = [f"Dataset: {N} specimens, {m.shape[1]} metadata columns"]
    checks: List[bool] = []

    def chk(cond: bool, msg: str):
        checks.append(bool(cond))
        report.append(("OK " if cond else "FAIL ") + msg)

    chk("risk_score_true" not in m.columns, "risk_score_true is absent from metadata")
    chk(not any(c.upper() == "OCR" for c in m.columns), "OCR is absent from metadata")
    chk(not any("OCR" in str(x) for x in names), "OCR is absent from static features")
    chk("PPR_max_true" in m.columns, "PPR_max_true is present as a diagnostic field")
    chk(not FORBIDDEN_MODEL_FEATURES.intersection(names),
        "CSR_max, nonstationarity and N_max are absent from static features")
    chk("csr_program_known_a_priori" in m.columns and bool(m.get("csr_program_known_a_priori", False).all()),
        "CSR(N) is marked as a loading program known in advance")

    rho = m["r"].to_numpy(float); vs_arr = m["V_s"].to_numpy(float)
    g0_recon = rho * vs_arr ** 2 / 1000.0
    vs_err = float(np.nanmax(np.abs(vs_from_g0(g0_recon, rho) - vs_arr)))
    chk(vs_err < 1.0, f"Vs == sqrt(G0·1000/ρ) (digitrock), max|Δ|={vs_err:.3f} m/s")
    vs = m["V_s"].astype(float)
    chk((vs.min() >= 40) and (vs.max() < 600), f"Vs range {vs.min():.0f}-{vs.max():.0f} m/s (median {vs.median():.0f})")
    chk(np.isfinite(g0_recon).all() and (g0_recon > 0).all(), f"G0 is defined for all specimens (median {np.median(g0_recon):.1f} MPa)")

    gcols = [f"gran_{k}" for k in FRACTION_KEYS]
    fr = m[gcols].to_numpy(float); has = fr.sum(1) > 1.0
    pc = plaxis_classification(fr)
    agree = (pd.Series(pc["plaxis_class"])[has].to_numpy() == m["plaxis_class"][has].to_numpy()).mean() if has.sum() else 1.0
    chk(agree > 0.999, f"plaxis_class agrees for {int(has.sum())} specimens with grain-size data ({agree*100:.1f}%)")
    cu_chk = pc["D60"] / np.maximum(pc["D10"], 1e-9)
    cu_err = float(np.nanmax(np.abs(cu_chk[has] - m["Cu"].to_numpy(float)[has]))) if has.sum() else 0.0
    chk(cu_err < 0.5, f"Cu == D60/D10, max|Δ|={cu_err:.3f}")

    sf = z["static_features"]
    chk(np.isfinite(sf).all(), f"static_features contain no NaN/inf {sf.shape}")
    chk("crr_obs" in z and z["crr_obs_mask"].sum() > 0, f"CRR(N): {int(z['crr_obs_mask'].sum())}/{N} specimens")

    report.append("plaxis_class: " + str(m["plaxis_class"].value_counts().to_dict()))
    report.append("DONE")
    return all(checks), report
